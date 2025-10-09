#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数字人流式服务 - WebRTC版本
接收麦克风音频 -> AI推理 -> 实时返回视频流
"""

import asyncio
import argparse
import base64
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import cv2
import numpy as np
from aiohttp import web

# WebRTC相关
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder
import av

# 音频处理
try:
    from scipy.signal import resample_poly

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，将使用简单重采样")

# 导入您的处理器
from dihuman_run import DiHumanProcessor as WenetProcessor

if TYPE_CHECKING:
    from dihuman_run_hubert import DiHumanProcessorHubert
    from dihuman_run_ave import DiHumanProcessorAve

ProcessorType = Union[WenetProcessor, "DiHumanProcessorHubert", "DiHumanProcessorAve"]

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_filename(name: str, suffix: str = ".mp4") -> str:
    """Remove path separators and ensure the file has the desired suffix."""
    if not name:
        name = f"synth_{int(time.time())}"
    name = name.replace("\\", "_").replace("/", "_").strip()
    if not name.lower().endswith(suffix):
        name = f"{name}{suffix}"
    return name


def _load_wav_file(path: Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit PCM WAV file and return mono samples with sample rate."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV is supported")
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        audio_bytes = wf.readframes(frames)

    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).astype(np.float32)
        audio = np.mean(audio, axis=1)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio, sample_rate


def _synthesize_tts_to_wav(text: str, output_path: Path) -> None:
    try:
        import pyttsx3  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pyttsx3 is required for text-to-speech. Install it with 'pip install pyttsx3'."
        ) from exc

    engine = pyttsx3.init()
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    engine.stop()


def _create_processor(asr_mode: str, data_path: Path) -> ProcessorType:
    """Instantiate the requested audio front-end."""
    mode = asr_mode.lower()
    if mode == "hubert":
        logger.info("Loading HuBERT processor")
        from dihuman_run_hubert import DiHumanProcessorHubert  # delayed import
        return DiHumanProcessorHubert(str(data_path))
    if mode == "ave":
        logger.info("Loading AVE processor")
        from dihuman_run_ave import DiHumanProcessorAve  # delayed import
        return DiHumanProcessorAve(str(data_path))
    if mode == "wenet":
        return WenetProcessor(str(data_path))
    raise ValueError(f"Unsupported ASR mode: {asr_mode}")


def _run_offline_synthesis(
    audio_16k: np.ndarray,
    data_path: Path,
    fps: int,
    work_dir: Path,
    asr_mode: str,
) -> dict:
    """Generate talking-head video from 16kHz mono audio samples."""
    processor = _create_processor(asr_mode, data_path)

    frames = []
    audio_chunks = []
    chunk_size = 160  # 10ms @ 16kHz

    total_samples = int(audio_16k.shape[0])
    for start in range(0, total_samples, chunk_size):
        chunk = audio_16k[start:start + chunk_size]
        if chunk.shape[0] < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - chunk.shape[0]))

        img, playing_audio, check_img = processor.process(chunk)

        if playing_audio is not None and playing_audio.size:
            audio_chunks.append(playing_audio.copy())

        if check_img and img is not None:
            frames.append(img.copy())

    # Flush remaining audio/playback buffers with silence
    for _ in range(50):
        if getattr(processor, "audio_play_list", None) == []:
            break
        img, playing_audio, check_img = processor.process(np.zeros(chunk_size, dtype=np.int16))
        if playing_audio is not None and playing_audio.size:
            audio_chunks.append(playing_audio.copy())
        if check_img and img is not None:
            frames.append(img.copy())

    if not frames:
        raise RuntimeError("No video frames were produced. Ensure the models and data_path are correct.")

    height, width = frames[0].shape[:2]
    video_path = work_dir / "synth_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer. Check OpenCV codecs support.")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    if audio_chunks:
        audio_out = np.concatenate(audio_chunks).astype(np.int16)
    else:
        audio_out = np.zeros(chunk_size, dtype=np.int16)

    audio_path = work_dir / "synth_audio.wav"
    with contextlib.closing(wave.open(str(audio_path), "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_out.tobytes())

    muxed_path = None
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        muxed_path = work_dir / "synth_output.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(muxed_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - diagnostics only
            muxed_path = None
            logger.warning("ffmpeg mux failed: %s", exc.stderr.decode(errors="ignore")[:500])
    else:  # pragma: no cover - environment dependent
        logger.warning("ffmpeg not found; returning separate audio/video files.")

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "muxed_path": muxed_path,
    }


class AudioResampler:
    """音频重采样工具类"""

    @staticmethod
    def resample_to_16k(samples: np.ndarray, in_rate: int) -> np.ndarray:
        """将音频重采样到16kHz"""
        if in_rate == 16000:
            return samples.astype(np.int16)

        if HAS_SCIPY:
            # 高质量重采样
            out = resample_poly(samples.astype(np.float32), 16000, in_rate)
        else:
            # 简单线性重采样
            ratio = 16000.0 / float(in_rate)
            src_idx = np.arange(len(samples))
            dst_len = int(len(samples) * ratio)
            dst_idx = np.linspace(0, len(samples) - 1, dst_len)
            out = np.interp(dst_idx, src_idx, samples.astype(np.float32))

        return np.clip(out, -32768, 32767).astype(np.int16)

    @staticmethod
    def resample_from_16k(samples: np.ndarray, out_rate: int) -> np.ndarray:
        """从16kHz重采样到目标采样率"""
        if out_rate == 16000:
            return samples.astype(np.int16)

        if HAS_SCIPY:
            out = resample_poly(samples.astype(np.float32), out_rate, 16000)
        else:
            # 简单重复采样
            factor = out_rate // 16000
            out = np.repeat(samples, factor)

        return np.clip(out, -32768, 32767).astype(np.int16)


class ProcessorBridge:
    """处理器桥接层 - 连接WebRTC和DiHumanProcessor"""

    def __init__(self, processor: ProcessorType, fps: int = 25):
        self.processor = processor
        self.fps = fps

        # 音频缓冲
        self.audio_buffer = deque()
        self.audio_buffer_lock = asyncio.Lock()

        # 视频帧缓存
        self.latest_frame = None
        self.frame_lock = asyncio.Lock()

        # 音频输出缓冲
        self.audio_out_buffer = deque()
        self.audio_out_lock = asyncio.Lock()

        # 控制标志
        self._running = False
        self._task = None

    async def start(self):
        """启动处理循环"""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("处理器桥接启动")

    async def stop(self):
        """停止处理循环"""
        self._running = False
        if self._task:
            await self._task
        logger.info("处理器桥接停止")

    async def push_audio(self, audio_16k: np.ndarray):
        """推送16kHz音频到缓冲区"""
        async with self.audio_buffer_lock:
            self.audio_buffer.append(audio_16k)

    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的视频帧"""
        async with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    async def get_audio_out(self, samples_needed: int) -> np.ndarray:
        """获取输出音频"""
        async with self.audio_out_lock:
            if not self.audio_out_buffer:
                return np.zeros(samples_needed, dtype=np.int16)

            # 收集足够的音频样本
            collected = []
            collected_samples = 0

            while self.audio_out_buffer and collected_samples < samples_needed:
                chunk = self.audio_out_buffer.popleft()
                collected.append(chunk)
                collected_samples += len(chunk)

            if collected:
                audio = np.concatenate(collected)
                if len(audio) > samples_needed:
                    # 多余的放回缓冲区
                    self.audio_out_buffer.appendleft(audio[samples_needed:])
                    audio = audio[:samples_needed]
                elif len(audio) < samples_needed:
                    # 不足的部分补零
                    audio = np.pad(audio, (0, samples_needed - len(audio)))
                return audio
            else:
                return np.zeros(samples_needed, dtype=np.int16)

    async def _process_loop(self):
        """主处理循环"""
        accumulated_audio = np.array([], dtype=np.int16)

        while self._running:
            # 从缓冲区获取音频
            async with self.audio_buffer_lock:
                if not self.audio_buffer:
                    await asyncio.sleep(0.001)
                    continue

                # 收集所有可用音频
                chunks = []
                while self.audio_buffer:
                    chunks.append(self.audio_buffer.popleft())

            if chunks:
                new_audio = np.concatenate(chunks)
                accumulated_audio = np.concatenate([accumulated_audio, new_audio])

            # 按10ms块处理音频
            while len(accumulated_audio) >= 160:  # 10ms @ 16kHz
                audio_frame = accumulated_audio[:160]
                accumulated_audio = accumulated_audio[160:]

                try:
                    # 调用处理器
                    img, playing_audio, check_img = self.processor.process(audio_frame)

                    # 更新视频帧
                    if check_img and img is not None:
                        async with self.frame_lock:
                            self.latest_frame = img.copy()

                    # 保存输出音频
                    if playing_audio is not None:
                        async with self.audio_out_lock:
                            self.audio_out_buffer.append(playing_audio.copy())

                            # 防止缓冲区过大
                            if len(self.audio_out_buffer) > 100:  # 约1秒缓冲
                                self.audio_out_buffer.popleft()

                except Exception as e:
                    logger.error(f"处理器错误: {e}")

            await asyncio.sleep(0.001)


class VideoOutputTrack(MediaStreamTrack):
    """视频输出轨道"""
    kind = "video"

    def __init__(self, bridge: ProcessorBridge, fps: int = 25):
        super().__init__()
        self.bridge = bridge
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_time = time.time()

    async def recv(self):
        """生成视频帧"""
        # 控制帧率
        current_time = time.time()
        sleep_time = self.frame_interval - (current_time - self.last_time)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        self.last_time = time.time()

        # 获取最新帧
        frame_data = await self.bridge.get_latest_frame()

        if frame_data is None:
            # 生成黑色占位帧
            frame_data = np.zeros((720, 1280, 3), dtype=np.uint8)
            import cv2
            cv2.putText(frame_data, "Waiting for audio input...",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 转换为视频帧
        frame = av.VideoFrame.from_ndarray(frame_data, format="bgr24")
        frame.pts = frame.time_base = None

        return frame


class AudioOutputTrack(MediaStreamTrack):
    """音频输出轨道"""
    kind = "audio"

    def __init__(self, bridge: ProcessorBridge, sample_rate: int = 48000):
        super().__init__()
        self.bridge = bridge
        self.sample_rate = sample_rate
        self.samples_per_frame = int(sample_rate * 0.02)  # 20ms帧

    async def recv(self):
        """生成音频帧"""
        # 获取16kHz音频
        audio_16k = await self.bridge.get_audio_out(320)  # 20ms @ 16kHz

        # 重采样到48kHz
        audio_48k = AudioResampler.resample_from_16k(audio_16k, self.sample_rate)

        # 创建音频帧
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(audio_48k))
        frame.planes[0].update(audio_48k.tobytes())
        frame.sample_rate = self.sample_rate

        return frame


# 全局连接管理
peer_connections = set()


async def index(request):
    """返回客户端HTML页面"""
    content = open('client.html', 'r', encoding='utf-8').read()
    return web.Response(text=content, content_type='text/html')


async def offer(request):
    """处理WebRTC连接请求"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 获取配置
    data_path: Path = request.app["data_path"]
    fps = request.app["fps"]
    asr_mode = request.app["asr_mode"]

    # 创建连接
    pc = RTCPeerConnection()
    peer_connections.add(pc)

    # 为每个连接创建独立的处理器
    processor = _create_processor(asr_mode, data_path)
    bridge = ProcessorBridge(processor, fps)
    await bridge.start()

    # 存储bridge引用，确保不被垃圾回收
    pc.bridge = bridge

    @pc.on("track")
    async def on_track(track):
        """处理接收到的音频轨道"""
        if track.kind == "audio":
            logger.info(f"接收到音频轨道: {track}")

            async def audio_consumer():
                """消费音频数据"""
                while True:
                    try:
                        frame = await track.recv()

                        # 获取音频数据
                        audio = frame.to_ndarray()

                        # 处理多声道
                        if audio.ndim == 2:
                            audio = np.mean(audio, axis=0)  # 转单声道
                        elif audio.ndim == 1:
                            pass  # 已经是单声道
                        else:
                            continue

                        # 重采样到16kHz
                        audio_16k = AudioResampler.resample_to_16k(
                            audio, frame.sample_rate or 48000
                        )

                        # 推送到处理器
                        await bridge.push_audio(audio_16k)

                    except Exception as e:
                        logger.error(f"音频处理错误: {e}")
                        break

            asyncio.create_task(audio_consumer())

        @track.on("ended")
        async def on_ended():
            logger.info(f"轨道结束: {track.kind}")
            await bridge.stop()

    # 添加输出轨道
    pc.addTrack(VideoOutputTrack(bridge, fps))
    pc.addTrack(AudioOutputTrack(bridge))

    # 设置远程描述
    await pc.setRemoteDescription(offer)

    # 创建应答
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def synthesize(request):
    """离线合成接口: 接收文本或音频, 输出数字人视频"""

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"status": "error", "message": "Invalid JSON payload"}, status=400)

    text = payload.get("text")
    audio_b64 = payload.get("audio_base64")
    audio_path_param = payload.get("audio_path")

    provided_inputs = [bool(text), bool(audio_b64), bool(audio_path_param)]
    if sum(provided_inputs) != 1:
        return web.json_response(
            {
                "status": "error",
                "message": "Provide exactly one of 'text', 'audio_base64', or 'audio_path'."
            },
            status=400,
        )

    output_name = payload.get("output_name", "")
    app = request.app
    data_path: Path = app["data_path"]
    fps: int = app["fps"]
    output_dir: Path = app["output_dir"]
    asr_mode: str = app["asr_mode"]

    temp_dir = Path(tempfile.mkdtemp(prefix="dihuman_tmp_"))
    loop = asyncio.get_running_loop()

    try:
        source_wav_path: Path

        if text:
            source_wav_path = temp_dir / "tts.wav"
            try:
                await loop.run_in_executor(None, _synthesize_tts_to_wav, text, source_wav_path)
            except RuntimeError as exc:
                return web.json_response({"status": "error", "message": str(exc)}, status=400)
        elif audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                return web.json_response({"status": "error", "message": "audio_base64 is invalid"}, status=400)
            source_wav_path = temp_dir / "upload.wav"
            source_wav_path.write_bytes(audio_bytes)
        else:
            source_wav_path = Path(audio_path_param)
            if not source_wav_path.exists():
                return web.json_response(
                    {"status": "error", "message": f"audio_path not found: {source_wav_path}"},
                    status=404,
                )

        try:
            audio_samples, sample_rate = await loop.run_in_executor(None, _load_wav_file, source_wav_path)
        except Exception as exc:
            return web.json_response({"status": "error", "message": str(exc)}, status=400)

        if audio_samples.size == 0:
            return web.json_response({"status": "error", "message": "Audio file is empty"}, status=400)

        audio_16k = AudioResampler.resample_to_16k(audio_samples, sample_rate)

        result = await loop.run_in_executor(
            None,
            _run_offline_synthesis,
            audio_16k,
            data_path,
            fps,
            temp_dir,
            asr_mode,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = _sanitize_filename(output_name)

        response_payload = {"status": "ok"}

        if result["muxed_path"] is not None:
            final_video = output_dir / sanitized_name
            shutil.move(str(result["muxed_path"]), final_video)
            response_payload["video"] = str(final_video)
        else:
            final_video = output_dir / sanitized_name
            shutil.move(str(result["video_path"]), final_video)
            audio_name = f"{Path(sanitized_name).stem}.wav"
            final_audio = output_dir / audio_name
            shutil.move(str(result["audio_path"]), final_audio)
            response_payload.update(
                {
                    "video": str(final_video),
                    "audio": str(final_audio),
                    "message": "ffmpeg not available; returned separate audio and video.",
                }
            )

        return web.json_response(response_payload)

    except Exception as exc:
        logger.exception("Synthesis failed")
        return web.json_response({"status": "error", "message": str(exc)}, status=500)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def on_shutdown(app):
    """关闭所有连接"""
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()


def main():
    parser = argparse.ArgumentParser(description="实时数字人流式服务")
    parser.add_argument("--data_path", required=True, help="模型和数据目录路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8080, help="服务器端口")
    parser.add_argument("--fps", type=int, default=25, choices=[20, 25], help="视频帧率")
    parser.add_argument("--asr", choices=["wenet", "hubert", "ave"], default="wenet", help="音频分支模式")

    args = parser.parse_args()

    # 验证数据路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"错误: 数据路径不存在: {data_path}")
        return

    # 创建应用
    app = web.Application()
    app["data_path"] = data_path
    app["fps"] = args.fps
    app["asr_mode"] = args.asr
    output_dir = data_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    app["output_dir"] = output_dir

    # 配置路由
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/synthesize", synthesize)

    # 配置关闭处理
    app.on_shutdown.append(on_shutdown)

    # 启动服务
    logger.info(f"启动服务: http://{args.host}:{args.port}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"视频帧率: {args.fps} FPS")
    logger.info(f"音频模式: {args.asr}")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
