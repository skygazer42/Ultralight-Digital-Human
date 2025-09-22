#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数字人流式服务 - WebRTC版本
接收麦克风音频 -> AI推理 -> 实时返回视频流
"""

import asyncio
import argparse
import json
import logging
import os
import time
from typing import Optional
from collections import deque

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
from dihuman_run import DiHumanProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self, processor: DiHumanProcessor, fps: int = 25):
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
    data_path = request.app["data_path"]
    fps = request.app["fps"]

    # 创建连接
    pc = RTCPeerConnection()
    peer_connections.add(pc)

    # 为每个连接创建独立的处理器
    processor = DiHumanProcessor(data_path)
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

    args = parser.parse_args()

    # 验证数据路径
    if not os.path.exists(args.data_path):
        print(f"错误: 数据路径不存在: {args.data_path}")
        return

    # 创建应用
    app = web.Application()
    app["data_path"] = args.data_path
    app["fps"] = args.fps

    # 配置路由
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    # 配置关闭处理
    app.on_shutdown.append(on_shutdown)

    # 启动服务
    logger.info(f"启动服务: http://{args.host}:{args.port}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"视频帧率: {args.fps} FPS")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()