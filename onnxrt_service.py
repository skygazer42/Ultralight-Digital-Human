"""
Lightweight HTTP service that performs Ultralight Digital Human inference purely with ONNX Runtime.

Usage
-----
python onnxrt_service.py \
    --data-path ./runtime_assets \
    --asr wenet \
    --host 0.0.0.0 \
    --port 8000 \
    --fps 20 \
    --use-smooth \
    --use-soft-mask \
    --use-seamless

The service exposes three endpoints:
  - GET /ping                     -> Health check.
  - POST /process                 -> Stream a 10ms (160 @16kHz) pcm16 audio chunk.
  - POST /reset                   -> Reset processor state between conversations.

POST /process expects JSON payload:
{
  "audio": "<base64 encoded int16 PCM>"
}

The response contains inferred audio and, when available, the composed video frame:
{
  "ok": true,
  "has_frame": true,
  "frame_jpeg": "<base64>",
  "audio": "<base64 pcm16>"
}

Dependencies
------------
pip install fastapi uvicorn[standard] numpy opencv-python onnxruntime
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dihuman_run import DiHumanProcessor  # wenet branch
from dihuman_run_hubert import DiHumanProcessorHubert
from dihuman_run_ave import DiHumanProcessorAve


class AudioChunk(BaseModel):
    audio: str  # base64 encoded int16 PCM, 160 samples (10ms @ 16kHz)


class ServiceState:
    """Holds the processor instance and ensures sequential execution."""

    def __init__(self, processor):
        self.processor = processor
        self.lock = asyncio.Lock()


def _create_processor(
    asr: str,
    data_path: Path,
    fps: int,
    use_smooth: bool,
    use_soft_mask: bool,
    use_seamless: bool,
):
    mode = asr.lower()
    if mode == "wenet":
        return DiHumanProcessor(
            str(data_path),
            fps=fps,
            use_smooth=use_smooth,
            use_soft_mask=use_soft_mask,
            use_seamless=use_seamless,
        )
    if mode == "hubert":
        return DiHumanProcessorHubert(
            str(data_path),
            fps=fps,
            use_smooth=use_smooth,
            use_soft_mask=use_soft_mask,
            use_seamless=use_seamless,
        )
    if mode == "ave":
        return DiHumanProcessorAve(
            str(data_path),
            fps=fps,
            use_smooth=use_smooth,
            use_soft_mask=use_soft_mask,
            use_seamless=use_seamless,
        )
    raise ValueError(f"Unsupported ASR mode: {asr}")


def _ensure_chunk(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.int16:
        raise ValueError("Audio samples must be int16 PCM.")
    if samples.size == 0:
        raise ValueError("Audio chunk is empty.")
    if samples.size % 160 != 0:
        raise ValueError("Audio chunk length must be a multiple of 160 samples (10ms @16kHz).")
    return samples


def build_app(state: ServiceState) -> FastAPI:
    app = FastAPI(title="Ultralight Digital Human ONNX Runtime Service")

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    @app.post("/reset")
    async def reset() -> dict:
        async with state.lock:
            state.processor.reset()
        return {"ok": True}

    @app.post("/process")
    async def process(chunk: AudioChunk) -> dict:
        try:
            audio_bytes = base64.b64decode(chunk.audio)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {exc}") from exc

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        try:
            samples = _ensure_chunk(samples)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        frames = []
        audio_out = []
        has_frame = False
        frame_jpeg: Optional[str] = None

        async with state.lock:
            for offset in range(0, samples.size, 160):
                sub_chunk = samples[offset : offset + 160]
                img, playing_audio, check_img = state.processor.process(sub_chunk)

                if playing_audio is not None:
                    audio_out.append(playing_audio.astype(np.int16))

                if check_img and img is not None:
                    has_frame = True
                    frames.append(img)

        response = {"ok": True, "has_frame": has_frame}

        if audio_out:
            merged = np.concatenate(audio_out).astype(np.int16)
            response["audio"] = base64.b64encode(merged.tobytes()).decode("ascii")

        if frames:
            # Only return the most recent frame by default
            _, buffer = cv2.imencode(".jpg", frames[-1])
            frame_jpeg = base64.b64encode(buffer.tobytes()).decode("ascii")
            response["frame_jpeg"] = frame_jpeg

        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralight Digital Human ONNX Runtime HTTP service")
    parser.add_argument("--data-path", required=True, type=Path, help="Directory containing img_inference/lms_inference")
    parser.add_argument("--asr", choices=["wenet", "hubert", "ave"], default="wenet", help="Audio encoder branch")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS (wenet=20, hubert/ave=25)")
    parser.add_argument("--use-smooth", action="store_true", help="Enable landmark smoothing")
    parser.add_argument("--use-soft-mask", action="store_true", help="Enable soft edge blending")
    parser.add_argument("--use-seamless", action="store_true", help="Enable seamlessClone compositing")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path: Path = args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    processor = _create_processor(
        asr=args.asr,
        data_path=data_path,
        fps=args.fps,
        use_smooth=args.use_smooth,
        use_soft_mask=args.use_soft_mask,
        use_seamless=args.use_seamless,
    )
    state = ServiceState(processor)
    app = build_app(state)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
