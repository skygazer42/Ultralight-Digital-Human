#!/usr/bin/env python3
"""TensorRT inference example with optional CPU fallback via ONNX Runtime."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import tensorrt as trt
    try:
        from cuda import cudart  # type: ignore
    except ImportError:  # cuda-python >= 13 moved symbols under cuda.bindings
        from cuda.bindings import runtime as cudart  # type: ignore
except ImportError:  # pragma: no cover - optional dependencies
    trt = None
    cudart = None


def _check_cuda(err: int, msg: str) -> None:
    if err != cudart.cudaError_t.cudaSuccess:  # type: ignore[union-attr]
        raise RuntimeError(f"{msg}: {err}")


def _set_bindings_shapes(
    context: "trt.IExecutionContext",
    engine: "trt.ICudaEngine",
    inputs: Dict[str, np.ndarray],
) -> None:
    for name, array in inputs.items():
        binding_idx = engine.get_binding_index(name)
        if binding_idx < 0:
            raise KeyError(f"Binding '{name}' not found in engine.")
        if not engine.binding_is_input(binding_idx):
            raise ValueError(f"Binding '{name}' is not an input.")
        shape = tuple(array.shape)
        if not context.set_binding_shape(binding_idx, shape):
            raise RuntimeError(f"Failed to set binding shape {shape} for '{name}'.")


def _allocate_device_buffers(
    context: "trt.IExecutionContext",
    engine: "trt.ICudaEngine",
) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    binding_ptrs: List[int] = [0] * engine.num_bindings
    host_inputs: Dict[str, np.ndarray] = {}
    host_outputs: Dict[str, np.ndarray] = {}

    for binding_idx in range(engine.num_bindings):
        binding_name = engine.get_binding_name(binding_idx)
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        shape = tuple(context.get_binding_shape(binding_idx))
        if -1 in shape:
            raise RuntimeError(f"Binding {binding_name} shape is dynamic and unset: {shape}")

        host_array = np.empty(shape, dtype=dtype)
        err, device_ptr = cudart.cudaMalloc(host_array.nbytes)  # type: ignore[union-attr]
        _check_cuda(err, f"cudaMalloc failed for {binding_name}")
        binding_ptrs[binding_idx] = device_ptr

        if engine.binding_is_input(binding_idx):
            host_inputs[binding_name] = host_array
        else:
            host_outputs[binding_name] = host_array
    return binding_ptrs, host_inputs, host_outputs


def run_tensorrt(
    engine_path: Path,
    inputs: Dict[str, np.ndarray],
    device_id: int,
) -> Dict[str, np.ndarray]:
    if trt is None or cudart is None:
        raise RuntimeError("TensorRT or CUDA Python bindings are unavailable.")

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    logger = trt.Logger(trt.Logger.INFO)
    err, _ = cudart.cudaSetDevice(device_id)  # type: ignore[union-attr]
    _check_cuda(err, f"cudaSetDevice({device_id}) failed")

    runtime = trt.Runtime(logger)
    with engine_path.open("rb") as engine_file:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context.")

    _set_bindings_shapes(context, engine, inputs)
    binding_ptrs, host_inputs, host_outputs = _allocate_device_buffers(context, engine)
    err, stream = cudart.cudaStreamCreate()  # type: ignore[union-attr]
    _check_cuda(err, "cudaStreamCreate failed")

    try:
        for name, data in inputs.items():
            host_buf = host_inputs[name]
            np.copyto(host_buf, np.ascontiguousarray(data))
            idx = engine.get_binding_index(name)
            err = cudart.cudaMemcpyAsync(  # type: ignore[union-attr]
                binding_ptrs[idx],
                host_buf.ctypes.data,
                host_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
            _check_cuda(err, f"cudaMemcpyAsync H2D failed for {name}")

        if not context.execute_async_v2(binding_ptrs, stream):  # pragma: no cover - TensorRT execution
            raise RuntimeError("TensorRT execution failed.")

        for name, host_buf in host_outputs.items():
            idx = engine.get_binding_index(name)
            err = cudart.cudaMemcpyAsync(  # type: ignore[union-attr]
                host_buf.ctypes.data,
                binding_ptrs[idx],
                host_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
            _check_cuda(err, f"cudaMemcpyAsync D2H failed for {name}")

        err = cudart.cudaStreamSynchronize(stream)  # type: ignore[union-attr]
        _check_cuda(err, "cudaStreamSynchronize failed")
    finally:
        cudart.cudaStreamDestroy(stream)  # type: ignore[union-attr]
        for ptr in binding_ptrs:
            if ptr:
                cudart.cudaFree(ptr)  # type: ignore[union-attr]

    return {name: np.array(buf, copy=True) for name, buf in host_outputs.items()}


def run_onnxruntime(
    onnx_path: Path,
    inputs: Dict[str, np.ndarray],
    providers: List[str],
) -> Dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    outputs = session.run(None, {k: v.astype(np.float32, copy=False) for k, v in inputs.items()})
    output_names = [output.name for output in session.get_outputs()]
    return {name: value for name, value in zip(output_names, outputs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralight Digital Human inference with TensorRT or CPU fallback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", type=Path, help="TensorRT engine (.plan) path.")
    parser.add_argument("--onnx", type=Path, help="ONNX model path for fallback or CPU inference.")
    parser.add_argument("--img-npy", required=True, type=Path, help="Numpy file with mouth ROI tensor [1,6,H,W].")
    parser.add_argument("--audio-npy", required=True, type=Path, help="Numpy file with audio tensor [1,C,32,32].")
    parser.add_argument("--output", required=True, type=Path, help="Where to save the network prediction (.npy).")
    parser.add_argument(
        "--provider",
        choices=("auto", "trt", "cpu"),
        default="auto",
        help="Preferred backend: TensorRT (GPU), CPU (ONNX Runtime), or auto fallback.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index for TensorRT.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow automatic fallback to CPU when TRT fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inputs = {
        "input": np.load(args.img_npy).astype(np.float32),
        "audio": np.load(args.audio_npy).astype(np.float32),
    }

    outputs: Dict[str, np.ndarray]
    run_trt_first = args.provider in ("auto", "trt")

    if run_trt_first:
        if args.engine is None:
            raise SystemExit("--engine is required for TensorRT inference.")
        try:
            outputs = run_tensorrt(args.engine, inputs, args.device)
        except Exception as exc:
            if args.provider == "trt" or not args.allow_fallback:
                raise
            print(f"[WARN] TensorRT execution failed ({exc}); falling back to CPU.")
            run_trt_first = False
        else:
            np.save(args.output, outputs["output"])
            return

    if not run_trt_first:
        if args.onnx is None:
            raise SystemExit("Fallback requested but --onnx was not provided.")

        providers = ["CPUExecutionProvider"]
        with contextlib.suppress(ImportError):
            import onnxruntime  # noqa: F401
            # TensorRT Execution Provider with FP16 acceleration when available
            if args.provider == "auto":
                providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

        outputs = run_onnxruntime(args.onnx, inputs, providers)
        np.save(args.output, outputs["output"])


if __name__ == "__main__":
    main()
