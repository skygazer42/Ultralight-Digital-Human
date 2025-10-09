#!/usr/bin/env python3
"""Helper script for building TensorRT engines from Ultralight Digital Human ONNX models."""

import argparse
import os
from pathlib import Path

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover - TensorRT is optional
    raise SystemExit(
        "TensorRT Python package not found. Install it from NVIDIA before running this script."
    ) from exc


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_mb: int,
    enable_fp16: bool,
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as onnx_file:
        if not parser.parse(onnx_file.read()):  # pragma: no cover - TensorRT parser messages
            first_error = parser.get_error(0)
            for i in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
            raise RuntimeError(f"Failed to parse ONNX: {first_error}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    profile = builder.create_optimization_profile()
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        if tensor.is_shape_tensor:  # pragma: no cover - unlikely for this graph
            continue
        shape = tuple(tensor.shape)
        profile.set_shape(tensor.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if enable_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.log(trt.Logger.INFO, "FP16 mode enabled.")
        else:
            logger.log(trt.Logger.WARNING, "FP16 requested but platform lacks fast FP16; falling back to FP32.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as f:
        f.write(serialized_engine)
    logger.log(trt.Logger.INFO, f"Engine saved to {engine_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ultralight Digital Human ONNX models to TensorRT engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", required=True, type=Path, help="Path to ONNX model (e.g. unet_hubert.onnx)")
    parser.add_argument("--engine", required=True, type=Path, help="Output path for TensorRT engine.")
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Workspace size in MB to allocate for TensorRT optimizations.",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX model not found: {args.onnx}")

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    build_engine(args.onnx, args.engine, args.workspace, args.fp16)


if __name__ == "__main__":
    main()

