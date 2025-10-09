"""Utility to convert trained checkpoints to ONNX with the correct audio branch."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import torch


IMG_SHAPES: Dict[str, Tuple[int, int, int, int]] = {
    "base": (1, 6, 160, 160),
    "328": (1, 6, 320, 320),
}

AUDIO_SHAPES: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "base": {
        "hubert": (1, 16, 32, 32),
        "wenet": (1, 128, 16, 32),
    },
    "328": {
        "hubert": (1, 32, 32, 32),
        "wenet": (1, 256, 16, 32),
        "ave": (1, 32, 16, 16),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ultralight Digital Human UNet to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the trained *.pth checkpoint")
    parser.add_argument(
        "--output", type=Path, default=Path("unet.onnx"), help="Destination path for the exported ONNX file"
    )
    parser.add_argument(
        "--variant",
        choices=sorted(IMG_SHAPES.keys()),
        default="base",
        help="Model variant used during training (base=160px UNet, 328=larger UNet)",
    )
    parser.add_argument(
        "--asr",
        choices=sorted({mode for variants in AUDIO_SHAPES.values() for mode in variants}),
        default="wenet",
        help="Audio branch used during training (must match the checkpoint)",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run an ONNXRuntime round-trip check to ensure the export matches PyTorch",
    )
    return parser.parse_args()


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def check_onnx(torch_out: torch.Tensor, torch_img: torch.Tensor, torch_audio: torch.Tensor, onnx_path: Path) -> None:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    import onnxruntime  # Imported lazily to avoid dependency when not verifying

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)

    ort_inputs = {
        ort_session.get_inputs()[0].name: torch_img.cpu().numpy(),
        ort_session.get_inputs()[1].name: torch_audio.cpu().numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        torch_out.cpu().numpy(),
        np.asarray(ort_outs[0]),
        rtol=1e-03,
        atol=1e-05,
    )


def main() -> None:
    args = parse_args()

    if args.variant not in IMG_SHAPES:
        raise ValueError(f"Unsupported model variant: {args.variant}")

    variant_audio_shapes = AUDIO_SHAPES.get(args.variant, {})
    if args.asr not in variant_audio_shapes:
        supported = ", ".join(sorted(variant_audio_shapes))
        raise ValueError(f"Variant '{args.variant}' does not support ASR '{args.asr}'. Supported: {supported}")

    audio_shape = variant_audio_shapes[args.asr]
    img_shape = IMG_SHAPES[args.variant]

    if args.variant == "328":
        from unet_328 import Model
    else:
        from unet import Model

    model = Model(n_channels=img_shape[1], mode=args.asr).eval()
    load_state_dict(model, args.checkpoint)

    dummy_img = torch.zeros(img_shape, dtype=torch.float32)
    dummy_audio = torch.zeros(audio_shape, dtype=torch.float32)

    print(f"Exporting '{args.checkpoint}' with mode='{args.asr}'")
    print(f"  image input shape: {img_shape}")
    print(f"  audio input shape: {audio_shape}")
    print(f"  writing ONNX to: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch_out = model(dummy_img, dummy_audio)
        torch.onnx.export(
            model,
            (dummy_img, dummy_audio),
            str(args.output),
            input_names=["input", "audio"],
            output_names=["output"],
            opset_version=args.opset,
            export_params=True,
        )

    if args.verify:
        print("Running ONNXRuntime verification ...")
        check_onnx(torch_out, dummy_img, dummy_audio, args.output)
        print("ONNX verification passed.")


if __name__ == "__main__":
    main()
