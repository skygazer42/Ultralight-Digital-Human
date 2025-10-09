"""Export Ultralight Digital Human models to ONNX and optional MNN format.

This helper consolidates the conversion steps needed to deploy the UNet
renderers and the AVE audio encoder inside C++ inference stacks.  It can:

* Load the PyTorch checkpoints shipped with the repository.
* Export the selected component to ONNX with stable tensor shapes.
* Optionally run MNNConvert to produce an `.mnn` artifact for MNN runtimes.
* (Optionally) verify the ONNX output against PyTorch when onnxruntime is installed.

Example usage (AVE audio encoder → ONNX + MNN):

    python tools/export_models.py ^
        --component audio_encoder ^
        --onnx export/audio_encoder.onnx ^
        --mnn export/audio_encoder.mnn ^
        --mnnconvert C:/path/to/MNNConvert.exe

Example usage (328 UNet for AVE → ONNX):

    python tools/export_models.py ^
        --component unet ^
        --variant 328 ^
        --asr ave ^
        --checkpoint checkpoint/May/100000.pth ^
        --onnx export/unet_328_ave.onnx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import torch


@dataclass
class ExportArgs:
    checkpoint: Path
    onnx_path: Path
    opset: int
    verify: bool
    dynamic_batch: bool


def _load_audio_encoder(checkpoint: Path):
    from data_utils.ave.models.audioEnc import AudioEncoder

    model = AudioEncoder().eval()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "audio_encoder" in state and isinstance(state["audio_encoder"], dict):
        state = state["audio_encoder"]

    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format for audio encoder: {type(state)}")

    if all(key.startswith("audio_encoder.") for key in state.keys()):
        mapped_state = state
    elif any(key.startswith("audio_encoder.") for key in state.keys()):
        mapped_state = state
    else:
        mapped_state = {f"audio_encoder.{k}": v for k, v in state.items()}

    model.load_state_dict(mapped_state)
    return model


def export_audio_encoder(args: ExportArgs) -> Sequence[torch.Tensor]:
    print(f"[INFO] Exporting audio encoder from {args.checkpoint}")
    print(f"[INFO] Writing ONNX to {args.onnx_path}")
    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_audio_encoder(args.checkpoint)
    dummy = torch.zeros(1, 1, 80, 16, dtype=torch.float32)
    dynamic_axes = {"mel": {0: "batch"}, "embedding": {0: "batch"}} if args.dynamic_batch else None
    with torch.no_grad():
        torch_out = model(dummy)
        torch.onnx.export(
            model,
            dummy,
            str(args.onnx_path),
            input_names=["mel"],
            output_names=["embedding"],
            opset_version=args.opset,
            export_params=True,
            dynamic_axes=dynamic_axes,
        )
    return (dummy, torch_out)


def _load_unet_model(variant: str, asr: str):
    if variant == "328":
        from unet_328 import Model
    else:
        from unet import Model
    return Model(6, asr).eval()


def export_unet(
    checkpoint: Path,
    onnx_path: Path,
    variant: str,
    asr: str,
    opset: int,
) -> Sequence[torch.Tensor]:
    from pth2onnx import IMG_SHAPES, AUDIO_SHAPES, load_state_dict as load_unet_state

    if variant not in IMG_SHAPES:
        raise ValueError(f"Unsupported UNet variant: {variant}")
    audio_shapes: Dict[str, tuple[int, int, int, int]] = AUDIO_SHAPES.get(variant, {})
    if asr not in audio_shapes:
        supported = ", ".join(sorted(audio_shapes))
        raise ValueError(f"Variant '{variant}' does not support ASR '{asr}'. Supported: {supported}")

    img_shape = IMG_SHAPES[variant]
    audio_shape = audio_shapes[asr]
    model = _load_unet_model(variant, asr)
    load_unet_state(model, checkpoint)

    dummy_img = torch.zeros(img_shape, dtype=torch.float32)
    dummy_audio = torch.zeros(audio_shape, dtype=torch.float32)

    print(f"[INFO] Exporting UNet variant={variant} asr={asr} from {checkpoint}")
    print(f"[INFO] image input shape: {img_shape}")
    print(f"[INFO] audio input shape: {audio_shape}")
    print(f"[INFO] Writing ONNX to {onnx_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch_out = model(dummy_img, dummy_audio)
        torch.onnx.export(
            model,
            (dummy_img, dummy_audio),
            str(onnx_path),
            input_names=["image", "audio"],
            output_names=["prediction"],
            opset_version=opset,
            export_params=True,
        )
    return (dummy_img, dummy_audio, torch_out)


def _verify_single_output(
    model_outputs: Sequence[torch.Tensor],
    example_inputs: Sequence[torch.Tensor],
    onnx_path: Path,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed; skipping ONNX verification.")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_inputs = {
        inp.name: tensor.detach().cpu().numpy()
        for inp, tensor in zip(session.get_inputs(), example_inputs)
    }
    ort_outs = session.run(None, ort_inputs)

    for idx, (torch_out, ort_out) in enumerate(zip(model_outputs[-1:], ort_outs)):
        np.testing.assert_allclose(
            torch_out.detach().cpu().numpy(),
            ort_out,
            rtol=1e-03,
            atol=1e-05,
        )
    print("[INFO] ONNX verification passed.")


def run_mnnconvert(
    onnx_path: Path,
    mnn_path: Path,
    mnnconvert: Path,
    biz_code: str,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    if not mnnconvert.exists():
        raise FileNotFoundError(f"MNNConvert not found: {mnnconvert}")
    mnn_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(mnnconvert),
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--bizCode",
        biz_code,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[INFO] Running MNNConvert: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] Wrote MNN model to {mnn_path}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export models to ONNX and optional MNN format.")
    parser.add_argument(
        "--component",
        choices=["audio_encoder", "unet"],
        required=True,
        help="Select which component to export.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Path to the PyTorch checkpoint.")
    parser.add_argument("--onnx", type=Path, required=True, help="Destination ONNX path.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to target.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare ONNX runtime output against PyTorch (requires onnxruntime).",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export audio encoder with a dynamic batch dimension.",
    )

    # UNet-specific options
    parser.add_argument("--variant", choices=["base", "328"], help="UNet image branch size.")
    parser.add_argument("--asr", choices=["hubert", "wenet", "ave"], help="Audio branch used during training.")

    # Optional MNN conversion
    parser.add_argument("--mnn", type=Path, help="Destination path for the generated MNN model.")
    parser.add_argument("--mnnconvert", type=Path, help="Path to the MNNConvert executable.")
    parser.add_argument("--mnn-biz-code", default="UDH", help="bizCode to annotate in the MNN file.")
    parser.add_argument(
        "--mnn-extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended when invoking MNNConvert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    if args.component == "audio_encoder":
        checkpoint = args.checkpoint or Path("model/checkpoints/audio_visual_encoder.pth")
        export_args = ExportArgs(
            checkpoint=checkpoint,
            onnx_path=args.onnx,
            opset=args.opset,
            verify=args.verify,
            dynamic_batch=args.dynamic_batch,
        )
        example_tensors = export_audio_encoder(export_args)
        outputs = (example_tensors[1],)
        inputs = (example_tensors[0],)
    else:  # UNet
        if args.variant is None or args.asr is None:
            raise SystemExit("--variant and --asr are required when exporting the UNet.")
        checkpoint = args.checkpoint
        if checkpoint is None:
            raise SystemExit("--checkpoint must be provided for UNet export.")
        tensors = export_unet(checkpoint, args.onnx, args.variant, args.asr, args.opset)
        inputs = tensors[:2]
        outputs = (tensors[2],)

    if args.verify:
        _verify_single_output(outputs, inputs, args.onnx)

    if args.mnn:
        if not args.mnnconvert:
            raise SystemExit("--mnnconvert is required when specifying --mnn output.")
        run_mnnconvert(args.onnx, args.mnn, args.mnnconvert, args.mnn_biz_code, args.mnn_extra)


if __name__ == "__main__":
    main()
