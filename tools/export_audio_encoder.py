import argparse
from pathlib import Path

import torch

from data_utils.ave.models.audioEnc import AudioEncoder


def export_audio_encoder(checkpoint: Path, output: Path) -> None:
    device = torch.device("cpu")
    model = AudioEncoder().to(device)
    state = torch.load(checkpoint, map_location=device)

    if all(key.startswith("audio_encoder.") for key in state.keys()):
        model.load_state_dict({key.replace("audio_encoder.", "audio_encoder."): value for key, value in state.items()})
    elif "audio_encoder" in state:
        model.load_state_dict(state["audio_encoder"])
    else:
        new_state = {}
        for key, value in state.items():
            new_state[f"audio_encoder.{key}"] = value
        model.load_state_dict(new_state)

    model.eval()
    dummy = torch.zeros(1, 1, 80, 16, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Exported audio encoder to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AVE audio encoder to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to audio_visual_encoder.pth")
    parser.add_argument("--output", type=Path, required=True, help="Destination ONNX path")
    args = parser.parse_args()

    export_audio_encoder(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
