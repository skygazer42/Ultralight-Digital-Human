import argparse
import copy
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DoubleConvDW(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=2):

        super(DoubleConvDW, self).__init__() 
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride, use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1, use_res_connect=True, expand_ratio=2)
        )

    def forward(self, x):
        return self.double_conv(x)

class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConvDw, self).__init__() 
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=1, use_res_connect=False, expand_ratio=2)
        )
    def forward(self, x):
        return self.inconv(x)

class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):

        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv =  DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], axis=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class AudioConvWenet(nn.Module):
    def __init__(self):
        super(AudioConvWenet, self).__init__()
        # ch = [16, 32, 64, 128, 256]   # if you want to run this model on a mobile device, use this. 
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[2], ch[3], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1,2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.conv4(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = self.conv6(x)
        x = self.conv7(x)
    
        return x
    
class AudioConvHubert(nn.Module):
    def __init__(self):
        super(AudioConvHubert, self).__init__()
        # ch = [16, 32, 64, 128, 256]   # if you want to run this model on a mobile device, use this. 
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(16, ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2,2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.conv4(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = self.conv6(x)
        x = self.conv7(x)
    
        return x

class Model(nn.Module):
    def __init__(self,n_channels=6, mode='wenet'):
        super(Model, self).__init__()
        self.n_channels = n_channels   #BGR
        # ch = [16, 32, 64, 128, 256]  # if you want to run this model on a mobile device, use this. 
        ch = [32, 64, 128, 256, 512]
        
        if mode=='hubert':
            self.audio_model = AudioConvHubert()
        if mode=='wenet':
            self.audio_model = AudioConvWenet()
            
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4]*2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1)
        )

        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.up1 = Up(ch[4], ch[3]//2)
        self.up2 = Up(ch[3], ch[2]//2)
        self.up3 = Up(ch[2], ch[1]//2)
        self.up4 = Up(ch[1], ch[0])

        self.outc = OutConv(ch[0], 3)

    def forward(self, x, audio_feat):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        audio_feat  = self.audio_model(audio_feat)
        x5 = torch.cat([x5, audio_feat], axis=1)
        x5 = self.fuse_conv(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = F.sigmoid(out)
        return out

def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return an inference-ready copy of ``model`` when it exposes ``reparameterize``."""
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


def _build_export_inputs(device: torch.device, audio_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    img = torch.zeros([1, 6, 160, 160], device=device)
    audio = torch.zeros([1, audio_channels, 16, 32], device=device)
    return img, audio


def _profile_model(net: torch.nn.Module, img: torch.Tensor, audio: torch.Tensor) -> None:
    try:
        from thop import profile, clever_format
    except ImportError:
        print("thop not installed; skip FLOPs/params profiling.")
        return

    flops, params = profile(net, (img, audio))
    macs, params = clever_format([flops, params], "%3f")
    print(macs, params)


def _export_to_onnx(net: torch.nn.Module, img: torch.Tensor, audio: torch.Tensor, onnx_path: Path) -> torch.Tensor:
    with torch.no_grad():
        torch_out = net(img, audio)
        print(torch_out.shape)
        torch.onnx.export(
            net,
            (img, audio),
            str(onnx_path),
            input_names=['input', "audio"],
            output_names=['output'],
            opset_version=11,
            export_params=True,
        )
    return torch_out


def _check_onnx(
    onnx_path: Path,
    torch_out: torch.Tensor,
    torch_in: torch.Tensor,
    audio: torch.Tensor,
    providers: List[str],
) -> None:
    import numpy as np
    import onnx
    import onnxruntime

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {
        ort_session.get_inputs()[0].name: torch_in.cpu().numpy(),
        ort_session.get_inputs()[1].name: audio.cpu().numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(
        torch_out[0].cpu().numpy(),
        ort_outs[0][0],
        rtol=1e-03,
        atol=1e-05,
    )
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def _convert_to_mnn(
    onnx_path: Path,
    mnn_path: Path,
    converter: str = "MNNConvert",
    biz_code: str = "MNN",
    extra_args: Optional[List[str]] = None,
) -> None:
    if shutil.which(converter) is None:
        raise FileNotFoundError(
            f"Cannot find '{converter}'. Install MNN or provide an explicit path via --mnn-converter."
        )

    cmd = [
        converter,
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

    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU for export.")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ultralight Digital Human model to ONNX/MNN.")
    parser.add_argument("--onnx-path", default="./unet.onnx", help="Output path for the ONNX model.")
    parser.add_argument("--mnn-path", default="./unet.mnn", help="Output path for the MNN model when exporting.")
    parser.add_argument("--export-mnn", action="store_true", help="Trigger ONNX->MNN conversion after export.")
    parser.add_argument("--mnn-converter", default="MNNConvert", help="MNN converter executable or full path.")
    parser.add_argument("--biz-code", default="MNN", help="BizCode argument passed to MNNConvert.")
    parser.add_argument("--device", default="cuda", help="Torch device used during export (e.g., cuda or cpu).")
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=128,
        help="Audio tensor channel count. Use 128 for wenet and 16 for hubert mode.",
    )
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma separated ONNXRuntime providers for validation.",
    )
    parser.add_argument("--skip-profile", action="store_true", help="Skip FLOPs/params profiling step.")
    parser.add_argument("--skip-onnx-check", action="store_true", help="Skip ONNXRuntime validation step.")
    parser.add_argument(
        "--mnn-extra-args",
        nargs="*",
        default=None,
        help="Extra arguments passed to MNNConvert.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    device = _resolve_device(args.device)

    net = Model(6).eval().to(device)
    img, audio = _build_export_inputs(device, args.audio_channels)

    if not args.skip_profile:
        _profile_model(net, img, audio)

    torch_out = _export_to_onnx(net, img, audio, onnx_path)

    if not args.skip_onnx_check:
        providers = [p.strip() for p in args.providers.split(',') if p.strip()]
        if not providers:
            providers = ["CPUExecutionProvider"]
        _check_onnx(onnx_path, torch_out, img, audio, providers)

    if args.export_mnn:
        mnn_path = Path(args.mnn_path)
        _convert_to_mnn(
            onnx_path,
            mnn_path,
            converter=args.mnn_converter,
            biz_code=args.biz_code,
            extra_args=args.mnn_extra_args,
        )
        print(f"MNN model written to {mnn_path}")


if __name__ == '__main__':
    main()
