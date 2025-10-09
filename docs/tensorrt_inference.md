# TensorRT 推理加速示例

`example/trt/` 提供了一个使用 TensorRT 加速 UNet 推理的小型示例，并在 GPU 不可用时支持回退到 CPU（ONNX Runtime）。该示例沿用 Python 版本的预处理输出（`img_concat.npy`、`audio_feat.npy`），方便与现有脚本进行结果对比。

---

## 1. 环境准备

1. 安装 NVIDIA TensorRT（>= 8.5）以及匹配的 CUDA 驱动
2. 安装 Python 依赖：
   ```bash
   pip install cuda-python numpy onnxruntime-gpu
   ```
   > 若仅需 CPU 回退，可安装 `onnxruntime` 即可。

3. 准备 ONNX 模型（参考 `docs/mobile_inference.md` 的导出步骤），以及一份前处理好的输入：
   ```bash
   python inference.py --dataset demo --audio_feat demo/audio.npy --save_path demo/tmp.avi \
     --checkpoint checkpoints/ultralight_dh.pth --asr hubert --use_smooth
   # 运行后可在调试代码中保存 img_concat.npy / audio_feat.npy 以用于离线验证
   ```

---

## 2. 构建 TensorRT 引擎

```bash
python example/trt/build_engine.py \
  --onnx export/unet_hubert.onnx \
  --engine export/unet_hubert_fp16.plan \
  --workspace 2048 \
  --fp16
```

- `--fp16` 会在硬件支持时启用半精度推理（Tensor Core），无法启用时自动退回 FP32
- `--workspace` 设置优化所需的显存上限（MB），较大的值通常会带来更激进的 fuse/conv 优化

生成的 `.plan` 文件可直接部署到服务器或桌面端，若要迁移到 Jetson/DLA，可根据需要在脚本中添加 profile 配置。

---

## 3. TensorRT 推理 + CPU 回退

```bash
python example/trt/infer.py \
  --engine export/unet_hubert_fp16.plan \
  --onnx export/unet_hubert.onnx \
  --img-npy demo/img_concat.npy \
  --audio-npy demo/audio_feat.npy \
  --output demo/pred_trt.npy \
  --provider auto \
  --allow-fallback
```

- 默认优先使用 TensorRT；若执行失败或找不到 GPU，脚本会自动退回 ONNX Runtime CPU（需 `--allow-fallback`）
- 想要强制 TensorRT，可加 `--provider trt`；若只想在 CPU 上验证，可加 `--provider cpu`

运行结束后会得到与 Python 版本相同的输出 `demo/pred_trt.npy`，可以对比精度差异：

```python
import numpy as np
pred_trt = np.load("demo/pred_trt.npy")
pred_ref = np.load("demo/pred.npy")
print(np.max(np.abs(pred_trt - pred_ref)))
```

---

## 4. 常见问题

- **`ImportError: No module named tensorrt`**：需要先安装官方发布的 TensorRT Python Wheel，并确保 `LD_LIBRARY_PATH` / `PATH` 指向对应的 `.so`/`.dll`
- **`pycuda`/`cuda-python` 未安装**：示例使用 [cuda-python](https://github.com/NVIDIA/cuda-python) 分配 GPU 内存；安装后重试
- **FP16 精度偏差**：可选 `--fp16` 或者直接构建 FP32 引擎；通过 `infer.py --provider auto` 验证两者误差
- **Jetson 设备**：建议使用 JetPack 自带的 TensorRT，并根据设备算力调整 `workspace`，必要时添加 INT8 校准

---

借助该示例，你可以快速把现有的 ONNX 模型迁移至 TensorRT，并在缺少 GPU 的环境下回退到 CPU 推理，方便在 CI 或开发机上做功能验证。***
