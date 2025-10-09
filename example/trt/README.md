# TensorRT 推理示例（中文）

`example/trt/` 目录展示如何基于本仓库导出的 UNet ONNX 模型构建 TensorRT 引擎，并在 Python 环境下运行推理，同时提供 CPU 回退能力。脚本定位于开发测试与性能验证阶段，可与 `docs/tensorrt_inference.md` 互补使用。

---

## 环境准备

1. **GPU 与驱动**
   - NVIDIA GPU，推荐 Ampere 架构及以上。
   - 驱动版本需满足 TensorRT 与 CUDA Toolkit 的要求（示例环境：驱动 575+，CUDA 12.4）。

2. **TensorRT 与 CUDA 运行库**
   - 安装 TensorRT 10.x（含 Python wheel 与 `/usr/src/tensorrt` 工具链）。
   - 确保 `libnvinfer*` 等动态库可被系统找到。
   - 导出 Python 搜索路径（使用系统自带 wheel 时）：
     ```bash
     export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
     ```

3. **Python 依赖**
   ```bash
   pip install numpy onnxruntime cuda-python
   ```
   - `cuda-python` 提供 CUDA Runtime API 绑定。
   - 若只跑 CPU 回退，可安装 `onnxruntime` 即可。

4. **ONNX 模型**
   - 使用仓库根目录的 `pth2onnx.py` 导出，例如：
     ```bash
     python pth2onnx.py \
       --checkpoint model/unet/best_hubert.pth \
       --output export/unet_hubert.onnx \
       --variant base \
       --asr hubert
     ```

---

## 脚本一览

| 文件 | 说明 |
|------|------|
| `build_engine.py` | 将 UNet ONNX 转换为 TensorRT `.plan` 引擎，可选 FP16、配置 workspace |
| `infer.py` | 使用 TensorRT 进行推理，必要时自动回退至 ONNX Runtime CPU；接受 `.npy` 输入 |

辅助建议：在真实数据上运行前，可先使用 `numpy` 随机生成 `img_concat.npy`（形状 `[1,6,160,160]` 或 `[1,6,320,320]`）与 `audio_feat.npy`（取决于 ASR）以验证环境。

---

## 1. 构建 TensorRT 引擎

```bash
python example/trt/build_engine.py \
  --onnx export/unet_hubert.onnx \
  --engine export/unet_hubert_fp16.plan \
  --workspace 2048 \
  --fp16
```

- `--workspace`：TensorRT 使用的最大临时显存（MB），值越大优化空间越充分。
- `--fp16`：若 `builder.platform_has_fast_fp16=True`，则启用半精度加速；否则保持 FP32。
- 首次构建可能耗时数分钟，日志中会输出激活内存、权重内存等指标。

若 Python 运行失败，可用命令行工具验证：
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=export/unet_hubert.onnx \
  --saveEngine=export/unet_hubert_fp16.plan \
  --fp16
```

---

## 2. 推理与回退机制

```bash
python example/trt/infer.py \
  --engine export/unet_hubert_fp16.plan \
  --onnx export/unet_hubert.onnx \
  --img-npy export/img_concat.npy \
  --audio-npy export/audio_feat.npy \
  --output export/pred_trt.npy \
  --provider auto \
  --allow-fallback
```

- `--provider auto`：优先使用 TensorRT，失败时尝试 ONNX Runtime（TensorRT EP → CUDA EP → CPU）。
- `--provider trt`：仅尝试 TensorRT，失败即报错。
- `--provider cpu`：直接走 ONNX Runtime CPU，可用于对比结果或调试。
- `--allow-fallback`：允许自动切换到 CPU；若不加，TensorRT 失败时脚本会立即退出。

**输出文件**：`export/pred_trt.npy` 保存为 `[1,3,H,W]` 的 `float32` 张量，可与 PyTorch/ONNX 参考结果逐元素比较。

---

## 3. 常见问题

| 问题 | 处理建议 |
|------|---------|
| `TensorRT Python package not found` | 确认已安装 Python wheel，并配置 `PYTHONPATH` |
| `cudaErrorInsufficientDriver` | 驱动版本过低或 CUDA Runtime 不匹配；参考 `nvidia-smi` 与 `cuda-python` 的版本要求 |
| `Specified provider 'TensorrtExecutionProvider' is not in available provider names` | 当前安装的 onnxruntime 未编译 TensorRT/CUDA EP，可改用 CPU 或重新安装带 EP 的版本 |
| `scrfd_2.5g_kps.onnx` 缺失导致预处理失败 | 将人脸检测模型放入 `data_utils/`，或修改 `fa_landmark.py` 指定路径 |

---

## 4. 推荐验证流程

1. 在 `pth2onnx.py` 导出的 ONNX 上构建引擎；
2. 使用 `trtexec` 确认引擎可在 GPU 上运行，满意时转到 Python；
3. 准备真实数据（`img_concat.npy` / `audio_feat.npy`），用 `infer.py --provider auto` 测试；
4. 对比 `np.max(np.abs(pred_trt - pred_cpu))`，评估 FP16 的数值误差；
5. 若计划部署在服务器或移动端，可将 `.plan`、后处理脚本与关键点平滑模块相结合。

---

## 5. 参考资料

- `docs/tensorrt_inference.md`：更完整的部署流程与性能调优建议
- NVIDIA 官方文档：[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- CUDA Python（cuda-python）项目：https://github.com/NVIDIA/cuda-python

如需扩展到 Jetson 或 INT8 量化，可在 `build_engine.py` 基础上添加 Profile、Calibrator 相关逻辑。欢迎根据业务需求修改脚本或提交改进。***
