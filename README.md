# Ultralight Digital Human

Ultralight Digital Human 提供一套端到端的口型驱动数字人解决方案，从数据准备、加权采样训练到跨平台推理部署均有覆盖。项目围绕轻量级 UNet 结构与多种 ASR 分支（HuBERT/Wenet/AVE）设计，兼容 PyTorch、ONNX Runtime、TensorRT、MNN 等推理框架。

---

## 核心特性
- **全流程工具链**：数据清洗、关键点预测、音频编码、SyncNet 判别器、主网络训练及推理脚本一应俱全。
- **多分支 ASR 支持**：HuBERT（25 fps）、Wenet（20 fps）、AVE（320×320 输入）。
- **灵活部署**：参考示例涵盖 C++ ONNX Runtime、移动端 MNN、TensorRT FP16 引擎。
- **加权采样训练策略**：覆盖均衡 + 困难样本挖掘（Auto A1 Fusion），提升模型稳健性。
- **详尽实验记录**：持续跟踪不同骨干与 Landmark 组合的表现，为模型迭代提供依据。

---

## 环境准备

```bash
conda activate dh
```

建议在同一环境中安装 PyTorch（CUDA 12.x）、onnxruntime、opencv-python、cuda-python 等依赖。若需要 TensorRT Python API，可设置：

```bash
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
```

> 注意：HuBERT/AVE 分支要求输入视频帧率为 25fps，Wenet 分支为 20fps。

---

## 仓库结构总览

| 目录/脚本 | 说明 |
|-----------|------|
| `data_utils/` | 数据预处理、特征提取、采样权重构建与 Landmark 检测脚本 |
| `train.py` / `train_328.py` | Base（160×160）与 328 模型训练入口 |
| `inference.py` | PyTorch 离线推理 |
| `example/cpp` | ONNX Runtime C++ 离线推理示例 |
| `example/mnn` | Android MNN Demo |
| `example/trt` | TensorRT 引擎构建与推理（含 CPU 回退） |
| `docs/*.md` | 详细部署与扩展说明（手机端、C++、TensorRT、HTTP 服务等） |

---

## 快速上手

1. **数据预处理**
   ```bash
   python process.py test.mp4 --asr hubert
   # 如需其他分支
   python process.py test.mp4 --asr wenet
   python process.py test.mp4 --asr ave
   ```
   - 输出位于 `data_utils/`，包括 `full_body_img/`、`landmarks/`、`aud.wav` 等。
   - 若出现 “Using hubert/ave, your video fps should be 25!!!”，请先用 `ffmpeg -r 25` 重采样。
   - Landmark 检测依赖 `scrfd_2.5g_kps.onnx`，请将模型放置在 `data_utils/`。

2. **覆盖均衡权重**
   ```bash
   python data_utils/build_sampling_weights.py --root ./data_utils --asr hubert --k 50
   python data_utils/build_sampling_weights.py --root ./data_utils --asr wenet  --k 50
   ```

3. **同步网（SyncNet）训练**
   ```bash
   python syncnet.py --save_dir ./syncnet_hubert/ --dataset_dir ./data_utils/ --asr hubert
   python syncnet.py --save_dir ./syncnet_wenet/  --dataset_dir ./data_utils/ --asr wenet
   python syncnet.py --save_dir ./syncnet_ave/    --dataset_dir ./data_utils/ --asr ave
   ```

4. **困难样本挖掘（Auto A1）**
   ```bash
   python data_utils/build_hard_weights.py --root ./data_utils --asr hubert --ckpt ./syncnet_ckpt/39.pth
   python data_utils/build_hard_weights.py --root ./data_utils --asr wenet  --ckpt ./syncnet_/39.pth
   ```

5. **UNet 加权训练**
   ```bash
   python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_hubert/ --asr hubert \
     --use_syncnet --syncnet_checkpoint ./syncnet_hubert/39.pth --weighted_sampling

   python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_wenet/ --asr wenet \
     --use_syncnet --syncnet_checkpoint ./syncnet_wenet/39.pth --weighted_sampling

   python train_328.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_ave/ --asr ave \
     --use_syncnet --syncnet_checkpoint ./syncnet_ave/39.pth --weighted_sampling
   ```

6. **音频特征提取与离线推理**
   ```bash
   cd data_utils/
   python hubert.py --wav aud.wav
   cd ..

   python inference.py --asr hubert --dataset ./data_utils/ \
     --audio_feat data_utils/aud_hu.npy --save_path a.mp4 --checkpoint checkpoint/best.pth

   ffmpeg -i a.mp4 -i aud.wav -c:v libx264 -c:a aac atest.mp4
   ```

7. **流程速览（Hubert 示例）**
   ```bash
   python process.py test.mp4 --asr hubert
   python data_utils/build_sampling_weights.py --root ./data_utils --asr hubert --k 50
   python syncnet.py --save_dir ./syncnet_ckpt/ --dataset_dir ./data_utils/ --asr hubert
   python data_utils/build_hard_weights.py --root ./data_utils --asr hubert --ckpt ./syncnet_ckpt/39.pth
   python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint/ --asr hubert \
     --use_syncnet --syncnet_checkpoint ./syncnet_ckpt/39.pth --weighted_sampling
   python inference.py --asr hubert --dataset ./data_utils/ \
     --audio_feat data_utils/aud_hu.npy --save_path result.mp4 --checkpoint your_trained_ckpt.pth
   ```

---

## 推理与部署选项
- **PyTorch 离线推理**：`inference.py`
- **ONNX Runtime HTTP 服务**：`onnxrt_service.py`，文档见 `docs/onnxrt_service.md`
- **C++ / MNN / TensorRT 示例**：
  - `example/cpp` + `docs/cpp_example.md`
  - `example/mnn/android`
  - `example/trt` + `docs/tensorrt_inference.md`
    - 使用 `pth2onnx.py` 导出 ONNX
    - `example/trt/build_engine.py` 生成 `.plan`（可启用 FP16）
    - `example/trt/infer.py` 在 TensorRT 与 CPU 之间自动回退

---

## 故障排查
- **帧率不匹配**：HuBERT/AVE 必须使用 25fps 源视频，Wenet 使用 20fps。
- **Landmark 模型缺失**：`scrfd_2.5g_kps.onnx` 需手动放到 `data_utils/`。
- **TensorRT Python 运行时报错**：确认 `cuda-python`、TensorRT 版本匹配，并设置 `PYTHONPATH` 指向系统 TensorRT 安装目录。若遇到 `cudaErrorInsufficientDriver`，可先用 `/usr/src/tensorrt/bin/trtexec` 验证引擎，随后升级驱动或切换合适的 CUDA 运行库。

---

## 模型实验记录

> 以下记录用于跟踪骨干网络、Landmark 模型与平滑策略的组合性能。推荐在复现时依据同样顺序对比。

### A. 初始实验（原始数据集）
1. 权重 + UNet + VGG19 + PFLD Landmark（嘴型对齐）
2. 权重 + UNet 改造 + MobileNetV3 + FA Landmark
3. 权重 + UNet Plus + MobileNetV3 + FA Landmark
4. UNet 改造 + MobileNetV3 + FA Landmark
5. UNet + MobileNetV3 + FA Landmark（轻微右下角）
6. UNet Plus + MobileNetV3 + PFLD Landmark（嘴型对齐）
7. 权重 + UNet Plus + ResNet50 + FA Landmark
8. UNet + MobileNetV3 + FA Landmark + Weight（表现优秀）
9. UNet + MobileNetV3 + FA Landmark + Weight + Smooth Filter
10. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter（表现优秀）
11. UNet + MobileNetV3 + FA Landmark + Smooth Filter（轻微右下角）
12. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + Smooth Filter（优秀，但人脸框异常）
13. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter + 新数据（优秀）
14. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + 新数据 + Smooth Filter（表现良好）
15. UNet 改 + MobileNetV3 + FA Landmark + Weight + 新数据 + Smooth Filter
16. UNet + MobileNetV3 + FA Landmark + Weight + 新数据 + Smooth Filter（版本 3）

### B. 重新测试（引入新数据）
1.0 UNet + VGG19 + PFLD Landmark + 新数据（效果欠佳）  
1.1 权重 + UNet + VGG19 + PFLD Landmark + 新数据（效果欠佳）  
1.2 UNet + ResNet50 + PFLD Landmark + Weight + 新数据（略有提升）  
2.  权重 + UNet 改造 + MobileNetV3 + FA Landmark  
3.  权重 + UNet Plus + MobileNetV3 + FA Landmark  
4.  UNet 改造 + MobileNetV3 + FA Landmark  
5.  UNet + MobileNetV3 + FA Landmark（轻微右下角）  
6.  UNet Plus + MobileNetV3 + PFLD Landmark（嘴型对齐）  
7.  权重 + UNet Plus + ResNet50 + FA Landmark  
8.  UNet + MobileNetV3 + FA Landmark + Weight（优秀）  
9.  UNet + MobileNetV3 + FA Landmark + Weight + Smooth Filter  
10. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter（优秀）  
11. UNet + MobileNetV3 + FA Landmark + Smooth Filter（轻微右下角）  
12. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + Smooth Filter（优秀，但人脸框异常）  
13. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter + 新数据（优秀）  
14. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + 新数据 + Smooth Filter（表现良好）

---

## 参考文档
- `docs/mobile_inference.md`：手机端 ONNX/MNN 部署指南
- `docs/cpp_example.md`：C++ ONNX Runtime 推理示例
- `docs/onnxrt_service.md`：HTTP 推理服务说明
- `docs/tensorrt_inference.md`：TensorRT 引擎构建、FP16 加速与 CPU 回退流程

欢迎针对不同部署场景扩展示例或贡献新的实验记录。若在复现过程中遇到问题，可在上述排查建议基础上进一步定位。***
