# Ultralight Digital Human

本项目提供从数据预处理、加权采样训练、到多种推理部署（PyTorch、ONNX Runtime、TensorRT、MNN）的完整流程。本 README 汇总常用命令及实验记录，详细部署说明见 `docs/`。

---

## 1. 环境准备

```bash
conda activate dh
```

确保已安装项目依赖（PyTorch、opencv-python、numpy、onnxruntime 等）。

---

## 2. 数据预处理

1. 生成训练所需的音频、图像、关键点等数据：
   ```bash
   python process.py test.mp4 --asr hubert
   ```
   - `--asr` 支持 `hubert` / `wenet` / `ave`，用于决定音频特征形状。
   - 输出保存在 `data_utils/` 目录，可用于后续同步网（SyncNet）与 UNet 训练。

2. 若需为不同 ASR 分支准备数据，可重复执行：
   ```bash
   python process.py test.mp4 --asr wenet
   python process.py test.mp4 --asr ave
   ```

---

## 3. 加权采样训练流水线

### 3.1 覆盖均衡（Coverage Re-weighting）

```bash
python data_utils/build_sampling_weights.py --root ./data_utils --asr hubert --k 50
python data_utils/build_sampling_weights.py --root ./data_utils --asr wenet --k 50
```

### 3.2 训练 SyncNet 判别器

```bash
cd ..
python syncnet.py --save_dir ./syncnet_hubert/ --dataset_dir ./data_utils/ --asr hubert
python syncnet.py --save_dir ./syncnet_wenet/  --dataset_dir ./data_utils/ --asr wenet
python syncnet.py --save_dir ./syncnet_ave/    --dataset_dir ./data_utils/ --asr ave
```

### 3.3 困难样本挖掘（Hard Mining，自动融合 A1）

```bash
python data_utils/build_hard_weights.py --root ./data_utils --asr hubert --ckpt ./syncnet_ckpt/39.pth
python data_utils/build_hard_weights.py --root ./data_utils --asr wenet  --ckpt ./syncnet_/39.pth
```

> `--ckpt` 指向前一步训练得到的 SyncNet 权重文件。

### 3.4 加权采样训练 UNet

```bash
python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_hubert/ --asr hubert \
  --use_syncnet --syncnet_checkpoint ./syncnet_hubert/39.pth --weighted_sampling

python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_wenet/ --asr wenet \
  --use_syncnet --syncnet_checkpoint ./syncnet_wenet/39.pth --weighted_sampling

python train_328.py --dataset_dir ./data_utils/ --save_dir ./checkpoint_ave/ --asr ave \
  --use_syncnet --syncnet_checkpoint ./syncnet_ave/39.pth --weighted_sampling
```

---

## 4. 音频特征提取与推理

1. 提取 Hubert 音频特征（其他分支类似）：
   ```bash
   cd data_utils/
   python hubert.py --wav aud.wav
   ```

2. 运行离线推理：
   ```bash
   python inference.py --asr hubert --dataset ./data_utils/ \
     --audio_feat data_utils/aud_hu.npy --save_path a.mp4 --checkpoint checkpoint/best.pth
   ```

3. 将生成的视频与原始音频合并：
   ```bash
   ffmpeg -i a.mp4 -i aud.wav -c:v libx264 -c:a aac atest.mp4
   ```

4. 若需要测试其他训练权重，可重复执行：
   ```bash
   python inference.py --asr hubert --dataset ./data_utils/ \
     --audio_feat data_utils/aud_hu.npy --save_path output.mp4 --checkpoint your_trained_ckpt.pth
   ```

---

## 5. 精炼流程（示例：Hubert 分支）

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

## 6. 模型实验记录

### 6.1 初始实验（原始数据）

1. 权重 + UNet + VGG19 + PFLD Landmark（对齐嘴型）
2. 权重 + UNet 改造 + MobileNetV3 + FA Landmark
3. 权重 + UNet Plus + MobileNetV3 + FA Landmark
4. UNet 改造 + MobileNetV3 + FA Landmark
5. UNet + MobileNetV3 + FA Landmark（轻微右下角）
6. UNet Plus + MobileNetV3 + PFLD Landmark（对齐嘴型）
7. 权重 + UNet Plus + ResNet50 + FA Landmark
8. UNet + MobileNetV3 + FA Landmark + Weight（优秀）
9. UNet + MobileNetV3 + FA Landmark + Weight + Smooth Filter
10. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter（优秀）
11. UNet + MobileNetV3 + FA Landmark + Smooth Filter（仍有轻微下角）
12. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + Smooth Filter（优秀，但人脸框异常）
13. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter + 新数据（优秀）
14. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + 新数据 + Smooth Filter（表现良好）
15. UNet 改 + MobileNetV3 + FA Landmark + Weight + 新数据 + Smooth Filter
16. UNet + MobileNetV3 + FA Landmark + Weight + 新数据 + Smooth Filter（版本 3）

### 6.2 重新测试（新数据集）

1.0 UNet + VGG19 + PFLD Landmark + 新数据（效果欠佳）  
1.1 权重 + UNet + VGG19 + PFLD Landmark + 新数据（效果欠佳）  
1.2 UNet + ResNet50 + PFLD Landmark + Weight + 新数据（略有提升）  
2.  权重 + UNet 改造 + MobileNetV3 + FA Landmark  
3.  权重 + UNet Plus + MobileNetV3 + FA Landmark  
4.  UNet 改造 + MobileNetV3 + FA Landmark  
5.  UNet + MobileNetV3 + FA Landmark（轻微右下角）  
6.  UNet Plus + MobileNetV3 + PFLD Landmark（对齐嘴型）  
7.  权重 + UNet Plus + ResNet50 + FA Landmark  
8.  UNet + MobileNetV3 + FA Landmark + Weight（优秀）  
9.  UNet + MobileNetV3 + FA Landmark + Weight + Smooth Filter  
10. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter（优秀）  
11. UNet + MobileNetV3 + FA Landmark + Smooth Filter（仍有轻微下角）  
12. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + Smooth Filter（优秀，但人脸框异常）  
13. UNet 改 + MobileNetV3 + FA Landmark + Weight + Smooth Filter + 新数据（优秀）  
14. UNet 改 + MobileNetV3 + PFLD Landmark + Weight + 新数据 + Smooth Filter（表现良好）

> 备注：Landmark 质量会显著影响成片效果；请确保人脸检测框正确。

---

## 7. 示例与文档

- `example/cpp`：ONNX Runtime C++ 离线推理  
- `example/mnn`：移动端 MNN Demo  
- `example/trt`：TensorRT FP16 加速示例，含 CPU 回退脚本  
- `docs/mobile_inference.md`：手机端部署  
- `docs/cpp_example.md`：C++ 推理  
- `docs/onnxrt_service.md`：ONNX Runtime HTTP 服务  
- `docs/tensorrt_inference.md`：TensorRT 引擎构建与推理流程
 
