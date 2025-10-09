# 项目运行全流程策略（训练 → 推理 → 多端部署）

本文档给出从数据准备、模型训练、推理测试到导出 ONNX/MNN/TensorRT 并在多端部署的详细步骤，帮助团队高效落地项目。

---

## 1. 总览
- **核心模型**：基于 UNet 的图像渲染（`unet.py` / `unet_328.py`）+ 音频编码器（Hubert / Wenet / AVE）。
- **训练产物**：`checkpoint/<角色名>/<step>.pth` 模型权重。
- **推理脚本**：`inference.py`（160 输入）、`inference_328.py`（328 输入）。
- **导出工具**：`pth2onnx.py` 与新增的 `tools/export_models.py`。
- **跨端部署**：ONNX Runtime（C++/Python）、MNN（Android/C++）、TensorRT（C++/Python）。

---

## 2. 环境准备
1. **基础环境**
   - Ubuntu 20.04 / Windows 10，建议使用 NVIDIA GPU + CUDA 11.x。
   - Python ≥ 3.8，安装依赖：
     ```bash
     pip install -r requirements.txt
     pip install onnx onnxruntime onnxsim  # 导出验证
     pip install mnn                       # 可选，Python 下快速验证
     ```
   - Hugging Face 模型（HuBERT）需可联网或提前离线缓存。

2. **工具链**
   - MNN：下载对应平台的 `MNNConvert` 和运行时库。
   - TensorRT：安装 NVIDIA TensorRT 8.x，确保 `trtexec` 在 PATH 中。
   - Android：准备 Android Studio + NDK + CMake（若部署到移动端）。

---

## 3. 数据准备
1. **素材采集**
   - 高清视频（目标角色），16kHz 单声道音频。
   - 对齐的关键点、分割图、原始图像帧。

2. **数据预处理流程**
   - `data_utils/detect_face.py`：人脸检测与裁剪。
   - `data_utils/fa_landmark.py`：关键点提取。
   - `data_utils/ave/test_w2l_audio.py`或`data_utils/hubert.py`/`data_utils/wenet_infer.py`：生成音频特征。
   - `smooth_filter.py`、`mouth_roi.py`：可选的口腔 ROI 与平滑。

3. **目录结构标准**
   ```
   dataset/
     └── May/
         ├── full_body_img/   # 0.jpg, 1.jpg, ...
         ├── landmarks/       # 0.lms, 1.lms, ...
         ├── parsing/         # 可选语义分割 png
         ├── aud.wav          # 原始语音，可用于后期合成
         └── ...              # 其他中间特征
   checkpoint/
     └── May/
         ├── 100000.pth
         └── ...
   ```

---

## 4. 模型训练阶段
1. **配置检查**
   - 根据输入尺寸选择脚本：`train.py`（160）或 `train_328.py`。
   - 调整训练参数（batch size、学习率、数据路径）：
     ```python
     # train_328.py 示例位置
     parser.add_argument('--dataset_root', default='./dataset/May')
     parser.add_argument('--checkpoint_root', default='./checkpoint/May')
     ```

2. **启动训练**
   ```bash
   python train_328.py \
     --dataset_root ./dataset/May \
     --checkpoint_root ./checkpoint/May \
     --epochs 200 \
     --save_interval 5000
   ```
   - 训练日志与模型保存在 `checkpoint/<name>`。
   - 建议使用 TensorBoard 监控（如脚本支持）。

3. **训练完成后的操作**
   - 确认最新 `.pth` 权重能在推理脚本中正确加载。
   - 对关键 Checkpoint 做归档（含训练参数 JSON、副本上传存档）。

---

## 5. 推理阶段（PyTorch 基线）
1. **基础推理**
   ```bash
   python inference_328.py \
     --name May \
     --audio_path demo/talk_hb.wav \
     --asr ave \
     --use_smooth \
     --use_soft_mask \
     --use_seamless
   ```
   - `--name`：对应 `dataset/<name>` 与 `checkpoint/<name>`。
   - 可选 `--parsing` 使用语义分割图。
   - `--smooth_fps` 控制平滑器帧率。

2. **Hubert/Wenet 特殊说明**
   - `--asr hubert`：需预先安装 `transformers` 并缓存模型。
   - `--asr wenet`：需准备 `conf/wenetspeech_unified_conformer/avg10.pt` 及词典。

3. **推理验证**
   - 检查输出目录 `result/` 是否生成目标视频。
   - 使用 `ffmpeg` 自动合成音频：脚本会尝试调用本地 `ffmpeg`。

---

## 6. 训练后模型导出

### 6.1 导出 ONNX
- **UNet 导出（推荐使用新脚本）**
  ```bash
  python tools/export_models.py \
    --component unet \
    --variant 328 \
    --asr ave \
    --checkpoint checkpoint/May/100000.pth \
    --onnx export/unet_328_ave.onnx \
    --verify
  ```
  - `--verify` 需安装 `onnxruntime`；通过 RTOL=1e-3/ATOL=1e-5 校验一致性。

- **AVE 音频编码器导出**
  ```bash
  python tools/export_models.py \
    --component audio_encoder \
    --checkpoint model/checkpoints/audio_visual_encoder.pth \
    --onnx export/audio_encoder.onnx \
    --dynamic-batch \
    --verify
  ```
  - 默认 dummy 输入为 `(1, 1, 80, 16)`，可选择动态 batch。

### 6.2 导出 MNN
```bash
python tools/export_models.py \
  --component unet \
  --variant 328 \
  --asr ave \
  --checkpoint checkpoint/May/100000.pth \
  --onnx export/unet_328_ave.onnx \
  --mnn export/unet_328_ave.mnn \
  --mnnconvert C:/SDK/MNN/MNNConvert.exe \
  --mnn-biz-code UDHuman \
  --mnn-extra --fp16
```
- `--mnn-extra` 后的参数原样追加给 MNNConvert，可用于量化或特殊配置。
- 导出后使用 Python/MNN 或 C++/MNN 做一次加载验证。

### 6.3 导出 TensorRT Engine
1. 先使用上一节得到的 ONNX。
2. 构建 engine（FP16 示例）：
   ```bash
   
  trtexec --onnx=export/unet_328_ave.onnx \
           --saveEngine=export/unet_328_ave_fp16.plan \
           --workspace=4096 \
           --fp16
   ```
3. 如需 batch 多样化，可引入 `--optShapes`、`--minShapes`、`--maxShapes`。

### 6.4 版本管理
- 将 `onnx/`, `mnn/`, `trt/` 文件夹与 `checkpoint/` 同步纳入版本目录。
- 建议命名方式：`unet_328_ave_step100k.onnx`，便于对齐训练步。
- 保留转换日志与验证脚本输出，写入 `docs/export_history.md`（可选）。

---

## 7. 多平台部署策略

### 7.1 Python 推理（快速验证）
1. 使用 `onnxruntime`：
   ```python
   import onnxruntime as ort
   import numpy as np
   session = ort.InferenceSession("export/unet_328_ave.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
   logits = session.run(None, {"image": image_tensor, "audio": audio_tensor})[0]
   ```
2. 使用 `mnn`（pip 安装版）快速跑通 `.mnn`。

### 7.2 C++ + ONNX Runtime
1. **环境搭建**：下载 ONNX Runtime release，配置 CMake：
   ```cmake
   find_package(onnxruntime REQUIRED)
   target_link_libraries(app PRIVATE onnxruntime)
   ```
2. **推理流程**：
   - 读取图像、裁剪至 328×328，复制到 `float32` buffer。
   - 加载音频特征（与训练对齐的 mel 特征）。
   - 调用 `Ort::Session::Run` 获取输出，把口腔区域贴回原图。
3. **优化建议**：
   - 使用 CUDA/TensorRT Execution Provider。
   - 对输入 tensor 使用 pinned memory，减少拷贝。

### 7.3 C++ + MNN
1. **编译 MNN**：`cmake -DMNN_BUILD_CONVERTER ...`。
2. **运行时调用**：
   ```cpp
   auto net = MNN::Interpreter::createFromFile("export/unet_328_ave.mnn");
   auto session = net->createSession(config);
   auto input = net->getSessionInput(session, "image");
   // 填充数据 → runSession → getSessionOutput
   ```
3. **移动端优化**：
   - Android 使用 `libMNN.so` + JNI/NDK。
   - 启用 Winograd、FP16、OpenCL（若目标设备支持）。

### 7.4 Android（MNN）
1. 将 `.mnn` 模型放入 `assets/`。
2. Gradle 引入 MNN AAR 或自行编译 JNI。
3. Java/Kotlin 层封装：
   - 通过 JNI 将 Bitmap 转成 NCHW tensor。
   - 调用推理，使用 GLES/OpenCL 加速。
4. 处理音频：
   - 若需端上实时 AVE 编码，可先在 PC 生成特征，或另行部署轻量化编码器。

### 7.5 TensorRT（C++/Python）
1. 使用 `trtexec` 或 `tensorrt.Builder` 生成 plan 文件。
2. C++ 加载 engine，绑定输入输出：
   ```cpp
   nvinfer1::IRuntime* runtime = createInferRuntime(logger);
   auto engine = runtime->deserializeCudaEngine(plan_data, size, nullptr);
   auto context = engine->createExecutionContext();
   ```
3. CUDA 流上执行 `enqueueV2`，处理推理结果。
4. 注意：TensorRT 默认 NCHW，需保证输入输出匹配导出的 ONNX。

---

## 8. 导出工具说明（`tools/export_models.py`）
| 参数 | 说明 |
| ---- | ---- |
| `--component {unet,audio_encoder}` | 选择导出的模型组件 |
| `--checkpoint PATH` | 指定 `.pth`，音频编码器默认读取 `model/checkpoints/audio_visual_encoder.pth` |
| `--variant {base,328}` | UNet 输入尺寸变体 |
| `--asr {hubert,wenet,ave}` | 训练时使用的音频分支 |
| `--onnx PATH` | ONNX 输出文件 |
| `--mnn PATH` | 可选，生成 `.mnn` |
| `--mnnconvert PATH` | MNNConvert 可执行文件 |
| `--mnn-biz-code` | MNN 文件中的业务标识 |
| `--mnn-extra ...` | 附加传给 MNNConvert 的参数 |
| `--verify` | 安装 onnxruntime 后可开启一致性校验 |
| `--dynamic-batch` | 音频编码器导出时打开 batch 维度动态 |

建议将常用组合写入 `scripts/` 下的 shell/bat 以便复用。

---

## 9. 质量验证清单
- [ ] 训练日志中无异常 NAN，验证集指标达标。
- [ ] 推理脚本（`inference.py` / `inference_328.py`）在目标样本上产出稳定视频。
- [ ] ONNX 导出通过 `--verify` 检查。
- [ ] MNN 模型可在 PC 上用 MNN Python/C++ 成功推理。
- [ ] TensorRT engine 在部署机上通过单帧与多帧测试。
- [ ] Android/C++ Demo 能加载模型并输出正确帧。
- [ ] 所有重要命令写入 README 或内部 Wiki，确保新同事可复现。

---

## 10. 常见问题与建议
1. **HuBERT 模型加载慢**：预先执行一次 `python data_utils/hubert.py` 缓存权重，或改成离线模型路径。
2. **Wenet 权重缺失**：从官方发布下载 `avg10.pt` 和词典后放入 `data_utils/conf/wenetspeech_unified_conformer/`。
3. **MNN 精度下降明显**：确保转换时关闭不必要的量化；使用 `--fp16` 时先做验证。
4. **TensorRT 运行爆显存**：调整 `--workspace`，使用 `--fp16` 或 INT8 校准。
5. **移动端性能不足**：考虑裁剪 UNet 通道、降低分辨率，或改用更轻量的音频编码方案。

---

通过以上策略，可从训练到多端部署形成完整闭环，便于快速验证与团队协同。如果后续扩展新的 ASR 分支或模型变体，建议同步更新本文档与对应的导出脚本，确保流程一致。​យ
