# 手机端 ONNX/MNN 推理部署指南

本文档演示如何将 `Ultralight-Digital-Human` 训练得到的权重导出为 ONNX/MNN，并在手机端完成推理集成。整体流程分为三部分：

1. 导出并校验 ONNX 模型  
2. 可选：转换为 MNN 模型  
3. 在移动端接入模型，并实现与仓库推理脚本一致的前/后处理

---

## 1. 导出 PyTorch 权重到 ONNX

先安装依赖（建议在虚拟环境中执行）：

```bash
pip install torch onnx onnxruntime numpy
```

使用仓库自带的 `pth2onnx.py` 完成导出：

```bash
python pth2onnx.py \
  --checkpoint checkpoints/ultralight_dh.pth \  # 训练得到的 .pth
  --output export/unet_hubert.onnx \            # 导出的 ONNX
  --variant base \                              # base=160 模型, 328=320 模型
  --asr hubert \                                # 训练时采用的语音分支: hubert | wenet
  --opset 11                                    # ONNX Opset，可根据需求调整
```

> 若权重来自 `train_328.py` 且使用 AVE 编码器，请改用：
>
> ```bash
> python pth2onnx.py \
>   --checkpoint checkpoint/May/120000.pth \
>   --output export/unet_ave.onnx \
>   --variant 328 \
>   --asr ave
> ```

可追加 `--verify`，自动调用 ONNX Runtime 做一次前后向比对，确保导出的模型与 PyTorch 一致。

导出时会使用以下两个静态输入：

- 图像：`input`，`base` 变体为 `[1, 6, 160, 160]`，`328` 变体为 `[1, 6, 320, 320]`
- 音频：`audio`
  - `base`: Hubert `[1, 16, 32, 32]`，Wenet `[1, 128, 16, 32]`
  - `328`: Hubert `[1, 32, 32, 32]`，Wenet `[1, 256, 16, 32]`，AVE `[1, 32, 16, 16]`

输出名称为 `output`，`base` 变体形状 `[1, 3, 160, 160]`，`328` 变体为 `[1, 3, 320, 320]`。

---

## 2. 用 Python 脚本快速验证 ONNX

可使用下述示例脚本完成一次跑通（需准备和 `inference.py` 一致的图片/音频特征）：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("export/unet_hubert.onnx", providers=["CPUExecutionProvider"])

img_concat = np.load("demo/img_concat.npy")  # 形状 (1, 6, 160, 160)
audio_feat = np.load("demo/audio_feat.npy")  # 形状 (1, 16, 32, 32)

pred = session.run(["output"], {"input": img_concat, "audio": audio_feat})[0]
np.save("demo/pred.npy", pred)
```

---

## 3. 转换为 MNN 模型（可选）

MNN 官方提供的 `MNNConvert` 工具支持直接把 ONNX 转换成 MNN。安装完成后执行：

```bash
MNNConvert \
  -f ONNX \
  --modelFile export/unet_hubert.onnx \
  --MNNModel export/unet_hubert.mnn \
  --bizCode UltralightDH
```

如需量化，可追加 `--quantBits 8` 等参数，并按照 MNN 文档提供校准数据。

---

## 4. 前处理与后处理要点

- 参考 `inference.py` 的实现完成图像与关键点处理。核心逻辑位于：
  - 图像裁剪与缩放：`mouth_roi.py`、`inference.py`
  - 关键点平滑：`smooth_filter.py`（可选）
  - 拼接输入张量：真实口腔裁剪与遮挡图像分别归一化到 `[0,1]` 并在 channel 维拼接
- 音频特征需要提前离线提取，确保输入格式与训练阶段保持一致：
  - Hubert：`(frames, 16 * 32 * 32)` reshape 成 `[1, 16, 32, 32]`
  - Wenet：`(frames, 128 * 16 * 32)` reshape 成 `[1, 128, 16, 32]`
- 模型输出需乘以 255，再拼接回原始帧。仓库中相关逻辑在 `inference.py` 末尾。

建议把前后处理逻辑迁移到移动端本地实现，以保证与 Python 版本完全一致。

---

## 5. 手机端集成示例

### 5.1 ONNX Runtime（Android，Kotlin 示例）

```kotlin
val options = SessionOptions().apply { setExecutionMode(ExecutionMode.ORT_SEQUENTIAL) }
val env = OrtEnvironment.getEnvironment()
val modelBytes = loadAssetBytes("unet_hubert.onnx")
val session = env.createSession(modelBytes, options)

val inputTensor = OnnxTensor.createTensor(env, imgConcat)  // float32 [1,6,160,160]
val audioTensor = OnnxTensor.createTensor(env, audioFeat)  // float32 [1,16,32,32]

val results = session.run(mapOf("input" to inputTensor, "audio" to audioTensor))
val output = results.first().value as Array<FloatArray>
```

- 使用 [onnxruntime-mobile](https://onnxruntime.ai/docs/build/ios-and-android/) 版本可以显著减小包体  
- 推荐将 `unet_hubert.onnx` 放入 `app/src/main/assets`，并在打包前使用 `onnxruntime-tools` 做裁剪

### 5.2 MNN（C++，Android/iOS 通用）

```cpp
MNN::Interpreter* net = MNN::Interpreter::createFromFile("unet_hubert.mnn");
MNN::ScheduleConfig cfg;
cfg.type = MNN_FORWARD_AUTO;
cfg.numThread = 4;
auto session = net->createSession(cfg);

auto input = net->getSessionInput(session, "input");
auto audio = net->getSessionInput(session, "audio");
net->resizeTensor(input, {1, 6, 160, 160});
net->resizeTensor(audio, {1, 16, 32, 32});
net->resizeSession(session);

auto inputPtr = input->host<float>();
auto audioPtr = audio->host<float>();
// 拷贝前处理后的数据至 inputPtr / audioPtr

net->runSession(session);
auto output = net->getSessionOutput(session, "output");
```

- 若目标平台缺少 GPU，可选择 CPU 推理并开启 `MNN::BackendConfig::Power_Normal` 折中能耗  
- 在量化模型时注意同步使用 int8 前处理

---

## 6. 常见问题与调试建议

- **导出失败**：确认 `torch`, `onnx`, `onnxruntime`, `numpy` 版本兼容；若显存不足，可在 CPU 上导出  
- **精度下降**：检查前处理是否与 `inference.py` 完全一致；验证时建议先在 PC 上对比 PyTorch、ONNX、MNN 输出  
- **性能不足**：开启 ONNX Runtime NNAPI / CoreML Delegate，或使用 MNN GPU 后端；可考虑使用量化 INT8 模型  
- **资源打包**：建议将模型、字典类文件存放在应用私有目录，首次启动时复制到可读写路径

---

至此，您即可将训练好的权重部署到手机端完成实时或离线推理。如需进一步的自动化脚本（例如批量导出/转换），可以根据上述命令封装成 Shell 或 Python 程序。
