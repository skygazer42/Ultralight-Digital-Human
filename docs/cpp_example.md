# C++ ONNX Runtime 推理示例

本示例位于 `example/cpp/`，展示如何使用 C++ + ONNX Runtime 实现离线合成，与 Python 版本的 `inference.py`、`inference_328.py` 对应。

---

## 1. 准备工作

1. **安装依赖**
   - ONNX Runtime C/C++ SDK（解压后得到 `include/`、`lib/` 或 `lib64/`）
   - OpenCV（用于图像处理、视频写出）
   - CMake ≥ 3.14、C++17 编译器

2. **准备模型与数据**
   - 将 `unet.onnx`（或 `unet_ave.onnx`）放入推理目录
   - 数据集目录结构需包含：
     ```
     dataset/
       full_body_img/0.jpg ...
       landmarks/0.lms ...
       aud_xxx.npy          # 由 python 预处理脚本导出的音频特征
     ```

---

## 2. 编译

```bash
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

可通过 `-DOpenCV_DIR=/path/to/opencv` 指定 OpenCV 安装位置（如未自动检测）。

---

## 3. 运行

```bash
./ultralight_dh_onnx \
  --data /path/to/dataset \
  --audio /path/to/aud_hu.npy \
  --unet /path/to/unet.onnx \
  --output result.mp4 \
  --variant base \
  --asr hubert \
  --fps 25 \
  --use-smooth \
  --use-soft-mask \
  --use-seamless
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--data` | 数据集根目录（含 `full_body_img/`、`landmarks/`） |
| `--audio` | 音频特征 `.npy`（与 Python 预处理保持一致） |
| `--unet` | 导出的 UNet ONNX 模型 |
| `--output` | 输出 MP4 路径 |
| `--variant` | `base`（160×160）或 `328`（320×320） |
| `--asr` | `wenet` / `hubert` / `ave`，决定音频张量尺寸 |
| `--fps` | 输出帧率（`wenet`=20，`hubert/ave`=25） |
| `--use-smooth` | 启用一阶欧拉滤波平滑关键点 |
| `--use-soft-mask` | 启用软边缘混合 |
| `--use-seamless` | 使用 `seamlessClone` 粘贴下巴区域 |

> 注意：若使用 `ave` 模型，请同时设置 `--variant 328 --fps 25`。

---

## 4. 代码结构

- `main.cpp`：完整的推理流程（加载数据、构建 6 通道输入、ONNX 推理、后处理、视频写出）
- `CMakeLists.txt`：编译示例所需的构建脚本

主要逻辑与 Python 版保持一致，可作为迁移到移动端（Android/iOS C++ 层）的参考实现。

---

如需进一步自定义（例如改为流式输入、对接自定义音频编码器），可在该示例的基础上扩展。***
