# ONNX Runtime HTTP 服务

本指南演示如何使用纯 ONNX Runtime 搭建一个轻量级的数字人推理服务，便于后续迁移到移动端或其它推理框架中。脚本位于仓库根目录的 `onnxrt_service.py`。

---

## 1. 依赖安装

```bash
pip install fastapi uvicorn[standard] numpy opencv-python onnxruntime
```

确保已经按照主仓库 README 的说明准备好模型文件（例如 `data_path` 下的 `unet.onnx` / `unet_ave.onnx`、`encoder.onnx 或 audio_encoder.onnx`、`img_inference/`、`lms_inference/` 等）。

---

## 2. 启动服务

```bash
python onnxrt_service.py \
  --data-path ./runtime_assets \
  --asr wenet \
  --fps 20 \
  --use-smooth \
  --use-soft-mask \
  --use-seamless \
  --host 0.0.0.0 \
  --port 8000
```

常用参数：

- `--data-path`：推理素材目录，需包含 `img_inference/`、`lms_inference/`、ONNX 模型等资源。
- `--asr`：音频分支。`wenet`（fps=20）、`hubert`（fps=25）、`ave`（fps=25）。
- `--fps`：视频帧率，需与训练时一致。
- `--use-smooth` / `--use-soft-mask` / `--use-seamless`：启用关键点平滑、软边缘融合、`seamlessClone` 贴合。

---

## 3. 接口说明

服务启动后暴露三个 HTTP 接口：

| 方法 | 路径       | 作用                                   |
|------|------------|----------------------------------------|
| GET  | `/ping`    | 健康检查                               |
| POST | `/reset`   | 重置内部状态（音频缓冲、滤波器等）     |
| POST | `/process` | 推理 10ms 音频块，返回音频和可选帧结果 |

### 3.1 `/process` 请求/响应示例

请求体（JSON）：

```json
{
  "audio": "BASE64_ENCODED_PCM16"
}
```

- `audio`：以 `base64` 编码的 `int16` PCM 数据，长度需为 `160` 的倍数（每 10ms @ 16kHz）。

响应体：

```json
{
  "ok": true,
  "has_frame": true,
  "frame_jpeg": "BASE64_JPEG",
  "audio": "BASE64_PCM16"
}
```

- `has_frame`：指示本次调用是否生成了新的视频帧。
- `frame_jpeg`：当 `has_frame=true` 时返回合成后的 JPEG（base64 编码）。
- `audio`：模型返回的播放音频（16kHz，int16 PCM，base64 编码）。

> 提示：如果需要合成连续的视频，可自行保存 `frame_jpeg` 并按照帧率组合；返回的音频片段可串联成完整的 PCM 流。

---

## 4. 集成建议

1. **桌面验证**：先使用 `curl` 或简单脚本向服务发送离线音频数据，确认生成帧与音频逻辑正确。
2. **移动端对接**：将 `/process` 逻辑映射到手机端的麦克风采集循环，每 10ms 发送一帧音频，按需重现桌面端的后处理。
3. **状态管理**：需要切换说话人或重新开场时调用 `/reset`，避免历史缓冲影响新输入。
4. **实时 AVE**：若选择 `--asr ave`，请确保在 `data_path` 中同时提供 `audio_encoder.onnx`（或对应的 MNN 版本），并复现仓库中的 Mel 计算参数。
5. **批量模式**：若想离线批量生成，可把整段音频拆分成 160 样本块循环调用 `/process`，并在本地使用 `ffmpeg` 合并输出音频/视频。

---

至此，你即可基于纯 ONNX Runtime 提供数字人流式推理服务，并将其作为移动端或其它平台的参考实现。若需要更紧凑的部署（例如 C++/MNN），可在此基础上进一步迁移。欢迎基于脚本进行二次开发和优化。***
