# Web 实时演示示例

本示例演示如何通过浏览器 + WebRTC 方式，使用 `server.py` 搭建一个实时语音驱动数字人服务。前端页面位于 `client.html`，包含麦克风采集、状态提示与连接统计；后端基于 `aiortc`、`aiohttp` 完成音频流接入与视频帧回传。

---

## 1. 环境准备

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   > 需要 `aiortc`、`aiohttp`、`numpy`、`opencv-python`、`torch`、`onnxruntime` 等；建议直接复用训练环境。

2. 准备模型与素材资源，将其放在同一目录（例如 `runtime_assets/`），目录结构需包含：
   ```
   runtime_assets/
     unet.onnx 或 unet_ave.onnx
     encoder.onnx / audio_encoder.onnx      # 取决于 asr 模式
     img_inference/                         # 参考图像
     lms_inference/                         # 关键点
     assets/...                             # 其它依赖
   ```

3. 若使用 HuBERT / AVE 分支，请保证输入视频及关键点对应 25 FPS；Wenet 为 20 FPS。

---

## 2. 启动服务器

```bash
python server.py \
  --data_path ./runtime_assets \
  --host 0.0.0.0 \
  --port 8080 \
  --asr hubert \
  --fps 25 \
  --use-smooth \
  --use-soft-mask
```

- `--data_path`：模型与素材所在目录。
- `--asr`：选择 `wenet` / `hubert` / `ave`。
- `--fps`：需与训练一致（`wenet`=20，`hubert/ave`=25）。
- `--use-smooth` / `--use-soft-mask` / `--use-seamless`：可按需开启增强策略。

成功启动后，控制台会显示访问地址，例如 `http://127.0.0.1:8080/`。

---

## 3. 打开前端页面

1. 浏览器访问服务器地址，若部署在本机可直接打开：
   ```
   http://127.0.0.1:8080/
   ```

2. 页面功能说明：
   - **开始**：请求麦克风权限，建立 WebRTC 连接，并开始推理。
   - **停止**：主动断开连接、释放资源。
   - **状态栏**：展示连接状态（未连接 / 连接中 / 已连接 / 失败）。
   - **连接信息**：显示实时帧率、分辨率、音频状态等统计信息。

3. 建议在 Chrome / Edge (Chromium) 等支持 WebRTC 的浏览器中使用。

---

## 4. 调试建议

- 若页面无法连接，检查终端日志是否成功处理 `/offer` 请求。
- 确保 STUN 服务器可用（默认 `stun.l.google.com:19302`），或在内网环境配置 TURN。
- 如需自定义前端样式或文案，可直接编辑 `client.html`。
- 若输出视频存在卡顿，可在服务器端关闭 `--use-soft-mask` 等耗时操作，或使用更强 GPU。

---

## 5. 目录结构

```
example/web/
  README.md      # 当前说明
  client.html    # 浏览器端页面
```

服务器端仍使用仓库根目录的 `server.py`；若希望在其它项目复用，可将二者一起复制，并按需修改资源路径。欢迎结合 `docs/onnxrt_service.md` / `docs/tensorrt_inference.md` 等文档拓展更多部署形态。***
