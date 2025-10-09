# Android MNN 示例

此示例展示如何在 Android 端使用 **MNN** 推理 `Ultralight Digital Human` 模型，可作为移动端集成的起点。示例代码仅包含推理关键逻辑和必要的项目结构片段，需在现有 App 工程中按需补全。

---

## 1. 目录结构

```
example/mnn/android/
├── README.md
├── app/
│   ├── build.gradle.kts         # Gradle 片段
│   ├── src/main/cpp/            # JNI + MNN 推理引擎
│   │   ├── CMakeLists.txt
│   │   ├── digital_human_engine.cpp/hpp
│   │   ├── digital_human_jni.cpp
│   │   └── npy_reader.cpp/hpp
│   └── src/main/java/com/example/dh/
│       ├── MainActivity.kt       # 麦克风采集 + 渲染
│       ├── DigitalHumanNative.kt # JNI 封装
│       ├── audio/AudioStreamer.kt
│       ├── audio/AudioPlayer.kt
│       └── util/AssetUtils.kt
```

> 若要运行，需要将文件拷贝或参考到实际 Android 工程，并补齐标准的 Gradle/Manifest 结构。

---

## 2. 准备模型与资源

1. 将导出的 `.mnn` 模型放入 `app/src/main/assets/models/`，示例针对 AVE：
   - `models/unet_ave.mnn`
   - `models/audio_encoder.mnn`
2. 将推理素材放入 `app/src/main/assets/dataset/`，包括：
   - `full_body_img/0.jpg ...`（328 版本 ROI）
   - `landmarks/0.lms ...`
   - 如需离线校验，可放置 `aud_ave.npy`；实时模式可为空。
3. 若切换其它 ASR 分支，请同步替换模型和初始化参数。

```
app/src/main/assets/
 ├── dataset/
 │   ├── full_body_img/0.jpg ...
 │   └── landmarks/0.lms ...
 └── models/
     ├── unet_ave.mnn
     └── audio_encoder.mnn
```

AndroidManifest 需声明麦克风权限：

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

---

## 3. build.gradle / CMake 关键配置

```kotlin
// app/build.gradle.kts
plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.dh"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.dh"
        minSdk = 24
        targetSdk = 34
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
        }
    }

    packagingOptions {
        pickFirst("lib/*/libMNN.so")
        pickFirst("lib/*/libMNN_CL.so")
        pickFirst("lib/*/libMNN_Vulkan.so")
    }
}

android {
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}

> 将 `libMNN.so`（各 ABI）置于 `app/src/main/jniLibs/<abi>/`，同时在 `third_party/MNN/include/` 放置头文件；OpenCV Android SDK 需在工程 `settings.gradle` 中配置 `OpenCV_DIR` 或通过 `sdk/native/jni` 路径导入。

```
app/
  src/main/jniLibs/arm64-v8a/libMNN.so
  src/main/jniLibs/armeabi-v7a/libMNN.so
third_party/MNN/include/**        # MNN 头文件
OpenCV-android-sdk/sdk/native/jni # OpenCV NDK 目录（设置 OpenCV_DIR 指向此路径）
```
```

> 请替换 `MNN-android-*.aar` 为官方发布的实际版本。若使用 jCenter/mavenCentral 镜像，也可通过 `implementation("com.tencent.mnn:mnn-android:<version>")` 引入。

---

## 4. Kotlin 推理封装

核心流程由 JNI `DigitalHumanNative` 驱动：

1. `AudioStreamer` 采集 16kHz PCM，每 10ms 将 160 采样推送到 JNI。
2. C++ 层执行 Mel Spectrogram → AVE `audio_encoder.mnn` → 512 维特征 → 组装 `[1, 32, 16, 16]` → `unet_ave.mnn`，并返回 JPEG 帧与可选音频回放数据。
3. `MainActivity` 将 JPEG 解码更新界面；若需要声音，可将 `audioPcm` 接入 `AudioTrack`。

---

## 5. 示例 Activity

`MainActivity.kt` 已示范如何加载资产、初始化 JNI 引擎、启动麦克风流并在 UI 上展示结果，可直接扩展为摄像头 + ROI 实时管线（需自行加入人脸关键点提取）。核心逻辑如下：

```kotlin
private fun startStreaming() {
    val engine = nativeEngine ?: return
    audioPlayer = AudioPlayer()
    audioStreamer = AudioStreamer(lifecycleScope).also { streamer ->
        lifecycleScope.launch {
            streamer.chunks.collectLatest { chunk ->
                val result = engine.process(chunk)
                val jpeg = result?.frameJpeg ?: return@collectLatest
                val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
                val audio = result.audioPcm
                withContext(Dispatchers.Main) {
                    preview.setImageBitmap(bitmap)
                    audio?.let { audioPlayer?.enqueue(it) }
                }
            }
        }
    }
    if (hasAudioPermission()) {
        audioStreamer?.start()
    }
}
```

- `AudioStreamer`：基于 `AudioRecord` 把麦克风数据切成 10ms（160 样本）块。
- `DigitalHumanNative`：JNI 封装，调用 C++/MNN 完成音频特征提取与口型生成。
- `AudioPlayer`：使用 `AudioTrack` 播放返回的 PCM 音频，实现嘴型与声音同步。
- `MainActivity` 自动请求 `RECORD_AUDIO` 权限，并在授权后加载资产 (`AssetUtils`) 初始化运行时资源。

---

## 6. 下一步

1. **音频编码器导出**：使用统一脚本 `tools/export_models.py` 将 `model/checkpoints/audio_visual_encoder.pth` 导出并转换：

   ```bash
   python tools/export_models.py \
     --component audio_encoder \
     --checkpoint model/checkpoints/audio_visual_encoder.pth \
     --onnx audio_encoder.onnx \
     --mnn audio_encoder.mnn \
     --mnnconvert /path/to/MNNConvert
   ```

2. **关键点与 ROI**：示例默认使用 assets 中的离线数据，实际项目需集成人脸检测/关键点模型（可在 JNI 层实现）以对接摄像头帧。
3. **音频回放**：JNI 会返回 `audioPcm`，可接入 `AudioTrack` 实现嘴型与声音同步。
4. **性能调优**：MNN 支持 OpenCL/Vulkan，可在 `digital_human_engine.cpp` 的 `ScheduleConfig` 中配置 GPU 后端，并结合 NEON/量化优化移动端性能。

---

至此，示例工程提供了一个可以在 Android 上直接调用 `.mnn` 模型生成嘴部贴图的最小实现。可在此基础上扩展为实时数字人应用。***
