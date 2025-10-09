#include <jni.h>
#include <android/log.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "digital_human_engine.hpp"

#define LOG_TAG "DHNativeJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

jstring get_string(JNIEnv* env, jstring js) {
    return js;
}

std::string jstring_to_string(JNIEnv* env, jstring js) {
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string str(chars);
    env->ReleaseStringUTFChars(js, chars);
    return str;
}

template <typename T>
T* from_handle(jlong handle) {
    return reinterpret_cast<T*>(handle);
}

jobject build_result(JNIEnv* env, const DigitalHumanEngine::Result& result) {
    jobjectArray arr = env->NewObjectArray(2, env->FindClass("java/lang/Object"), nullptr);

    if (result.has_frame && !result.frame.empty()) {
        std::vector<uint8_t> buffer;
        cv::imencode(".jpg", result.frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 95});
        jbyteArray frameArray = env->NewByteArray(buffer.size());
        env->SetByteArrayRegion(frameArray, 0, buffer.size(), reinterpret_cast<const jbyte*>(buffer.data()));
        env->SetObjectArrayElement(arr, 0, frameArray);
        env->DeleteLocalRef(frameArray);
    }

    if (!result.audio.empty()) {
        jshortArray audioArray = env->NewShortArray(result.audio.size());
        env->SetShortArrayRegion(audioArray, 0, result.audio.size(), result.audio.data());
        env->SetObjectArrayElement(arr, 1, audioArray);
        env->DeleteLocalRef(audioArray);
    }
    return arr;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_dh_DigitalHumanNative_nativeCreate(
    JNIEnv* env,
    jobject /*thiz*/,
    jstring datasetDir,
    jstring modelPath,
    jstring audioFeatPath,
    jstring audioModelPath,
    jstring asr,
    jstring variant,
    jdouble fps,
    jboolean useSmooth,
    jboolean useSoftMask,
    jboolean useSeamless) {

    try {
        DigitalHumanEngine::Options opt;
        opt.dataset_dir = jstring_to_string(env, datasetDir);
        opt.model_path = jstring_to_string(env, modelPath);
        opt.audio_feat_path = jstring_to_string(env, audioFeatPath);
        opt.audio_model_path = jstring_to_string(env, audioModelPath);
        opt.asr = jstring_to_string(env, asr);
        opt.variant = jstring_to_string(env, variant);
        opt.fps = fps;
        opt.use_smooth = useSmooth;
        opt.use_soft_mask = useSoftMask;
        opt.use_seamless = useSeamless;

        auto* engine = new DigitalHumanEngine(opt);
        return reinterpret_cast<jlong>(engine);
    } catch (const std::exception& e) {
        LOGE("nativeCreate failed: %s", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_dh_DigitalHumanNative_nativeReset(
    JNIEnv* /*env*/,
    jobject /*thiz*/,
    jlong handle) {
    if (auto* engine = from_handle<DigitalHumanEngine>(handle)) {
        engine->reset();
    }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_dh_DigitalHumanNative_nativeProcess(
    JNIEnv* env,
    jobject /*thiz*/,
    jlong handle,
    jshortArray audioChunk) {

    auto* engine = from_handle<DigitalHumanEngine>(handle);
    if (!engine) {
        LOGE("nativeProcess called with null engine");
        return nullptr;
    }
    jsize len = env->GetArrayLength(audioChunk);
    std::vector<int16_t> pcm(static_cast<size_t>(len));
    env->GetShortArrayRegion(audioChunk, 0, len, pcm.data());

    try {
        auto result = engine->process(pcm.data(), pcm.size());
        return build_result(env, result);
    } catch (const std::exception& e) {
        LOGE("nativeProcess failed: %s", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_dh_DigitalHumanNative_nativeDestroy(
    JNIEnv* /*env*/,
    jobject /*thiz*/,
    jlong handle) {
    if (auto* engine = from_handle<DigitalHumanEngine>(handle)) {
        delete engine;
    }
}
