package com.example.dh

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class DigitalHumanNative(
    context: Context,
    private val datasetDir: File,
    private val modelPath: File,
    private val audioFeat: File?,
    private val audioEncoder: File?,
    private val asr: String,
    private val variant: String,
    private val fps: Double,
    private val useSmooth: Boolean,
    private val useSoftMask: Boolean,
    private val useSeamless: Boolean
) {
    private var handle: Long = 0

    init {
        System.loadLibrary("dh_native")
        require(datasetDir.exists()) { "datasetDir missing: ${datasetDir.absolutePath}" }
        require(modelPath.exists()) { "model missing: ${modelPath.absolutePath}" }
        if (audioEncoder != null) {
            require(audioEncoder.exists()) { "audio encoder missing: ${audioEncoder.absolutePath}" }
        }
        if (audioFeat != null) {
            require(audioFeat.exists()) { "audio feature missing: ${audioFeat.absolutePath}" }
        }
        handle = nativeCreate(
            datasetDir.absolutePath,
            modelPath.absolutePath,
            audioFeat?.absolutePath ?: "",
            audioEncoder?.absolutePath ?: "",
            asr,
            variant,
            fps,
            useSmooth,
            useSoftMask,
            useSeamless
        )
        require(handle != 0L) { "Failed to create native engine" }
    }

    suspend fun process(pcm: ShortArray): ProcessResult? = withContext(Dispatchers.Default) {
        @Suppress("UNCHECKED_CAST")
        val arr = nativeProcess(handle, pcm) as? Array<Any?> ?: return@withContext null
        val frame = arr.getOrNull(0) as? ByteArray
        val audio = arr.getOrNull(1) as? ShortArray
        ProcessResult(frame, audio)
    }

    fun reset() {
        nativeReset(handle)
    }

    fun release() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0
        }
    }

    data class ProcessResult(
        val frameJpeg: ByteArray?,
        val audioPcm: ShortArray?
    )

    private external fun nativeCreate(
        datasetDir: String,
        modelPath: String,
        audioFeatPath: String,
        audioModelPath: String,
        asr: String,
        variant: String,
        fps: Double,
        useSmooth: Boolean,
        useSoftMask: Boolean,
        useSeamless: Boolean
    ): Long

    private external fun nativeProcess(handle: Long, pcm: ShortArray): Array<Any?>?
    private external fun nativeReset(handle: Long)
    private external fun nativeDestroy(handle: Long)
}
