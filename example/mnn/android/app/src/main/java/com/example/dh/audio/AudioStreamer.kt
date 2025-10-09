package com.example.dh.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch

class AudioStreamer(
    private val scope: CoroutineScope,
    private val sampleRate: Int = 16000
) {
    private val chunkSamples = 160
    private val minBufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    private val record = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        minBufferSize
    )

    private val _chunks = MutableSharedFlow<ShortArray>(
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val chunks: SharedFlow<ShortArray> = _chunks

    private var job: Job? = null

    fun start() {
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            throw IllegalStateException("AudioRecord init failed")
        }
        if (job?.isActive == true) return
        record.startRecording()
        job = scope.launch {
            val buffer = ShortArray(chunkSamples)
            while (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val read = record.read(buffer, 0, buffer.size)
                if (read == buffer.size) {
                    val chunk = buffer.copyOf()
                    _chunks.emit(chunk)
                }
            }
        }
    }

    fun stop() {
        job?.cancel()
        record.stop()
    }

    fun release() {
        stop()
        record.release()
    }
}
