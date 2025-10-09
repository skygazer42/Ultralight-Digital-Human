package com.example.dh.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack

class AudioPlayer(
    sampleRate: Int = 16000,
    private val bufferMillis: Int = 200
) {
    private val channelConfig = AudioFormat.CHANNEL_OUT_MONO
    private val encoding = AudioFormat.ENCODING_PCM_16BIT
    private val minBufferSize = AudioTrack.getMinBufferSize(sampleRate, channelConfig, encoding)
    private val track = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
        )
        .setAudioFormat(
            AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setChannelMask(channelConfig)
                .setEncoding(encoding)
                .build()
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(
            maxOf(
                minBufferSize,
                sampleRate * bufferMillis / 1000 * 2 // short -> bytes
            )
        )
        .build()

    init {
        track.play()
    }

    fun enqueue(pcm: ShortArray) {
        if (pcm.isEmpty()) return
        var offset = 0
        while (offset < pcm.size) {
            val written = track.write(pcm, offset, pcm.size - offset)
            if (written <= 0) break
            offset += written
        }
    }

    fun stop() {
        track.stop()
    }

    fun release() {
        track.release()
    }
}
