package com.example.dh

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.dh.audio.AudioPlayer
import com.example.dh.audio.AudioStreamer
import com.example.dh.util.AssetUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var preview: ImageView
    private var nativeEngine: DigitalHumanNative? = null
    private var audioStreamer: AudioStreamer? = null
    private var audioPlayer: AudioPlayer? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            lifecycleScope.launch { initializePipeline() }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preview = ImageView(this).apply { adjustViewBounds = true }
        setContentView(preview)

        if (hasAudioPermission()) {
            lifecycleScope.launch { initializePipeline() }
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    override fun onStart() {
        super.onStart()
        if (hasAudioPermission()) {
            audioStreamer?.start()
        }
    }

    override fun onStop() {
        audioStreamer?.stop()
        super.onStop()
    }

    override fun onDestroy() {
        audioStreamer?.release()
        audioPlayer?.release()
        nativeEngine?.release()
        nativeEngine = null
        super.onDestroy()
    }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private suspend fun initializePipeline() {
        prepareEngine()
        startStreaming()
    }

    private suspend fun prepareEngine() = withContext(Dispatchers.IO) {
        val runtimeDir = File(filesDir, "runtime").apply { mkdirs() }
        val datasetDir = File(runtimeDir, "dataset").apply { mkdirs() }
        val modelDir = File(runtimeDir, "models").apply { mkdirs() }
        AssetUtils.copyAssetDir(this@MainActivity, "dataset", datasetDir)
        AssetUtils.copyAssetDir(this@MainActivity, "models", modelDir)

        val unetFile = File(modelDir, "unet_ave.mnn")
        val audioEncoder = File(modelDir, "audio_encoder.mnn")

        nativeEngine = DigitalHumanNative(
            context = this@MainActivity,
            datasetDir = datasetDir,
            modelPath = unetFile,
            audioFeat = null,
            audioEncoder = audioEncoder,
            asr = "ave",
            variant = "328",
            fps = 25.0,
            useSmooth = true,
            useSoftMask = true,
            useSeamless = true
        )
    }

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
}
