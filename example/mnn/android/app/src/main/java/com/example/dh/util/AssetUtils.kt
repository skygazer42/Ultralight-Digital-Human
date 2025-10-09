package com.example.dh.util

import android.content.Context
import java.io.File

object AssetUtils {
    fun copyAssetDir(context: Context, assetDir: String, destDir: File) {
        if (!destDir.exists()) destDir.mkdirs()
        val files = context.assets.list(assetDir) ?: return
        for (file in files) {
            val srcPath = if (assetDir.isEmpty()) file else "$assetDir/$file"
            val dst = File(destDir, file)
            val children = context.assets.list(srcPath)
            if (children.isNullOrEmpty()) {
                context.assets.open(srcPath).use { input ->
                    dst.outputStream().use { output -> input.copyTo(output) }
                }
            } else {
                copyAssetDir(context, srcPath, dst)
            }
        }
    }
}
