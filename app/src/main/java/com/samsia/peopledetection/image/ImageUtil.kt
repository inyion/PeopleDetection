package com.samsia.peopledetection.image

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image
import android.util.Log
import java.io.ByteArrayOutputStream

object ImageUtil {

    fun yuv420ToBitmap(image: Image): Bitmap? {

        try {
            val planes = image.planes
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            // U and V are swapped
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e("ImageUtil", "Error processing image: ${e.message}")
            return null
        }

    }

    fun adjustBoundingBoxForRotation(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): FloatArray {
        return when (rotationDegrees) {
            90 -> floatArrayOf(
                top,
                imageWidth - right,
                bottom,
                imageWidth - left
            )
            180 -> floatArrayOf(
                imageWidth - right,
                imageHeight - bottom,
                imageWidth - left,
                imageHeight - top
            )
            270 -> floatArrayOf(
                imageHeight - bottom,
                left,
                imageHeight - top,
                right
            )
            else -> floatArrayOf(left, top, right, bottom) // 회전 없음
        }
    }
}