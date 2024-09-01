package com.samsia.peopledetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.samsia.peopledetection.ai.DetectionProcessor
import com.samsia.peopledetection.ai.model.PeopleDetectionModel
import com.samsia.peopledetection.image.ImageUtil.yuv420ToBitmap
import com.samsia.peopledetection.ui.theme.PeopleDetectionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@ExperimentalGetImage
class MainActivity : ComponentActivity() {
    private var peopleDetectionModel: PeopleDetectionModel? = null
    private var isModelInitialized = mutableStateOf(false)
    private var detectionProcessor: DetectionProcessor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        lifecycleScope.launch(Dispatchers.Default) {
            peopleDetectionModel = PeopleDetectionModel(applicationContext, modelInputSize, "model.tflite")
            isModelInitialized.value = true
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        setContent {
            PeopleDetectionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (isModelInitialized.value) {
                        peopleDetectionModel?.let { model ->
                            CameraPreview(model)
                        }
                    } else {
                        // Show loading indicator or placeholder
                        Text("Initializing model...")
                    }
                }
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { permission ->
        ContextCompat.checkSelfPermission(baseContext, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        // Camera initialization code here
    }

    @ExperimentalGetImage
    @Composable
    fun CameraPreview(peopleDetectionModel: PeopleDetectionModel) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        var bitmap by remember { mutableStateOf<Bitmap?>(null) }
        var detectionResults by remember { mutableStateOf<List<FloatArray>>(emptyList()) }

        val preview = Preview.Builder().build()
        val previewView = remember { PreviewView(context) }
        val imageAnalyzer = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        DisposableEffect(lifecycleOwner) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            val cameraProvider = cameraProviderFuture.get()

            imageAnalyzer.setAnalyzer(ContextCompat.getMainExecutor(context)) { imageProxy ->
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                val image = imageProxy.image
                val width = imageProxy.width
                val height = imageProxy.height
                if (image != null) {
                    val tempBitmap = yuv420ToBitmap(image)
                    bitmap = tempBitmap

                    if (tempBitmap != null) {
                        detectionResults = emptyList()
                        if (detectionProcessor == null || (detectionProcessor?.frameWidth != width || detectionProcessor?.frameHeight != height)) {
                            detectionProcessor = DetectionProcessor(width, height)
                        }

                        val rotatedBitmap = rotateBitmap(tempBitmap, rotationDegrees)
                        bitmap = rotatedBitmap

                        (lifecycleOwner as? ComponentActivity)?.lifecycleScope?.launch(Dispatchers.Default) {
                            val outputArray = peopleDetectionModel.detectObjects(rotatedBitmap)
                            detectionProcessor?.processDetectionResults(outputArray)?.let {
                                detectionResults = it
                            }

                        }
                    } else {
                        Log.e("PeopleDetection", "Bitmap is null")
                    }
                }
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )

                preview.setSurfaceProvider(previewView.surfaceProvider)
            } catch (exc: Exception) {
                Log.e("CameraPreview", "Use case binding failed", exc)
            }

            onDispose {
                cameraProvider.unbindAll()
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize())

            bitmap?.let { btm ->
                Image(
                    bitmap = btm.asImageBitmap(),
                    contentDescription = "Camera Preview",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawBoundingBoxes(detectionResults, btm.width, btm.height)
                }
            }
        }
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun DrawScope.drawBoundingBoxes(results: List<FloatArray>, imageWidth: Int, imageHeight: Int) {
        Log.d("DrawBoundingBoxes", "Number of boxes to draw: ${results.size}")

        val scaleX = size.width / imageWidth
        val scaleY = size.height / imageHeight

        results.forEachIndexed { index, result ->
            val left = result[0] * imageWidth * scaleX
            val top = result[1] * imageHeight * scaleY
            val right = result[2] * imageWidth * scaleX
            val bottom = result[3] * imageHeight * scaleY
            val confidence = result[4]

            Log.d("DrawBoundingBoxes", "Drawing box $index: left=$left, top=$top, right=$right, bottom=$bottom, confidence=$confidence")

            drawRect(
                color = Color.Red,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 2f)
            )

            drawContext.canvas.nativeCanvas.apply {
                val text = "Person %.2f".format(confidence)
                drawText(
                    text,
                    left,
                    top - 10f,
                    android.graphics.Paint().apply {
                        color = android.graphics.Color.RED
                        textSize = 30f
                        isFakeBoldText = true
                    }
                )
            }
        }
    }

//    fun processDetectionResults(outputArray: Array<Array<Array<Array<FloatArray>>>>): List<FloatArray> {
//        val boundingBoxes = mutableListOf<FloatArray>()
//        val confidenceThreshold = 0.7f
//        val anchors = arrayOf(
//            arrayOf(12f, 18f), arrayOf(37f, 49f), arrayOf(52f, 132f),
//            arrayOf(115f, 73f), arrayOf(119f, 199f), arrayOf(242f, 238f)
//        )
//        val strides = arrayOf(16f, 32f)
//
//        for (anchorIndex in outputArray[0].indices) {
//            val stride = strides[anchorIndex / 3]
//            for (y in outputArray[0][anchorIndex].indices) {
//                for (x in outputArray[0][anchorIndex][y].indices) {
//                    val data = outputArray[0][anchorIndex][y][x]
//                    val confidence = sigmoid(data[4])
//                    if (confidence > confidenceThreshold) {
//                        val cx = (sigmoid(data[0]) + x) * stride
//                        val cy = (sigmoid(data[1]) + y) * stride
//                        val w = exp(data[2]) * anchors[anchorIndex][0]
//                        val h = exp(data[3]) * anchors[anchorIndex][1]
//
//                        val left = (cx - w / 2) / modelInputSize
//                        val top = (cy - h / 2) / modelInputSize
//                        val right = (cx + w / 2) / modelInputSize
//                        val bottom = (cy + h / 2) / modelInputSize
//
//                        val (limitedLeft, limitedTop, limitedRight, limitedBottom) = limitBoxSize(left, top, right, bottom)
//                        boundingBoxes.add(floatArrayOf(limitedLeft, limitedTop, limitedRight, limitedBottom, confidence, 0f))
//                    }
//                }
//            }
//        }
//
//        Log.d("ObjectDetection", "Number of detections before NMS: ${boundingBoxes.size}")
//        val nmsResults = applyNMS(boundingBoxes)
//        Log.d("ObjectDetection", "Number of detections after NMS: ${nmsResults.size}")
//        val smoothedResults = detectionSmoother.smooth(nmsResults)
//        Log.d("ObjectDetection", "Number of detections after smoothing: ${smoothedResults.size}")
//
//        return smoothedResults
//    }
//
//    fun limitBoxSize(left: Float, top: Float, right: Float, bottom: Float): FloatArray {
//        val maxWidth = 0.8f // 이미지 너비의 80%
//        val maxHeight = 0.8f // 이미지 높이의 80%
//
//        val width = right - left
//        val height = bottom - top
//
//        val newWidth = minOf(width, maxWidth)
//        val newHeight = minOf(height, maxHeight)
//
//        val centerX = (left + right) / 2
//        val centerY = (top + bottom) / 2
//
//        val newLeft = maxOf(0f, centerX - newWidth / 2)
//        val newTop = maxOf(0f, centerY - newHeight / 2)
//        val newRight = minOf(1f, newLeft + newWidth)
//        val newBottom = minOf(1f, newTop + newHeight)
//
//        return floatArrayOf(newLeft, newTop, newRight, newBottom)
//    }



    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        val modelInputSize = 320
    }
}