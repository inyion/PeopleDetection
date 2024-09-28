package com.samsia.peopledetection.ai

import android.util.Log
import com.samsia.peopledetection.MainActivity.Companion.modelInputSize
import com.samsia.peopledetection.ai.Cal.calculateIoU
import com.samsia.peopledetection.ai.Cal.isCenterInside
import com.samsia.peopledetection.ai.Cal.sigmoid
import kotlin.math.exp

class DetectionProcessor(var frameWidth: Int, var frameHeight: Int) {
    private val detectionSmoother = DetectionSmoother() // 3프레임 평균
    val confidenceThreshold = 0.3f
    private val maxDetectionsThreshold = 30 // 비정상적으로 많은 감지로 간주할 임계값

    fun processDetectionResults(outputArray: Array<Array<Array<Array<FloatArray>>>>): List<FloatArray> {
        val boundingBoxes = mutableListOf<FloatArray>()
        val anchors = arrayOf(
            arrayOf(12f, 18f), arrayOf(37f, 49f), arrayOf(52f, 132f),
            arrayOf(115f, 73f), arrayOf(119f, 199f), arrayOf(242f, 238f)
        )
        val strides = arrayOf(16f, 32f)

        for (output in outputArray) {
            for (anchorIndex in output.indices) {
                val stride = strides[anchorIndex / 3]
                for (y in outputArray[0][anchorIndex].indices) {
                    for (x in outputArray[0][anchorIndex][y].indices) {
                        val data = outputArray[0][anchorIndex][y][x]
                        val confidence = sigmoid(data[4])
                        if (confidence > confidenceThreshold) {
                            val cx = (sigmoid(data[0]) + x) * stride
                            val cy = (sigmoid(data[1]) + y) * stride
                            val w = exp(data[2]) * anchors[anchorIndex][0]
                            val h = exp(data[3]) * anchors[anchorIndex][1]

                            val left = (cx - w / 2) / modelInputSize
                            val top = (cy - h / 2) / modelInputSize
                            val right = (cx + w / 2) / modelInputSize
                            val bottom = (cy + h / 2) / modelInputSize

                            boundingBoxes.add(floatArrayOf(left, top, right, bottom, confidence, 0f))
                        }
                    }
                }
            }
        }

        // 비정상적으로 많은 감지 결과 처리
        if (boundingBoxes.size > maxDetectionsThreshold) {
            Log.d("ObjectDetection", "Abnormally high number of detections: ${boundingBoxes.size}. Ignoring results.")
            return emptyList()
        }

        if (boundingBoxes.size > 0) {
            Log.d("ObjectDetection", "Number of detected boxes: ${boundingBoxes.size}")
        }
        if (boundingBoxes.isNotEmpty()) {
            Log.d("ObjectDetection", "Sample bounding box: ${boundingBoxes.first().contentToString()}")
        }

        val nmsResults = applyNMS(boundingBoxes)
        Log.d("ObjectDetection", "Number of boxes after NMS: ${nmsResults.size}")

        val smoothedResults = detectionSmoother.smooth(nmsResults)
        Log.d("ObjectDetection", "Number of boxes after smoothing: ${smoothedResults.size}")

        return smoothedResults
    }

    private fun applyNMS(boxes: List<FloatArray>, iouThreshold: Float = 0.7f): List<FloatArray> {
        val sortedBoxes = boxes.sortedByDescending { it[4] }
        val selectedBoxes = mutableListOf<FloatArray>()

        for (box in sortedBoxes) {
            if (selectedBoxes.none {
                    val iou = calculateIoU(box, it)
                    val centerOverlap = isCenterInside(box, it)
                    iou > iouThreshold || centerOverlap
                }) {
                selectedBoxes.add(box)
            }
        }

        return selectedBoxes
    }


}