package com.samsia.peopledetection.ai

import com.samsia.peopledetection.ai.Cal.calculateDistance
import com.samsia.peopledetection.ai.Cal.calculateIoU
import kotlin.math.abs

class DetectionSmoother(private val frameCount: Int = 3, private val smoothFactor: Float = 0.5f) {
    private val detectionHistory = mutableListOf<List<FloatArray>>()
    private var lastSmoothDetections = listOf<FloatArray>()

    @Synchronized
    fun smooth(detections: List<FloatArray>): List<FloatArray> {
        detectionHistory.add(detections)
        if (detectionHistory.size > frameCount) {
            detectionHistory.removeAt(0)
        }

        val averagedDetections = averageDetections()
        lastSmoothDetections = smoothTransition(averagedDetections)
        return lastSmoothDetections
    }

    private fun averageDetections(): List<FloatArray> {
        if (detectionHistory.isEmpty()) return emptyList()

        val allDetections = detectionHistory.flatten()
        val groupedDetections = allDetections.groupBy { detection ->
            // 바운딩 박스의 중심점을 기준으로 그룹화
            val centerX = (detection[0] + detection[2]) / 2
            val centerY = (detection[1] + detection[3]) / 2
            Pair((centerX * 100).toInt(), (centerY * 100).toInt())
        }

        return groupedDetections.mapNotNull { (_, detections) ->
            if (detections.size >= frameCount / 2) {
                val avgDetection = FloatArray(6) { i ->
                    detections.map { it[i] }.average().toFloat()
                }
                avgDetection
            } else null
        }
    }

    private fun smoothTransition(currentDetections: List<FloatArray>): List<FloatArray> {
        if (lastSmoothDetections.isEmpty()) return currentDetections

        val smoothedDetections = mutableListOf<FloatArray>()

        for (currentBox in currentDetections) {
            val matchingPrevBox = findMatchingBox(currentBox, lastSmoothDetections)
            if (matchingPrevBox != null) {
                val smoothedBox = FloatArray(6)
                for (i in 0..5) {
                    smoothedBox[i] = matchingPrevBox[i] + (currentBox[i] - matchingPrevBox[i]) * smoothFactor
                }
                smoothedDetections.add(smoothedBox)
            } else {
                smoothedDetections.add(currentBox)
            }
        }

        // 사라진 박스들을 부드럽게 제거
        for (prevBox in lastSmoothDetections) {
            if (findMatchingBox(prevBox, currentDetections) == null) {
                val fadingBox = FloatArray(6)
                System.arraycopy(prevBox, 0, fadingBox, 0, 6)
                fadingBox[4] *= (1 - smoothFactor) // confidence를 서서히 줄임
                if (fadingBox[4] > 0.1f) { // confidence가 일정 수준 이상일 때만 유지
                    smoothedDetections.add(fadingBox)
                }
            }
        }

        return smoothedDetections
    }

    private fun findMatchingBox(box: FloatArray, boxList: List<FloatArray>): FloatArray? {
        val centerX = (box[0] + box[2]) / 2
        val centerY = (box[1] + box[3]) / 2

        return boxList.minByOrNull { otherBox ->
            val otherCenterX = (otherBox[0] + otherBox[2]) / 2
            val otherCenterY = (otherBox[1] + otherBox[3]) / 2
            val distance = calculateDistance(centerX, centerY, otherCenterX, otherCenterY)
            val sizeDiff = abs((box[2] - box[0]) * (box[3] - box[1]) - (otherBox[2] - otherBox[0]) * (otherBox[3] - otherBox[1]))
            distance + sizeDiff
        }?.takeIf { calculateIoU(box, it) > 0.3f }
    }


}