package com.samsia.peopledetection.ai

import kotlin.math.exp

object Cal {
    fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
        val xMin = maxOf(box1[0], box2[0])
        val yMin = maxOf(box1[1], box2[1])
        val xMax = minOf(box1[2], box2[2])
        val yMax = minOf(box1[3], box2[3])

        val intersectionArea = maxOf(0f, xMax - xMin) * maxOf(0f, yMax - yMin)
        val box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun isCenterInside(box1: FloatArray, box2: FloatArray): Boolean {
        val center1X = (box1[0] + box1[2]) / 2
        val center1Y = (box1[1] + box1[3]) / 2

        return center1X > box2[0] && center1X < box2[2] && center1Y > box2[1] && center1Y < box2[3]
    }

    fun sigmoid(x: Float): Float {
        return (1 / (1 + exp(-x)))
    }
    fun calculateDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    }
}