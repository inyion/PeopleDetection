package com.samsia.peopledetection.ai.model

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.ByteArrayOutputStream

class PeopleDetectionModelForPython(private val context: Context) {
    private val py: Python
    private val peopleDetectionModule: PyObject
    private lateinit var model: PyObject

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        py = Python.getInstance()
        peopleDetectionModule = py.getModule("PeopleDetection")

        // PeopleDetectionModel 클래스 초기화
        val modelClass = peopleDetectionModule.get("PeopleDetectionModel")
        if (modelClass != null) {
            try {
                // Python 메서드 호출 시 키워드 인자를 직접 전달
                modelClass.callAttr("initialize",
                    PyObject.fromJava("framework"),
                    PyObject.fromJava("tflite"),
                    PyObject.fromJava("file"),
                    PyObject.fromJava("model.tflite"))
                // 인스턴스 생성
                model = modelClass.call()
            } catch (e: Exception) {
                Log.e("PeopleDetectionModel", "Error initializing model: ${e.message}")
                throw RuntimeException("Failed to initialize PeopleDetectionModel", e)
            }
        } else {
            throw RuntimeException("Failed to get PeopleDetectionModel class from Python module")
        }
    }

    fun run(bitmap: Bitmap): List<FloatArray> {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        val byteArray = stream.toByteArray()

        val result = model.callAttr("run", byteArray)
        return result.asList().map { pyObject ->
            pyObject.toJava(FloatArray::class.java)
        }
    }

    fun finalize() {
        model.callAttr("finalize")
    }
}