package com.samsia.peopledetection.ai.model

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class PeopleDetectionModelForPython(private val context: Context) {
    private val py: Python
    private val peopleDetectionModule: PyObject
//    private val modelClass: PyObject?
//    private var model: PyObject

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        py = Python.getInstance()
        peopleDetectionModule = py.getModule("PeopleDetection")

        // PeopleDetectionModel 클래스 초기화
//        modelClass = peopleDetectionModule.get("PeopleDetectionModel")
//        if (modelClass != null) {
//            try {
//                modelClass.callAttr("initialize")
//            } catch (e: Exception) {
//                Log.e("PeopleDetectionModel", "Error initializing model: ${e.message}")
//                throw RuntimeException("Failed to initialize PeopleDetectionModel", e)
//            }
//            model = modelClass.call()
//        } else {
//            throw RuntimeException("Failed to get PeopleDetectionModel class from Python module")
//        }
    }

    fun processImage(bitmap: Bitmap): List<FloatArray>? {
        val modelClass = peopleDetectionModule.get("PeopleDetectionModel")
        try {
            // 모델이 이미 초기화되지 않은 경우에만 초기화
            if (modelClass?.callAttr("is_model_initialized")?.toBoolean() == false) {
                modelClass.callAttr("initialize")
            }

            // 모델 인스턴스 생성
            val model = modelClass?.call()

            // 이미지 처리
            val byteArray = bitmap.toByteArray() // 이미지 데이터를 byteArray로 변환
            val dataMap = mapOf(
                "image" to byteArray,
                "height" to bitmap.height,
                "width" to bitmap.width,
                "channels" to if (bitmap.config == Bitmap.Config.ARGB_8888) 4 else 3
            )
            val result = model?.callAttr("run", dataMap)

            // 결과 변환
            val processedResults = result?.asList()?.map { pyObject ->
                pyObject.toJava(FloatArray::class.java)
            }

            // 리소스 해제는 필요할 때 호출합니다
            return processedResults

        } catch (e: PyException) {
            Log.e("PeopleDetection", "Error during model processing: ${e.message}")
            return null
        }
    }

//    fun run(bitmap: Bitmap): List<FloatArray> {
//        val byteArray = bitmap.toByteArray()
//        val dataMap = mapOf(
//            "image" to byteArray,
//            "height" to bitmap.height,
//            "width" to bitmap.width,
//            "channels" to if (bitmap.config == Bitmap.Config.ARGB_8888) 4 else 3
//        )
//
//        val result = model.callAttr("run", byteArray)
//        return result.asList().map { pyObject ->
//            pyObject.toJava(FloatArray::class.java)
//        }
//    }

    fun finalize() {
//        modelClass?.callAttr("finalize")
    }

    fun Bitmap.toByteArray(): ByteArray {
        val size = rowBytes * height
        val byteArray = ByteArray(size)
        ByteBuffer.wrap(byteArray).apply {
            rewind()
            copyPixelsToBuffer(this)
        }
        return byteArray
    }
}