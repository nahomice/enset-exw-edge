package et.edu.aau.exw

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * Runs the INT8 TFLite tile model. Input per tile: int8 = round(((pixel - 128) / 128) / inScale) + inZeroPoint
 * (with the exported model inScale = 1/128 and zero point 0, so simply pixel - 128). Output: 3 int8 logits
 * (EXW-suspect, other condition, healthy/background) -> softmax probabilities.
 */
class TileClassifier(context: Context, modelAsset: String = "model_int8.tflite", threads: Int = 2) {
    private val interpreter: Interpreter
    private val inScale: Float
    private val inZero: Int
    private val outScale: Float
    private val outZero: Int
    val tileSize = 96

    init {
        val fd = context.assets.openFd(modelAsset)
        val map = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(map, Interpreter.Options().setNumThreads(threads))
        val inQ = interpreter.getInputTensor(0).quantizationParams()
        val outQ = interpreter.getOutputTensor(0).quantizationParams()
        inScale = inQ.scale; inZero = inQ.zeroPoint
        outScale = outQ.scale; outZero = outQ.zeroPoint
    }

    private val input = ByteBuffer.allocateDirect(tileSize * tileSize * 3).order(ByteOrder.nativeOrder())
    private val output = Array(1) { ByteArray(3) }
    private val pixels = IntArray(tileSize * tileSize)

    /** Probabilities [EXW, other, healthy] for the tile whose top-left corner is (x, y). */
    fun classify(bitmap: Bitmap, x: Int, y: Int): FloatArray {
        bitmap.getPixels(pixels, 0, tileSize, x, y, tileSize, tileSize)
        input.rewind()
        for (p in pixels) {
            for (shift in intArrayOf(16, 8, 0)) {                      // R, G, B
                val v = (p shr shift) and 0xFF
                val q = (((v - 128) / 128f) / inScale).roundToInt() + inZero
                input.put(q.coerceIn(-128, 127).toByte())
            }
        }
        input.rewind()
        interpreter.run(input, output)
        val logits = FloatArray(3) { (output[0][it].toInt() - outZero) * outScale }
        val m = logits.max()
        val e = FloatArray(3) { exp(logits[it] - m) }
        val s = e.sum()
        return FloatArray(3) { e[it] / s }
    }

    fun close() = interpreter.close()
}
