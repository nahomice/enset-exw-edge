package et.edu.aau.exw

import android.graphics.Bitmap
import android.graphics.Color

/** Red overlay of the averaged EXW probability per pixel, like `exw.infer.heatmap.overlay` (alpha 0.45). */
object HeatmapOverlay {
    fun render(src: Bitmap, grid: Tiler.Grid, pExw: FloatArray, size: Int = 96, alpha: Float = 0.45f): Bitmap {
        val w = src.width; val h = src.height
        val heat = FloatArray(w * h); val cnt = FloatArray(w * h)
        var k = 0
        for (y in grid.ys) for (x in grid.xs) {
            val p = pExw[k++]
            for (yy in y until minOf(y + size, h)) for (xx in x until minOf(x + size, w)) {
                heat[yy * w + xx] += p; cnt[yy * w + xx] += 1f
            }
        }
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(w * h)
        out.getPixels(px, 0, w, 0, 0, w, h)
        for (i in px.indices) {
            if (cnt[i] == 0f) continue
            val a = alpha * heat[i] / cnt[i]
            val r = (Color.red(px[i]) * (1 - a) + 255 * a).toInt().coerceIn(0, 255)
            val g = (Color.green(px[i]) * (1 - a)).toInt().coerceIn(0, 255)
            val b = (Color.blue(px[i]) * (1 - a)).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }
}
