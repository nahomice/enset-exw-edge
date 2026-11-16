package et.edu.aau.exw

/** Tile grid identical to `exw.data.tiling.tile_grid` (the last row/column is clamped inside the image). */
object Tiler {
    data class Grid(val xs: List<Int>, val ys: List<Int>) {
        val rows get() = ys.size
        val cols get() = xs.size
    }

    fun grid(height: Int, width: Int, size: Int = 96, stride: Int = 48): Grid {
        val ys = (0..maxOf(height - size, 0) step stride).toMutableList()
        val xs = (0..maxOf(width - size, 0) step stride).toMutableList()
        if (ys.last() + size < height) ys.add(height - size)
        if (xs.last() + size < width) xs.add(width - size)
        return Grid(xs, ys)
    }
}
