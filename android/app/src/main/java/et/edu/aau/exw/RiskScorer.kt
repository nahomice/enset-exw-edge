package et.edu.aau.exw

/** Farm risk score and cluster count, identical to `exw.infer.heatmap.score_image` (4-connected clusters). */
object RiskScorer {
    data class Result(val risk: Float, val clusters: Int, val flagged: BooleanArray, val rows: Int, val cols: Int)

    /** `pExw` is the row-major grid of P(EXW-suspect) per tile. */
    fun score(pExw: FloatArray, rows: Int, cols: Int, tileThr: Float = 0.5f): Result {
        val flagged = BooleanArray(pExw.size) { pExw[it] >= tileThr }
        val risk = flagged.count { it }.toFloat() / flagged.size
        val seen = BooleanArray(flagged.size)
        var clusters = 0
        val stack = ArrayDeque<Int>()
        for (start in flagged.indices) {
            if (!flagged[start] || seen[start]) continue
            clusters++
            seen[start] = true; stack.addLast(start)
            while (stack.isNotEmpty()) {
                val c = stack.removeLast()
                val r = c / cols; val col = c % cols
                for ((dr, dc) in listOf(-1 to 0, 1 to 0, 0 to -1, 0 to 1)) {
                    val nr = r + dr; val nc = col + dc
                    if (nr !in 0 until rows || nc !in 0 until cols) continue
                    val n = nr * cols + nc
                    if (flagged[n] && !seen[n]) { seen[n] = true; stack.addLast(n) }
                }
            }
        }
        return Result(risk, clusters, flagged, rows, cols)
    }
}
