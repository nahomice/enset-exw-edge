package et.edu.aau.exw

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TilerTest {
    @Test fun gridOf960x720MatchesPython() {          // exw.data.tiling.tile_grid(720, 960, 96, 48): 14 rows x 19 cols
        val g = Tiler.grid(720, 960)
        assertEquals(14, g.rows); assertEquals(19, g.cols)
        assertEquals(624, g.ys.last()); assertEquals(864, g.xs.last())
    }

    @Test fun lastTileIsClampedInsideTheImage() {     // 100x100: starts 0 and 4 (4 + 96 = 100)
        val g = Tiler.grid(100, 100)
        assertEquals(listOf(0, 4), g.ys); assertEquals(listOf(0, 4), g.xs)
    }

    @Test fun clustersAreFourConnected() {
        // 3x3 grid, flagged: (0,0),(0,1) form one cluster; (2,2) is a second; diagonal (1,... ) not used
        val p = floatArrayOf(1f, 1f, 0f,  0f, 0f, 0f,  0f, 0f, 1f)
        val r = RiskScorer.score(p, 3, 3)
        assertEquals(2, r.clusters)
        assertTrue(Math.abs(r.risk - 3f / 9f) < 1e-6)
    }
}
