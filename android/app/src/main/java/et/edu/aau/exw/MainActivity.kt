package et.edu.aau.exw

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/** Pick a wide-view photo -> tile -> INT8 model -> heat map + risk score. Everything happens on the phone. */
class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var classifier: TileClassifier

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) analyze(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        classifier = TileClassifier(this)
        findViewById<Button>(R.id.pickButton).setOnClickListener { picker.launch("image/*") }
    }

    private fun analyze(uri: Uri) {
        val text = findViewById<TextView>(R.id.resultText)
        text.setText(R.string.analyzing)
        worker.execute {
            val bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it) }
                .copy(Bitmap.Config.ARGB_8888, false)
            val t0 = SystemClock.elapsedRealtime()
            val grid = Tiler.grid(bmp.height, bmp.width, classifier.tileSize, 48)
            val p = FloatArray(grid.rows * grid.cols)
            var k = 0
            for (y in grid.ys) for (x in grid.xs) p[k++] = classifier.classify(bmp, x, y)[0]
            val res = RiskScorer.score(p, grid.rows, grid.cols)
            val overlay = HeatmapOverlay.render(bmp, grid, p)
            val ms = SystemClock.elapsedRealtime() - t0
            runOnUiThread {
                findViewById<ImageView>(R.id.imageView).setImageBitmap(overlay)
                val verdict = getString(if (res.risk >= RISK_THRESHOLD) R.string.result_flagged else R.string.result_clear)
                text.text = "$verdict\nrisk R=%.2f  clusters=%d  tiles=%d  %d ms".format(res.risk, res.clusters, p.size, ms)
            }
        }
    }

    override fun onDestroy() {
        classifier.close(); worker.shutdown(); super.onDestroy()
    }

    companion object {
        /** Farm threshold tau: copy the value chosen on the VALIDATION split (report.json: farm_int8.threshold). */
        const val RISK_THRESHOLD = 0.05f
    }
}
