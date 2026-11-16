# android/ (Samsung Galaxy A13, Android 12)

Primary field device. The app picks a wide-view photo, tiles it (96 x 96, stride 48), runs the INT8 `.tflite` with
TensorFlow Lite, and shows the heat map, risk score R and "flag for expert inspection". It has **no internet
permission** by design.

```
android/
  settings.gradle.kts, build.gradle.kts, gradle.properties
  app/
    build.gradle.kts                      dependencies (TensorFlow Lite 2.14, AppCompat), noCompress for .tflite
    src/main/AndroidManifest.xml
    src/main/java/et/edu/aau/exw/
      Tiler.kt             same tile grid as exw.data.tiling
      TileClassifier.kt    INT8 inference, input/output (de)quantization, softmax
      RiskScorer.kt        risk R and 4-connected clusters, same as exw.infer.heatmap
      HeatmapOverlay.kt    red probability overlay
      MainActivity.kt      photo picker + background analysis + result text
    src/main/res/          layout, strings, theme
    src/main/assets/       put model_int8.tflite here (git-ignored)
    src/test/java/...      JUnit tests (grid parity with Python, clusters)
  benchmark/               protocol + log for latency and battery measurements
  parity/                  Python/Android agreement check on real photos
```

## First steps
1. Open `android/` in Android Studio (it creates the Gradle wrapper and offers version upgrades; the versions in the
   Gradle files are a starting point I have not built).
2. Copy the model: `results/<run>/model_seed0_int8.tflite` -> `app/src/main/assets/model_int8.tflite`.
3. Run the JUnit tests (`TilerTest`), then run on the phone.
4. Set `MainActivity.RISK_THRESHOLD` to the value chosen on the validation split (`report.json`, `farm_int8.threshold`).
5. Benchmark with `benchmark/README.md`; check Python parity with `parity/README.md`.

Status: **written but never compiled or run.** Expect small build fixes on first import.
