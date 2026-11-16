# Phone benchmark protocol (Samsung Galaxy A13, Android 12)

Goal: latency and energy per tile and per wide-view image, measured identically every time.

Setup (write the actual values into the log row)
- Airplane mode on, screen brightness fixed at 50 %, battery 50-100 %, phone not charging, no other apps open.
- Phone cooled to room temperature before the first run; note the room temperature.
- Same model file (record its SHA-256), same 64 tiles and 3 wide-view images for every run.

Procedure
1. 5 warm-up inferences, then 30 timed runs; time with `SystemClock.elapsedRealtimeNanos()` around the tile loop only
   (not image decoding). Report median, min, max in ms per tile and ms per image.
2. Repeat with 1, 2 and 4 threads (`TileClassifier(threads=...)`), and once with the NNAPI delegate if used.
3. Energy: read `BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER` (uAh) before and after 300 consecutive images and report
   mAh per image (the counter is coarse; use enough images that the drop is at least several counter steps). Repeat 3 times.
4. Log every run in `benchmark_log.csv`; never overwrite rows.

Remember: TFLite's optimized kernels can differ from the golden reference by up to 2 LSB on some outputs. That is
expected; check with `parity/` that decisions (flagged or not) agree.
