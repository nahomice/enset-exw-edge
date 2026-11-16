# ARM Cortex-A9 INT8 baseline (on the DE1-SoC HPS)

Purpose: the CPU number the FPGA is compared against, on the **same board, same tiles, same model file**.

Options (pick one, document which):
1. TensorFlow Lite C++ (or the Python `tflite_runtime` wheel if one exists for ARMv7 on the board image) running
   `model_seed0_int8.tflite`. Report this as "ARM INT8 (TFLite)".
2. A plain C implementation of the golden model (`exw.golden`) compiled with `-O3 -mfpu=neon`. Report as
   "ARM INT8 (C reference)". Slower, but it is exactly the arithmetic the FPGA implements.

Measurement protocol (same for every device): 5 warm-up runs, 30 timed runs of 64 tiles, report the median and
min/max in ms per tile; log in `fpga/docs/benchmark_log.csv`. Energy: inline DC meter on the 12 V input, idle power
subtracted, joules per tile = (P_load - P_idle) x time per tile.
