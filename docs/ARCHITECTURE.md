# Architecture

## Pipeline
`prepare_data -> train -> quantize -> evaluate -> export_fpga` (see `Makefile`). Each stage reads the config and writes
under `results_dir`.

## Tile CNN (`models/tilecnn.py`)
conv3x3/2 (16) -> 4 x [depthwise 3x3 + pointwise 1x1] (32, 64, 128, 256; strides 1,2,2,2) -> 6x6 average pool ->
dense(3). BatchNorm folded into convs at conversion; ReLU6 fused. ~49.8 k parameters (budget: < 200 k).
Input: `pixel - 128` as int8 (scale 1/128, zero point 0). Output: 3 logits (EXW-suspect, other, healthy/background).

Only CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED compute ops appear in the INT8 graph.
SHAPE / STRIDED_SLICE / PACK around the flatten are constant shape plumbing (ignored by the golden model).

## INT8 contract
Per-channel int8 weights, per-tensor int8 activations, int32 accumulators, Q31 multiplier requantization
(conv/depthwise: `SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`, i.e. double rounding; the final
per-channel FC layer: single rounding, found empirically: 7 of 5862 outputs differ otherwise), ReLU6 as a clamp in the quantized domain, SAME padding.
`GoldenModel` matches TFLite **reference** kernels bit for bit. TFLite's optimized (XNNPACK) kernels may differ by up to
1-2 LSB on some outputs: the FPGA follows the reference, the phone may show +-2 LSB; this is measured in `evaluate.py`.

## Farm decision
tile probabilities -> grid -> flagged tiles (P >= 0.5) -> risk R = flagged/total, connected-component cluster count ->
farm flagged if R >= tau. tau is chosen on validation (highest tau with sensitivity >= 90 %).

## When INT8 loses accuracy
If macro-F1 drops more than `quant.max_drop_pp`: (1) more/better calibration tiles, (2) per-channel check,
(3) quantization-aware training (add `tensorflow-model-optimization` QAT in `train.py`; not implemented yet).

## FPGA mapping (see `fpga/README.md`)
`export_fpga.py` writes weights, biases, multipliers, shifts (`.mem`) and per-layer test vectors from the golden
model. RTL passes when every layer output equals the vectors.
