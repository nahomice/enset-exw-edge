# Glossary

- **EXW**: Enset Xanthomonas Wilt, caused by *Xanthomonas vasicola* pv. *musacearum*.
- **Wide-view image**: photo showing many plants (field-edge shot; drone optional), not a leaf close-up.
- **Tile**: 96 x 96 pixel crop (stride 48, so tiles overlap by half).
- **Tile label**: EXW-suspect, other condition, or healthy/background, derived from coarse boxes by a fixed rule.
- **IGNORE**: tile that only partly overlaps a box; excluded from training.
- **Risk score R**: fraction of tiles flagged EXW-suspect in one image. **tau**: threshold on R, chosen on validation.
- **Heat map**: per-tile P(EXW) drawn over the photo; **clusters**: connected groups of flagged tiles.
- **Tier 0-3**: data quality tiers (synthetic; real healthy; real EXW/other from partners; with boxes).
- **Grouped split**: all images of a site/farm/date go to the same split, done before tiling.
- **test_shift**: held-out appearance style/site to measure domain shift.
- **PTQ / QAT**: post-training quantization / quantization-aware training.
- **INT8 full-integer**: int8 weights and activations, int32 accumulators, integer-only requantization.
- **Golden model**: the NumPy INT8 implementation that equals TFLite reference kernels bit for bit; the FPGA's specification.
- **LSB**: one integer step of the INT8 output.
- **HPS**: hard processor system (ARM Cortex-A9) of the Cyclone V SoC. **M10K / ALM / DSP**: FPGA memory block / logic unit / multiplier block.
