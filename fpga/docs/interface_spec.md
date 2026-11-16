# Accelerator interface (proposal -- finalise together with the RTL)

## Data format
- Tile: 96 x 96 x 3, int8 = pixel - 128, row-major, channel-last (HWC). 27 648 bytes.
- Output: 3 int8 logits (class order: EXW-suspect, other condition, healthy/background). Output scale and zero point are in
  `model_spec.json` (`output_scale`, `output_zero_point`). Softmax and the farm decision stay on the ARM/phone.
- Weights/biases/requantization constants: `L<k>_*.mem` files from `scripts/export_fpga.py`, one value per line in hex.

## Arithmetic contract (must equal `exw.golden`)
| Layer | Accumulate | Requantize | Clamp |
|---|---|---|---|
| conv / depthwise | `sum (x - zp_in) * w + bias`, int32 | multiply-high with doubling + rounding shift (double rounding) | ReLU6 range in int8 domain |
| avg pool 6x6 | sum of 36 int8, int32 | `(acc + 18) / 36`, half away from zero | int8 |
| FC | `sum (x - zp_in) * w + bias`, int32 | single rounding | none (-128..127) |

Padding: SAME; padded positions contribute zero after zero-point subtraction.

## Registers
See `fpga/hps/driver/fpga_regs.h` (base 0xFF200000, lightweight HPS-to-FPGA bridge).

## Acceptance tests (in order)
1. `tb_requantize` and `tb_mac_unit`: PASS.
2. Each layer output equals `vectors/tile_XX_L<k>.mem` for all exported tiles (`scripts/compare_rtl_outputs.py`).
3. 1000 test tiles: logits equal the golden logits exactly.
4. On the board: ARM-read logits equal the golden logits for the same tiles.
