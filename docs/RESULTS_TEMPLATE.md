# Results tables (fill from `report.json`; label the data tier in every caption)

## Table 1 -- Tile level (data: Tier ___, split: test / test_shift)
| Model | Macro-F1 | EXW recall | EXW precision | EXW AUC |
|---|---|---|---|---|
| FP32 | | | | |
| INT8 (reference kernels) | | | | |
INT8 macro-F1 drop: ___ pp (limit 2 pp; QAT if exceeded)

## Table 2 -- Farm level (threshold tau = ___ chosen on validation)
| Split | Images | Sensitivity (95 % CI) | Specificity (95 % CI) | AUC |
|---|---|---|---|---|
| test | | | | |
| test_shift | | | | |
CIs: bootstrap over farms (`exw.eval.stats.grouped_bootstrap`).

## Table 3 -- Golden-model agreement
| Check | Tiles | Mismatches |
|---|---|---|
| golden vs TFLite reference kernels | | 0 expected |
| TFLite optimized vs reference (max LSB) | | |
| RTL vs golden (per layer) | | |

## Table 4 -- Speed and energy (same tiles, same protocol)
| Device / backend | ms per tile (median, min-max) | ms per image | Energy per image | Notes |
|---|---|---|---|---|
| Lab CPU FP32 | | | n/a | |
| Lab CPU INT8 | | | n/a | |
| DE1-SoC ARM INT8 | | | | |
| DE1-SoC FPGA INT8 | | | | |
| Galaxy A13 TFLite INT8 | | | | |

## Table 5 -- FPGA resources (Quartus report)
ALMs ___ / 32 070, DSP ___ / 87, memory ___ Kb, Fmax ___ MHz.
