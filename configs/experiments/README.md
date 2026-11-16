# Planned experiments

Each row is one run family. Run it with the listed overrides, then log it in `docs/EXPERIMENT_LOG.md`.

| ID | Question | Overrides on top of the base config | Metric |
|---|---|---|---|
| X1 | How does tile size change farm-level detection? | `tiles.size` in {64, 96, 128}, `tiles.stride` = size/2 | farm sensitivity/specificity, tile macro-F1 |
| X2 | How small can the model be? | `model.width` in {0.5, 0.75, 1.0} | params, INT8 macro-F1, latency |
| X3 | Is INT8 enough or is QAT needed? | none (read `int8_macro_f1_drop_pp` in `report.json`) | drop in percentage points |
| X4 | Does the model survive appearance shift? | none (read `test_shift`) | farm metrics on `test_shift` vs `test` |
| X5 | How sensitive is the box-to-tile rule? | `tiles.box_frac` in {0.3, 0.5, 0.7}, `tiles.ignore_frac` in {0.05, 0.1, 0.2} | tile and farm metrics |
| X6 | How many calibration tiles are needed? | `quant.n_calib` in {50, 150, 300, 600} | INT8 drop |
| X7 | Effect of background balancing | `tiles.bg_keep_ratio` in {1, 3, 10} | false alarms on healthy farms |
| X8 | Operating point of the farm threshold | `farm.tile_prob_threshold` in {0.3, 0.5, 0.7} | ROC of risk score |

`scripts/run_ablation.sh` runs X1 and X2 on synthetic data. Run X3-X8 by hand with `--set` once real data exists.
