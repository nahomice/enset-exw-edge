# results/

Git-ignored except this file. Folder naming:

| Folder | Created by |
|---|---|
| `smoke/` | `make smoke` with `configs/default.yaml` (synthetic) |
| `smoke_tiny/` | `configs/smoke_test.yaml` |
| `real/` | `configs/real_data.yaml` |
| `ablation/<tag>/` | `scripts/run_ablation.sh` |

Inside a run folder: `model_seed*.keras`, `history_seed*.json`, `model_seed*_int8.tflite`, `report.json`,
`farm_scores_<split>.csv`, `figures/heatmaps_*.png`, `fpga_export/`.

Keep the small things you cite (report.json, figures) by copying them to `docs/figures/` and logging the run in
`docs/EXPERIMENT_LOG.md`.
