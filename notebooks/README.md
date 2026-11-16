# notebooks/

Exploration only; anything that produces a number for the thesis must also exist as a script in `scripts/` or a function
in `src/exw/`. Clear outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`).

| Notebook | Purpose |
|---|---|
| `01_explore_data.ipynb` | split counts, site/farm leakage check, contact sheet |
| `02_inspect_tiles_and_labels.ipynb` | tile grid + boxes + derived labels on one image; choose tile size |
| `03_results_figures.ipynb` | tables and figures from `report.json` (ROC, F1, latency) |
