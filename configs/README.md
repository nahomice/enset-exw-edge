# configs/

| File | Use |
|---|---|
| `default.yaml` | synthetic wide-view data, full smoke experiment (`make smoke`) |
| `smoke_test.yaml` | tiny and fast, to check wiring only |
| `real_data.yaml` | real images after `scripts/ingest_real.py`; set tile size here |
| `experiments/` | one file per planned experiment (documentation of what each run changes) |

Configs do not inherit from each other: each file is complete. To vary one value without a new file use
`--set key=value` (for example `--set tiles.size=128 --set tiles.stride=64`). Every cited result is recorded in
`docs/EXPERIMENT_LOG.md` with (git commit, config file, `--set` overrides, seed).
