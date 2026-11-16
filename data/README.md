# data/

Nothing in this folder except templates and `.gitkeep` files is committed. Real images and labels travel through the
team drive, never through git (see `docs/DATA_PROTOCOL.md`).

```
data/
  schema/                 templates and format definitions (committed)
    metadata_template.csv       one row per image, filled by whoever collects the photos
    boxes_example.json          coarse box annotation format
    manifest_columns.md         what every manifest column means
    capture_log_template.csv    field visit log (site, date, phone, weather)
    permission_template.md      data-sharing / permission record for each partner office
  raw/                    original photos exactly as received (never edited, not committed)
  interim/                cleaned / renamed / quality-checked copies (not committed)
  external/               public datasets used for pretraining or sanity checks (+ README with licence)
  synthetic/              Tier-0 generated scenes (git-ignored, recreated by `make data`)
  real/                   output of `scripts/ingest_real.py`: images/, manifest.csv, annotations.json (git-ignored)
  processed/              tile arrays (.npz) per split (git-ignored)
```

## Which file types appear where
| Type | Where | Meaning |
|---|---|---|
| `.jpg` / `.png` | raw, interim, real/images | wide-view photos |
| `.csv` | schema, real, results | metadata, manifest, per-image scores |
| `.json` | schema, real | box annotations |
| `.npz` | processed | uint8 tiles `X`, labels `y`, `image_id`, `xy` per split |

## Flow
`raw/` -> (rename + quality check) -> `interim/` -> `scripts/ingest_real.py` -> `real/` -> `build_all` -> `processed/`.
