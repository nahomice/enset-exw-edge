# Manifest columns (`manifest.csv`, one row per image)

| Column | Type | Meaning |
|---|---|---|
| image_id | str | unique id (file stem) |
| path | str | image path relative to the dataset folder |
| site_id | str/int | geographic site; default **grouping unit for splits** |
| farm_id | str/int | one farm; a farm never appears in two splits |
| date | YYYY-MM-DD | capture date |
| source | str | `synthetic` or `real` |
| tier | int | 0 synthetic, 1 real healthy/background, 2 real EXW/other (expert confirmed), 3 with boxes |
| style | int | synthetic appearance style; -1 for real images |
| farm_label | str | `healthy`, `other_condition`, `exw_suspect` (expert-confirmed field diagnosis) |
| has_boxes | bool | image carries coarse boxes (otherwise farm-level evaluation only) |
| sha256 | str | file hash, detects silent changes and exact duplicates |
| split | str | `train`, `val`, `test`, `test_shift` (assigned by `make_splits`, never by hand) |

`annotations.json`: `{image_id: {"boxes": [{x0,y0,x1,y1,cls}], "boxed": bool}}`, `cls` 0 = EXW-suspect, 1 = other condition.
Healthy plants are never boxed. Tile labels are derived from boxes by `exw.data.labels.tile_labels`.
