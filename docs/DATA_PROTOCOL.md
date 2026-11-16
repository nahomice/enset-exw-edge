# Data protocol

## Tiers
| Tier | Content | Use |
|---|---|---|
| 0 | synthetic wide-view scenes (`exw.data.synthetic`) | pipeline, tests, hardware vectors. **Never** accuracy claims |
| 1 | real healthy / background photos, taken from the field edge | background class, false-alarm rate |
| 2 | real EXW and other-condition photos from EAA / partners (expert-confirmed) | positives |
| 3 | coarse boxes on Tier-2 images | tile labels |

## Capture (field edge, no drone required)
- Stand at the field edge, phone held at chest height, several plants in view, **whole plants, not leaf close-ups**.
- 3-5 views per farm from slightly different positions; note site, farm, date, weather; keep original JPEGs.
- Include look-alikes: other diseases, drought, weeds, shadows. Include **healthy farms**: false alarms matter most.
- Photo resolution: at least 12 MP. Record the phone model.

## Metadata (`metadata.csv`)
`filename, site_id, farm_id, date, farm_label (healthy | other_condition | exw_suspect), tier`. The label is the
**expert's confirmed** field diagnosis, not what the model thinks.

## Boxes and tile labels (derived by program, not by hand)
Annotators draw coarse boxes around affected plants (class 0 = EXW-suspect, 1 = other condition).
Tile rule (`configs/default.yaml: tiles`):
- tile takes a box's class if >= 50 % of the **tile** lies inside the box (EXW has priority);
- 10-50 % overlap -> **IGNORE** (excluded from training);
- every other tile of a **boxed** image -> healthy/background;
- images **without boxes** produce no tile labels; they are used for farm-level evaluation only.

Known limitation: plants too small in the image (far rows) never fill 50 % of a tile and become IGNORE or background.
**Tile size must be chosen against plant size in pixels** (the synthetic generator exposes `plant_scale` for this).

## Splits
Group by `site_id` (or `farm_id` / `date`) **before** tiling; stratified by farm label; `assert_no_leakage` checks site
and farm. Needs roughly >= 10 sites per class for a meaningful val/test split. Near-duplicate photos are reported by
`ingest_real.py`; review them, never let a pair straddle splits.

## Quality checks
Blur (variance of Laplacian), exposure, perceptual-hash duplicates. Problems are **reported**, not silently dropped.

## Not committed
Real images, labels with farm identifiers, and models stay out of git. Share via the team drive with permission notes.
