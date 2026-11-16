"""Class definitions shared by the whole project (see docs/DATA_PROTOCOL.md)."""

CLASSES = ["exw_suspect", "other_condition", "healthy_background"]
EXW, OTHER, HEALTHY = 0, 1, 2
IGNORE = -1  # partly overlapping tiles: excluded from training and tile-level metrics
N_CLASSES = len(CLASSES)

FARM_LABELS = ["exw_suspect", "other_condition", "healthy"]
