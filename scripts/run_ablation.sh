#!/usr/bin/env bash
# X1 (tile size) and X2 (model width) ablations. Each combination gets its own data, tiles and results folder.
# Usage: bash scripts/run_ablation.sh [config]       (slow on CPU: run overnight)
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
for SIZE in 64 96 128; do
  STRIDE=$((SIZE / 2))
  for WIDTH in 0.5 1.0; do
    TAG="tile${SIZE}_w${WIDTH}"
    OV=(--config "$CONFIG"
        --set "tiles.size=${SIZE}" --set "tiles.stride=${STRIDE}" --set "model.width=${WIDTH}"
        --set "paths.data_dir=data/synthetic_${TAG}" --set "paths.tiles_dir=data/processed/${TAG}"
        --set "paths.results_dir=results/ablation/${TAG}" --set "train.seeds=[0]")
    echo "=== ${TAG} ==="
    python scripts/prepare_data.py "${OV[@]}"
    python scripts/train.py "${OV[@]}"
    python scripts/evaluate.py "${OV[@]}"
  done
done
echo "Done. Summaries: results/ablation/*/report.json"
