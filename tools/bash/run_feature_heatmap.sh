#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-configs_reid/reid_nuscenes_pts/testing/testing_mpfan.py}
CKPT=${2:-}
SPLIT=${3:-val}
INDEX=${4:-0}
OUTDIR=${5:-runs/feature_heatmaps}

PYTHON=${PYTHON:-python}

$PYTHON tools/analysis/feature_heatmap.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --split "$SPLIT" \
  --index "$INDEX" \
  --out-dir "$OUTDIR" \
  --method pca \
  --match backbone ED_DualReID transformer gcn edge \
  "$@"

echo "Feature heatmap saved under $OUTDIR"


