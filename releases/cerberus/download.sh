#!/bin/sh
#
# Download Cerberus PyTorch model weights from the public GCS bucket.
# No Google account or gcloud required -- just curl.
#
# Usage: ./download.sh [FOLD|all] [DEST_DIR]
#
#   FOLD     0-7, or "all"          (default: all)
#   DEST_DIR where to write         (default: current dir)
#
# Each fold is one model with human and mouse heads. Files land at:
#   DEST_DIR/models/f<FOLD>c0/model_best.pth

set -eu

FOLD="${1:-all}"
DEST="${2:-.}"

if [ "$FOLD" = "all" ]; then
  FOLDS="0 1 2 3 4 5 6 7"
else
  FOLDS="$FOLD"
fi

BASE="https://storage.googleapis.com/seqnn-share/cerberus/models"
MODELDIR="$DEST/models"

mkdir -p "$MODELDIR"

for f in $FOLDS; do
  outdir="$MODELDIR/f${f}c0"
  mkdir -p "$outdir"
  echo ">> $BASE/f${f}c0/model_best.pth"
  curl -fL --retry 3 -o "$outdir/model_best.pth" "$BASE/f${f}c0/model_best.pth"
done

echo "done -> $MODELDIR/"
