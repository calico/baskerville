#!/bin/sh
#
# Download Borzoi Prime PyTorch model weights from the public GCS bucket.
# No Google account or gcloud required -- just curl.
#
# Usage: ./download.sh [human|mouse] [FOLD|all] [DEST_DIR]
#
#   FOLD     0, 1, 2, 3, or "all"   (default: all)
#   DEST_DIR where to write         (default: current dir)
#
# Files land at:
#   DEST_DIR/models_<species>/f<FOLD>c0/model_best.pth

set -eu

SPECIES="${1:-human}"
FOLD="${2:-all}"
DEST="${3:-.}"

case "$SPECIES" in
  human|mouse) ;;
  *) echo "error: species must be 'human' or 'mouse' (got '$SPECIES')"; exit 1 ;;
esac

if [ "$FOLD" = "all" ]; then
  FOLDS="0 1 2 3"
else
  FOLDS="$FOLD"
fi

BASE="https://storage.googleapis.com/seqnn-share/prime_pt"
FAMILY="$BASE/models_$SPECIES"
FAMILYDIR="$DEST/models_$SPECIES"

mkdir -p "$FAMILYDIR"

for f in $FOLDS; do
  outdir="$FAMILYDIR/f${f}c0"
  mkdir -p "$outdir"
  echo ">> $FAMILY/f${f}c0/model_best.pth"
  curl -fL --retry 3 -o "$outdir/model_best.pth" "$FAMILY/f${f}c0/model_best.pth"
done

echo "done -> $FAMILYDIR/"
