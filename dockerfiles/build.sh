#!/bin/bash
# Build and push the gcprunner image with Cloud Build; see docs/gcprunner.md §8.2.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# The image is tagged commit-<sha>, so it must contain exactly that commit.
if [[ -n $(git status --porcelain) ]]; then
  echo "Working tree is dirty; commit (e.g. to a branch) before building." >&2
  git status --short >&2
  exit 1
fi

PROJECT=${GCPRUNNER_PROJECT:?set GCPRUNNER_PROJECT}
REGION=${GCPRUNNER_REGION:-us-central1}
IMAGE=${GCPRUNNER_IMAGE_BASE:-$REGION-docker.pkg.dev/$PROJECT/baskerville/baskerville}

gcloud builds submit --project="$PROJECT" --region="$REGION" \
  --config=dockerfiles/cloudbuild.yaml --ignore-file=.dockerignore \
  --substitutions="_IMAGE=$IMAGE,_SHA=$(git rev-parse HEAD)" .
