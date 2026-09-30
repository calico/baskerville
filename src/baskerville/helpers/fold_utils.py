"""Shared helpers for cross-fold evaluation scripts.

Both ``hound_eval_folds`` and ``hound_eval_genes_folds`` need the same logic to
(a) read the authoritative fold assignment a replicate was trained with,
(b) decide whether a replicate's weights exist (locally or in GCS marker mode),
and (c) resolve the ``model_best.pth`` path to pass to the eval command. They're
collected here so the fold scripts share one implementation.
"""

from __future__ import annotations

import json
import os

from baskerville import dataset
from baskerville.helpers.gcs_utils import gcs_file_exist

# Container mount point for a GCP-trained models tree read straight from GCS
# (marker mode). Distinct from the dataset (/workspace/data) and content-cache
# (/workspace/cache) mounts.
GCP_MODELS_MOUNT = "/workspace/models"


def read_fold_splits(train_dir, num_folds, fi, ci):
    """Return the ``(test_fold, valid_fold)`` training actually used.

    Training records the authoritative assignment in ``{train_dir}/folds.json``
    (see hound_train). Reading it keeps eval perfectly aligned with how the
    model was trained — in particular ``valid_fold = (fold + 1 + cross) %
    num_folds``, which is *not* ``fold + 1`` once cross > 0. Falls back to
    recomputing via the same ``dataset.compute_fold_splits`` if the file is
    missing (e.g. models trained before folds.json existed).
    """
    folds_file = f"{train_dir}/folds.json"
    if os.path.isfile(folds_file):
        with open(folds_file) as f:
            folds_log = json.load(f)
        return folds_log["test_fold"], folds_log["valid_fold"]
    print(
        f"Warning: {folds_file} not found; falling back to computed fold splits "
        f"for f{fi}c{ci}."
    )
    splits = dataset.compute_fold_splits(num_folds, fi, ci)
    return splits["test"], splits["valid"]


def model_present(gcp_backend, gcs_models_dir, train_dir, fold_cross):
    """Whether a replicate's weights exist (GCS in marker mode, else local)."""
    if gcp_backend and gcs_models_dir is not None:
        return gcs_file_exist(f"{gcs_models_dir}/{fold_cross}/train/model_best.pth")
    return os.path.isfile(f"{train_dir}/model_best.pth")


def resolve_model_file(
    gcp_backend, gcs_models_dir, container_models_dir, train_dir, fold_cross
):
    """The ``model_best.pth`` path to pass to the eval command for this replicate."""
    if not gcp_backend:
        return f"{train_dir}/model_best.pth"
    if gcs_models_dir is not None:
        return f"{GCP_MODELS_MOUNT}/{fold_cross}/train/model_best.pth"
    return f"{container_models_dir}/{fold_cross}/train/model_best.pth"
