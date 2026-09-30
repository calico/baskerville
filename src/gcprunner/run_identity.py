"""Run-identity labels for GCP Batch fold jobs — the single source of truth.

Concurrent distinct runs (different params/data → different ``run_id``) are told
apart by an exact match on these Batch labels, never by the human-readable
``--name``. Both sides of that match go through here:

- job **builders** (the ``hound_*_folds`` scripts) stamp ``identity_labels(...)``
  onto each Job;
- the **matchers** (double-launch detection, ``--conclude`` cancel/poll) build
  the same dict to filter active jobs.

Because the values are sanitized with the *same* ``_label_safe`` the spec builder
applies at submit time, a filter built here compares equal to what GCP actually
stored — even when a ``run_id`` contains characters GCP rewrites (uppercase, long
strings). Keeping the keys, the ``run_id`` derivation, and the sanitization in
one place is what makes the scheme safe to extend to new fold scripts.
"""

from __future__ import annotations

import os

from .batch_spec import _label_safe

# Label keys. Kept as constants so a rename touches one place, not ~30 literals.
RUN_KEY = "gcprunner_run"
KIND_KEY = "gcprunner_kind"
FOLD_KEY = "gcprunner_fold"


def run_id_from_gcs_dir(gcs_dir: str) -> str:
    """Extract the ``run_id`` from a run's GCS output dir (``…/<kind>/<run_id>``)."""
    return os.path.basename(gcs_dir.rstrip("/"))


def identity_labels(run_id: str, kind: str, fold: str | None = None) -> dict[str, str]:
    """Canonical (GCP-sanitized) identity labels for a fold job or a match filter.

    Values are passed through ``_label_safe`` — the same sanitizer
    ``build_batch_job_dict`` applies — so a filter built here matches the labels
    stored on the job. Omit ``fold`` to build a run-scoped filter (every fold of
    a run/kind), e.g. for ``--conclude``.
    """
    labels = {RUN_KEY: _label_safe(run_id), KIND_KEY: _label_safe(kind)}
    if fold is not None:
        labels[FOLD_KEY] = _label_safe(fold)
    return labels
