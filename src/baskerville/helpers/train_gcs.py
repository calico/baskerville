"""GCS restore/checkpoint helpers for training on the GCP backend.

When ``hound_train`` runs inside a gcprunner container, the gcprunner entrypoint
sets ``GCPRUNNER_OUTPUT_DIR_LOCAL`` (the local dir whose contents are synced to
GCS on exit) and ``GCPRUNNER_OUTPUT_DIR_GCS`` (the gs:// destination). These
helpers let the trainer:

1. **Restore** a prior checkpoint into ``out_dir`` before training, so the
   trainer's existing auto-resume (loads ``checkpoint.pth`` if present) picks up
   where a preempted/crashed attempt left off.
2. **Sync** ``out_dir`` to GCS after every epoch checkpoint (wired as the
   Trainer's ``checkpoint_callback``), so progress — including ``log.txt`` and
   the terminal COMPLETE/FAILED marker — is durable mid-run, not just on exit.

Outside a gcprunner container (Slurm/local) the env vars are absent and these
helpers are inert.
"""

from __future__ import annotations

import os

from baskerville.helpers.gcs_utils import (
    delete_from_gcs,
    download_folder_from_gcs,
    gcs_file_exist,
    sync_dir_to_gcs,
)


def resolve_gcs_dest(out_dir: str) -> str | None:
    """Return the gs:// destination mirroring ``out_dir``, or None if not on GCP.

    The container's output dir maps to GCS by relative path:
    ``GCPRUNNER_OUTPUT_DIR_GCS / relpath(out_dir, GCPRUNNER_OUTPUT_DIR_LOCAL)``.
    """
    gcs_root = os.environ.get("GCPRUNNER_OUTPUT_DIR_GCS")
    local_root = os.environ.get("GCPRUNNER_OUTPUT_DIR_LOCAL")
    if not gcs_root or not local_root:
        return None
    rel = os.path.relpath(os.path.abspath(out_dir), os.path.abspath(local_root))
    if rel == ".":
        return gcs_root.rstrip("/")
    return f"{gcs_root.rstrip('/')}/{rel}"


def restore_outdir(out_dir: str, gcs_dest: str) -> bool:
    """Download a prior checkpoint tree from ``gcs_dest`` into ``out_dir``.

    Returns True if a resumable ``checkpoint.pth`` was found and restored. No-op
    (returns False) when the GCS prefix has no checkpoint — i.e. a fresh fold.
    """
    if not gcs_file_exist(f"{gcs_dest}/checkpoint.pth"):
        return False
    os.makedirs(out_dir, exist_ok=True)
    download_folder_from_gcs(gcs_dest, out_dir)
    return True


def make_sync_callback(out_dir: str, gcs_dest: str):
    """Return a no-arg callable that mirrors ``out_dir`` to ``gcs_dest``."""

    def _sync() -> None:
        sync_dir_to_gcs(out_dir, gcs_dest, recursive=True)

    return _sync


def make_prune_callback(gcs_dest: str):
    """Return a callable(filename) that deletes ``gcs_dest/filename`` from GCS.

    The trainer prunes old ``model_best`` snapshots locally, but the per-epoch
    sync is upload-only, so without this their GCS copies would accumulate
    forever. Wiring this to the prune keeps GCS within the same window as disk.
    """

    def _prune(filename: str) -> None:
        delete_from_gcs(f"{gcs_dest}/{filename}")

    return _prune
