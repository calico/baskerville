"""Post-job finalization for GCP Batch SNP scoring.

When ``hound_snp_folds`` runs against the GCP backend, each per-job
``scores.h5`` lands in ``$GCPRUNNER_OUTPUT_PREFIX/snp/<run-id>/<fold_cross>/jobN/``
via gcprunner's exit-time rsync. This module mirrors what the slurm backend does
locally: download the per-job HDF5s, run ``collect_scores()`` to merge them,
upload the consolidated ``scores.h5`` back to GCS, and (when a local mirror is
requested) copy the merged per-fold files there in the same pass. ``fetch_results``
remains for callers that produce final per-fold files without a merge step.
"""

from __future__ import annotations

import os
import tempfile

from google.cloud.storage import Client

from baskerville.helpers.gcs_utils import split_gcs_uri
from baskerville.multi import collect_scores


def _upload_file(local_path: str, gcs_uri: str) -> None:
    bucket_name, object_name = split_gcs_uri(gcs_uri)
    client = Client()
    client.bucket(bucket_name).blob(object_name).upload_from_filename(local_path)


def finalize_fold(
    gcs_fold_dir: str,
    num_jobs: int,
    quantile_copy: bool,
    local_fold_dir: str | None = None,
) -> None:
    """Merge per-job ``scores.h5`` for one fold and upload the result.

    Args:
        gcs_fold_dir: ``gs://.../<run-id>/<fold_cross>`` — directory holding
            ``job0/``, ``job1/``, ... subdirs that gcprunner rsync'd from each VM.
        num_jobs: number of per-job subdirs to expect.
        quantile_copy: passed through to ``collect_scores``.
        local_fold_dir: if given, the merged outputs (``scores.h5`` plus any
            ``targets_*.txt``) are kept here in addition to being uploaded,
            avoiding a re-download from GCS. The per-job shards are staged in a
            temp dir inside it (so they land on the output's disk, not ``/tmp``)
            and removed afterward. If ``None``, the merge stays in a throwaway
            temp dir under ``/tmp`` (cloud-only, no local mirror).
    """
    bucket_name, fold_prefix = split_gcs_uri(gcs_fold_dir.rstrip("/"))
    client = Client()
    bucket = client.bucket(bucket_name)

    if local_fold_dir is not None:
        os.makedirs(local_fold_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="finalize_fold_", dir=local_fold_dir
    ) as tmp:
        # mirror gs://.../jobN/* → tmp/jobN/* for every job
        for job_i in range(num_jobs):
            job_prefix = f"{fold_prefix}/job{job_i}/"
            local_job_dir = os.path.join(tmp, f"job{job_i}")
            os.makedirs(local_job_dir, exist_ok=True)
            blobs = list(bucket.list_blobs(prefix=job_prefix))
            if not blobs:
                raise RuntimeError(
                    f"No blobs under gs://{bucket_name}/{job_prefix} — "
                    "did the Batch job for this shard fail or fail to sync?"
                )
            for blob in blobs:
                # Skip the directory placeholder object, if any.
                if blob.name == job_prefix:
                    continue
                rel = os.path.relpath(blob.name, job_prefix)
                dest = os.path.join(local_job_dir, rel)
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                blob.download_to_filename(dest)

        collect_scores(tmp, num_jobs, quantile_copy)

        # upload the merged outputs (scores.h5 plus any targets_*.txt) and, when
        # a local mirror is requested, move them up into it — collect_scores
        # writes only these consolidated files at the temp-dir top level, so the
        # per-job shards under jobN/ stay out of the mirror.
        for name in os.listdir(tmp):
            full = os.path.join(tmp, name)
            if not os.path.isfile(full):
                continue
            _upload_file(full, f"{gcs_fold_dir.rstrip('/')}/{name}")
            if local_fold_dir is not None:
                os.replace(full, os.path.join(local_fold_dir, name))


def fetch_results(gcs_out_dir: str, local_dir: str, fold_crosses: list[str]) -> None:
    """Download per-fold merged outputs (no intermediates) to a local mirror.

    Args:
        gcs_out_dir: ``gs://.../<run-id>``.
        local_dir: local directory; ``<fold_cross>/scores.h5`` written under it.
        fold_crosses: e.g. ``["f0c0", "f1c0", ...]``.
    """
    bucket_name, run_prefix = split_gcs_uri(gcs_out_dir.rstrip("/"))
    client = Client()
    bucket = client.bucket(bucket_name)

    for fc in fold_crosses:
        fold_dir_local = os.path.join(local_dir, fc)
        os.makedirs(fold_dir_local, exist_ok=True)
        # The merged files live directly under <run>/<fc>/, not in any jobN/ subdir.
        # List only at depth 1 by using delimiter and filtering.
        for name in (
            "scores.h5",
            "targets_cov.txt",
            "targets_covgene.txt",
            "targets_gene.txt",
        ):
            blob = bucket.blob(f"{run_prefix}/{fc}/{name}")
            if blob.exists():
                blob.download_to_filename(os.path.join(fold_dir_local, name))
