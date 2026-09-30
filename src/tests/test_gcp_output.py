"""Unit tests for the post-job merge + fetch helpers (no real GCS calls)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from baskerville.helpers import gcp_output


class _FakeBlob:
    def __init__(self, name, data=b"x", exists_=True):
        self.name = name
        self._data = data
        self._exists = exists_

    def download_to_filename(self, dest):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(self._data)

    def upload_from_filename(self, src):
        with open(src, "rb") as f:
            self._data = f.read()

    def exists(self):
        return self._exists


class _FakeBucket:
    def __init__(self, blobs):
        self._blobs = {b.name: b for b in blobs}

    def list_blobs(self, prefix):
        return [b for n, b in self._blobs.items() if n.startswith(prefix)]

    def blob(self, name):
        return self._blobs.setdefault(name, _FakeBlob(name, data=b"", exists_=False))


def _patch_client(monkeypatch, bucket):
    client = MagicMock()
    client.bucket.return_value = bucket
    monkeypatch.setattr(gcp_output, "Client", lambda: client)


def test_finalize_fold_downloads_jobs_runs_collect_and_uploads(monkeypatch):
    bucket = _FakeBucket(
        [
            _FakeBlob("runs/r1/f0c0/job0/scores.h5", data=b"j0-scores"),
            _FakeBlob("runs/r1/f0c0/job0/targets_cov.txt", data=b"j0-targets"),
            _FakeBlob("runs/r1/f0c0/job1/scores.h5", data=b"j1-scores"),
        ]
    )
    _patch_client(monkeypatch, bucket)

    captured = {}

    def fake_collect(out_dir, num_jobs, quantile_copy):
        captured["dir"] = out_dir
        captured["num_jobs"] = num_jobs
        captured["quantile_copy"] = quantile_copy
        # what collect_scores actually does: write a consolidated scores.h5 at root
        with open(os.path.join(out_dir, "scores.h5"), "wb") as f:
            f.write(b"merged")
        with open(os.path.join(out_dir, "targets_cov.txt"), "wb") as f:
            f.write(b"targets")

    monkeypatch.setattr(gcp_output, "collect_scores", fake_collect)

    gcp_output.finalize_fold("gs://bkt/runs/r1/f0c0", 2, quantile_copy=False)

    assert captured["num_jobs"] == 2
    assert captured["quantile_copy"] is False
    # Verify the merged outputs landed back at <fold>/scores.h5 and <fold>/targets_cov.txt
    assert bucket._blobs["runs/r1/f0c0/scores.h5"]._data == b"merged"
    assert bucket._blobs["runs/r1/f0c0/targets_cov.txt"]._data == b"targets"


def test_finalize_fold_mirrors_merged_files_locally(tmp_path, monkeypatch):
    bucket = _FakeBucket(
        [
            _FakeBlob("runs/r1/f0c0/job0/scores.h5", data=b"j0-scores"),
            _FakeBlob("runs/r1/f0c0/job1/scores.h5", data=b"j1-scores"),
        ]
    )
    _patch_client(monkeypatch, bucket)

    def fake_collect(out_dir, num_jobs, quantile_copy):
        with open(os.path.join(out_dir, "scores.h5"), "wb") as f:
            f.write(b"merged")
        with open(os.path.join(out_dir, "targets_cov.txt"), "wb") as f:
            f.write(b"targets")

    monkeypatch.setattr(gcp_output, "collect_scores", fake_collect)

    local_fold_dir = str(tmp_path / "f0c0")
    gcp_output.finalize_fold(
        "gs://bkt/runs/r1/f0c0", 2, quantile_copy=False, local_fold_dir=local_fold_dir
    )

    # Merged outputs uploaded back to GCS...
    assert bucket._blobs["runs/r1/f0c0/scores.h5"]._data == b"merged"
    # ...and copied into the local mirror without a re-download.
    assert (tmp_path / "f0c0" / "scores.h5").read_bytes() == b"merged"
    assert (tmp_path / "f0c0" / "targets_cov.txt").read_bytes() == b"targets"
    # Per-job intermediates never land in the mirror.
    assert not (tmp_path / "f0c0" / "job0").exists()


def test_finalize_fold_raises_when_job_missing(monkeypatch):
    bucket = _FakeBucket(
        [
            _FakeBlob("runs/r1/f0c0/job0/scores.h5", data=b"j0"),
            # job1 has no blobs
        ]
    )
    _patch_client(monkeypatch, bucket)
    monkeypatch.setattr(gcp_output, "collect_scores", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="No blobs under"):
        gcp_output.finalize_fold("gs://bkt/runs/r1/f0c0", 2, quantile_copy=False)


def test_fetch_results_skips_intermediates(tmp_path, monkeypatch):
    bucket = _FakeBucket(
        [
            _FakeBlob("runs/r1/f0c0/scores.h5", data=b"f0-merged", exists_=True),
            _FakeBlob("runs/r1/f0c0/job0/scores.h5", data=b"f0-j0", exists_=True),
            _FakeBlob("runs/r1/f1c0/scores.h5", data=b"f1-merged", exists_=True),
            _FakeBlob("runs/r1/f1c0/targets_cov.txt", data=b"f1-targets", exists_=True),
        ]
    )
    _patch_client(monkeypatch, bucket)

    gcp_output.fetch_results("gs://bkt/runs/r1", str(tmp_path), ["f0c0", "f1c0"])

    assert (tmp_path / "f0c0" / "scores.h5").read_bytes() == b"f0-merged"
    assert (tmp_path / "f1c0" / "scores.h5").read_bytes() == b"f1-merged"
    assert (tmp_path / "f1c0" / "targets_cov.txt").read_bytes() == b"f1-targets"
    # No per-job intermediates ended up locally.
    assert not (tmp_path / "f0c0" / "job0").exists()


def test_fetch_results_missing_blob_is_skipped(tmp_path, monkeypatch):
    bucket = _FakeBucket(
        [
            _FakeBlob("runs/r1/f0c0/scores.h5", data=b"present", exists_=True),
        ]
    )
    _patch_client(monkeypatch, bucket)

    # targets_cov.txt and targets_gene.txt blobs don't exist — should be no-ops.
    gcp_output.fetch_results("gs://bkt/runs/r1", str(tmp_path), ["f0c0"])
    assert (tmp_path / "f0c0" / "scores.h5").exists()
    assert not (tmp_path / "f0c0" / "targets_cov.txt").exists()
