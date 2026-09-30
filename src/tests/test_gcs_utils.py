"""Unit tests for GCS folder download (no real GCS calls)."""

from __future__ import annotations

from baskerville.helpers import gcs_utils


class _FakeBlob:
    def __init__(self, name):
        self.name = name


class _FakeBucket:
    def __init__(self, blob_names):
        self._blob_names = blob_names
        self.seen_prefix = None

    def list_blobs(self, prefix):
        self.seen_prefix = prefix
        # emulate GCS string-prefix matching
        return [_FakeBlob(n) for n in self._blob_names if n.startswith(prefix)]


class _FakeClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


def test_download_folder_ignores_sibling_prefix(monkeypatch, tmp_path):
    """A sibling object sharing the dir name as a string prefix (train.err vs
    train/) must not be pulled into the download — that produced a '../' relpath
    and a 404 on resume. Regression for the restart-on-checkpoint crash."""
    blob_names = [
        "output/train/run/f0c0/train.err",  # sibling — must be skipped
        "output/train/run/f0c0/train.out",  # sibling — must be skipped
        "output/train/run/f0c0/train/",  # dir placeholder — must be skipped
        "output/train/run/f0c0/train/checkpoint.pth",
        "output/train/run/f0c0/train/sub/model_best0.pth",
    ]
    bucket = _FakeBucket(blob_names)
    monkeypatch.setattr(gcs_utils, "Client", lambda: _FakeClient(bucket))

    downloaded = []
    monkeypatch.setattr(
        gcs_utils,
        "download_from_gcs",
        lambda gcs_path, local_path, bytes=True: downloaded.append(gcs_path),
    )

    gcs_utils.download_folder_from_gcs(
        "gs://my-bucket/output/train/run/f0c0/train", str(tmp_path)
    )

    # listing must be pinned to the path boundary
    assert bucket.seen_prefix == "output/train/run/f0c0/train/"
    # only objects strictly beneath train/ are fetched; no '../' paths
    assert downloaded == [
        "gs://my-bucket/output/train/run/f0c0/train/checkpoint.pth",
        "gs://my-bucket/output/train/run/f0c0/train/sub/model_best0.pth",
    ]


def test_download_folder_exclude_regex_skips_weights(monkeypatch, tmp_path):
    """exclude_regex must skip matching blobs (the light periodic mirror passes
    r'.*\\.pth$' to pull progress/logs but not the large checkpoint weights)."""
    blob_names = [
        "output/train/run/f0c0/train/progress.json",
        "output/train/run/f0c0/train/log.txt",
        "output/train/run/f0c0/train/checkpoint.pth",  # excluded
        "output/train/run/f0c0/train/model_best.pth",  # excluded
        "output/train/run/f0c0/train/sub/model_best0.pth",  # excluded
    ]
    bucket = _FakeBucket(blob_names)
    monkeypatch.setattr(gcs_utils, "Client", lambda: _FakeClient(bucket))

    downloaded = []
    monkeypatch.setattr(
        gcs_utils,
        "download_from_gcs",
        lambda gcs_path, local_path, bytes=True: downloaded.append(gcs_path),
    )

    gcs_utils.download_folder_from_gcs(
        "gs://my-bucket/output/train/run/f0c0/train",
        str(tmp_path),
        exclude_regex=r".*\.pth$",
    )

    # only the non-.pth files are fetched
    assert downloaded == [
        "gs://my-bucket/output/train/run/f0c0/train/progress.json",
        "gs://my-bucket/output/train/run/f0c0/train/log.txt",
    ]
