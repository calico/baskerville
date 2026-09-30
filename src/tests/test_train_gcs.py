"""Unit tests for training GCS resume helpers (no real GCS calls)."""

from __future__ import annotations

from baskerville.helpers import train_gcs


def test_resolve_gcs_dest_none_off_gcp(monkeypatch):
    monkeypatch.delenv("GCPRUNNER_OUTPUT_DIR_GCS", raising=False)
    monkeypatch.delenv("GCPRUNNER_OUTPUT_DIR_LOCAL", raising=False)
    assert train_gcs.resolve_gcs_dest("/some/out/train") is None


def test_resolve_gcs_dest_maps_relpath(monkeypatch):
    monkeypatch.setenv("GCPRUNNER_OUTPUT_DIR_GCS", "gs://b/output/train/h-d/f0c0")
    monkeypatch.setenv("GCPRUNNER_OUTPUT_DIR_LOCAL", "/workspace/out")
    assert (
        train_gcs.resolve_gcs_dest("/workspace/out/train")
        == "gs://b/output/train/h-d/f0c0/train"
    )


def test_resolve_gcs_dest_root(monkeypatch):
    monkeypatch.setenv("GCPRUNNER_OUTPUT_DIR_GCS", "gs://b/output/train/h-d/f0c0")
    monkeypatch.setenv("GCPRUNNER_OUTPUT_DIR_LOCAL", "/workspace/out")
    # out_dir == local root → dest is the GCS root (no trailing slash)
    assert (
        train_gcs.resolve_gcs_dest("/workspace/out") == "gs://b/output/train/h-d/f0c0"
    )


def test_resolve_gcs_dest_partial_env(monkeypatch):
    # only one of the two env vars set → not on GCP
    monkeypatch.setenv("GCPRUNNER_OUTPUT_DIR_GCS", "gs://b/x")
    monkeypatch.delenv("GCPRUNNER_OUTPUT_DIR_LOCAL", raising=False)
    assert train_gcs.resolve_gcs_dest("/workspace/out/train") is None


def test_prune_callback_deletes_gcs_copy(monkeypatch):
    deleted = []
    monkeypatch.setattr(train_gcs, "delete_from_gcs", lambda uri: deleted.append(uri))
    prune = train_gcs.make_prune_callback("gs://b/output/train/h-d/f0c0/train")
    prune("model_best3.pth")
    assert deleted == ["gs://b/output/train/h-d/f0c0/train/model_best3.pth"]
