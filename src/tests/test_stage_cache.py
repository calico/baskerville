"""Unit tests for the input staging cache (no real GCS calls)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from baskerville.helpers import stage_cache


@pytest.fixture(autouse=True)
def _isolate_hash_cache(tmp_path, monkeypatch):
    """Redirect the on-disk hash cache to a tmp file for each test."""
    monkeypatch.setattr(
        stage_cache, "_local_hash_cache_path", lambda: tmp_path / "hashes.json"
    )


def test_hash_file_deterministic(tmp_path):
    p = tmp_path / "foo.bin"
    p.write_bytes(b"hello world\n")
    assert stage_cache.hash_file(str(p)) == stage_cache.hash_file(str(p))


def test_hash_file_sensitive_to_content(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    b = tmp_path / "b.bin"
    b.write_bytes(b"world")
    assert stage_cache.hash_file(str(a)) != stage_cache.hash_file(str(b))


def test_hash_file_cache_short_circuits(tmp_path, monkeypatch):
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * 1024)
    first = stage_cache.hash_file(str(p))

    calls = {"n": 0}
    orig = stage_cache._hash_bytes

    def counting(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(stage_cache, "_hash_bytes", counting)
    second = stage_cache.hash_file(str(p))
    assert first == second
    assert calls["n"] == 0  # cache hit, never re-read


def test_hash_dir_invariant_to_listing_order(tmp_path):
    root = tmp_path / "models"
    (root / "f1c0" / "train").mkdir(parents=True)
    (root / "f0c0" / "train").mkdir(parents=True)
    (root / "f0c0" / "train" / "model_best.pth").write_bytes(b"weights-0")
    (root / "f1c0" / "train" / "model_best.pth").write_bytes(b"weights-1")
    first = stage_cache.hash_dir(str(root))

    # Mutate a dotfile that should be skipped — hash must not change.
    (root / ".DS_Store").write_bytes(b"junk")
    assert stage_cache.hash_dir(str(root)) == first

    # Change actual content — hash MUST change.
    (root / "f0c0" / "train" / "model_best.pth").write_bytes(b"weights-0-modified")
    assert stage_cache.hash_dir(str(root)) != first


def test_stage_file_cache_hit(tmp_path, monkeypatch):
    p = tmp_path / "test.vcf"
    p.write_bytes(b"##fileformat=VCFv4.2\n")
    uploads = []
    monkeypatch.setattr(stage_cache, "gcs_file_exist", lambda uri: True)
    monkeypatch.setattr(
        stage_cache, "_upload_blob", lambda l, g: uploads.append((l, g))
    )
    monkeypatch.setattr(
        stage_cache, "_write_done_marker", lambda t, s: uploads.append(("DONE", t, s))
    )

    sha, container = stage_cache.stage_file(str(p), "vcf")
    assert uploads == []
    assert (
        container
        == f"{stage_cache.CONTAINER_CACHE_MOUNT}/vcf/{stage_cache._key(sha)}/test.vcf"
    )


def test_stage_file_uploads_renamed_content(tmp_path, monkeypatch):
    """Same content staged under an old basename must not count as a hit."""
    p = tmp_path / "v2.json"
    p.write_bytes(b'{"k": 1}')
    uploads = []
    # the .DONE marker is there, but only under the pre-rename basename
    monkeypatch.setattr(
        stage_cache, "gcs_file_exist", lambda uri: not uri.endswith("/v2.json")
    )
    monkeypatch.setattr(stage_cache, "_upload_blob", lambda l, g: uploads.append(g))
    monkeypatch.setattr(
        stage_cache, "_write_done_marker", lambda t, s: uploads.append("DONE")
    )

    sha, container = stage_cache.stage_file(str(p), "params")
    key = stage_cache._key(sha)
    assert uploads == [f"{stage_cache.cache_prefix()}/params/{key}/v2.json", "DONE"]
    assert container == f"{stage_cache.CONTAINER_CACHE_MOUNT}/params/{key}/v2.json"


def test_stage_file_upload_when_missing(tmp_path, monkeypatch):
    p = tmp_path / "params.json"
    p.write_bytes(b'{"k": 1}')
    uploads = []
    monkeypatch.setattr(stage_cache, "gcs_file_exist", lambda uri: False)
    monkeypatch.setattr(stage_cache, "_upload_blob", lambda l, g: uploads.append(g))
    monkeypatch.setattr(
        stage_cache,
        "_write_done_marker",
        lambda t, s: uploads.append(f"DONE:{t}/{s[:6]}"),
    )

    sha, container = stage_cache.stage_file(str(p), "params")
    assert any(u.endswith("/params.json") for u in uploads if isinstance(u, str))
    assert uploads[-1] == f"DONE:params/{sha[:6]}"


def test_stage_dir_uploads_each_file(tmp_path, monkeypatch):
    root = tmp_path / "models"
    (root / "f0c0" / "train").mkdir(parents=True)
    (root / "f1c0" / "train").mkdir(parents=True)
    (root / "f0c0" / "train" / "model_best.pth").write_bytes(b"a")
    (root / "f1c0" / "train" / "model_best.pth").write_bytes(b"b")

    uploads = []
    monkeypatch.setattr(stage_cache, "gcs_file_exist", lambda uri: False)
    monkeypatch.setattr(stage_cache, "_upload_blob", lambda l, g: uploads.append(g))
    monkeypatch.setattr(
        stage_cache, "_write_done_marker", lambda t, s: uploads.append("DONE")
    )

    sha, container = stage_cache.stage_dir(str(root), "models")
    key = stage_cache._key(sha)
    assert container == f"{stage_cache.CONTAINER_CACHE_MOUNT}/models/{key}"
    # both per-fold weight files uploaded under the sha prefix at their tree positions
    expected_a = f"{stage_cache.cache_prefix()}/models/{key}/f0c0/train/model_best.pth"
    expected_b = f"{stage_cache.cache_prefix()}/models/{key}/f1c0/train/model_best.pth"
    assert expected_a in uploads
    assert expected_b in uploads
    assert uploads[-1] == "DONE"  # marker written last


def test_build_run_id_shape():
    rid = stage_cache.build_run_id("abc12345" * 8, "def67890" * 8)
    # YYYY-MM-DDTHHMMSS-abc12345-def67890
    parts = rid.split("-")
    assert len(parts) == 5
    assert parts[-2] == "abc12345"
    assert parts[-1] == "def67890"


def test_build_run_id_deterministic_pure_hash():
    """Deterministic run id is a stable pure hash, no timestamp prefix."""
    a = stage_cache.build_run_id("abc12345" * 8, "def67890" * 8, deterministic=True)
    b = stage_cache.build_run_id("abc12345" * 8, "def67890" * 8, deterministic=True)
    assert a == b == "abc12345-def67890"


def test_build_run_id_changes_with_inputs():
    base = stage_cache.build_run_id("a" * 64, "b" * 64, deterministic=True)
    diff_params = stage_cache.build_run_id("c" * 64, "b" * 64, deterministic=True)
    diff_data = stage_cache.build_run_id("a" * 64, "d" * 64, deterministic=True)
    assert base != diff_params
    assert base != diff_data


def test_prefixes_require_env(monkeypatch):
    """No built-in buckets: the prefix accessors raise when their env var is unset."""
    monkeypatch.delenv("GCPRUNNER_CACHE_PREFIX", raising=False)
    monkeypatch.delenv("GCPRUNNER_OUTPUT_PREFIX", raising=False)
    with pytest.raises(RuntimeError, match="GCPRUNNER_CACHE_PREFIX"):
        stage_cache.cache_prefix()
    with pytest.raises(RuntimeError, match="GCPRUNNER_OUTPUT_PREFIX"):
        stage_cache.output_prefix()
    monkeypatch.setenv("GCPRUNNER_OUTPUT_PREFIX", "gs://my-bucket/output/")
    assert stage_cache.output_prefix() == "gs://my-bucket/output"
