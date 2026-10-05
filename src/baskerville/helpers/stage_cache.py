"""Content-addressed input staging for GCP Batch runs.

When the GCP backend is selected, hash local inputs (VCF, models dir, params,
fasta, targets), check whether they already exist under ``GCPRUNNER_CACHE_PREFIX``,
upload the missing ones, and return the in-container path the user's script
should pass to ``hound_snp`` (etc.). The container sees these paths via a
GCSFuse mount of the cache bucket at ``/workspace/cache``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from google.cloud.storage import Client

from baskerville.helpers.gcs_utils import (
    gcs_file_exist,
    split_gcs_uri,
)


def _require_env(name: str, example: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set: point it at a gs:// prefix in a bucket you own "
            f'(e.g. {example}). See README "GCP configuration".'
        )
    return value.rstrip("/")


# No defaults on purpose: these must name buckets you own. Read at call time
# (not import) so non-GCP backends never need them set.
def cache_prefix() -> str:
    """gs:// prefix for content-addressed staged inputs (GCPRUNNER_CACHE_PREFIX)."""
    return _require_env("GCPRUNNER_CACHE_PREFIX", "gs://my-bucket/cache")


def output_prefix() -> str:
    """gs:// prefix under which run outputs land (GCPRUNNER_OUTPUT_PREFIX)."""
    return _require_env("GCPRUNNER_OUTPUT_PREFIX", "gs://my-bucket/output")


CONTAINER_CACHE_MOUNT = os.environ.get(
    "GCPRUNNER_CONTAINER_CACHE_MOUNT", "/workspace/cache"
)

# Marker a GCP fold run drops at the top of its local ``-o`` dir, recording the
# run's GCS output dir + GCP config. The local mirror excludes ``.pth`` so a
# GCP-trained models tree has no weights locally; this lets a downstream command
# (eval) find the weights in GCS and inherit the run's GCP settings without the
# user re-specifying them. It's a ``.json`` so the mirror's ``.pth`` filter never
# touches it.
RUN_MARKER_NAME = "gcp_run.json"

_CHUNK = 1 << 20  # 1 MiB

# Number of hex chars from the SHA256 used for cache directory names.
# 16 hex = 64 bits — collision probability is negligible at our scale
# (~10^-15 for 1k entries per type) while keeping paths readable.
_KEY_LEN = 16


def _key(sha: str) -> str:
    return sha[:_KEY_LEN]


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def _local_hash_cache_path() -> Path:
    return Path.home() / ".cache" / "baskerville" / "stage_hashes.json"


def _load_hash_cache() -> dict:
    p = _local_hash_cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_hash_cache(cache: dict) -> None:
    p = _local_hash_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _hash_bytes(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_file(path: str) -> str:
    """SHA256 of a file's contents. Memoized in ``~/.cache/baskerville``."""
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    key = f"file:{abspath}"
    cache = _load_hash_cache()
    entry = cache.get(key)
    if entry and entry["size"] == st.st_size and entry["mtime"] == st.st_mtime:
        return entry["hash"]
    digest = _hash_bytes(abspath)
    cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "hash": digest}
    _save_hash_cache(cache)
    return digest


def _iter_dir_files(root: str, *, filename: str | None = None):
    """Walk ``root`` yielding ``(relpath, abspath)``. Skips dotfiles/dotdirs.

    If ``filename`` is set, only files whose basename equals it are yielded —
    used to stage just the canonical weight files out of model trees that
    also contain large unrelated artifacts.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if filename is not None and name != filename:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            yield rel, full


def hash_text(text: str) -> str:
    """SHA-256 hex of a string, e.g. a job command keying a run id."""
    return hashlib.sha256(text.encode()).hexdigest()


def hash_dir(path: str, *, filename: str | None = None) -> str:
    """SHA256 of the sorted ``(relpath, file_sha256)`` manifest of a directory."""
    manifest = hashlib.sha256()
    for rel, full in _iter_dir_files(path, filename=filename):
        manifest.update(rel.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(hash_file(full).encode("ascii"))
        manifest.update(b"\0")
    return manifest.hexdigest()


# ---------------------------------------------------------------------------
# cache lookup + upload
# ---------------------------------------------------------------------------


def _done_marker_uri(type_: str, sha: str) -> str:
    return f"{cache_prefix()}/{type_}/{_key(sha)}/.DONE"


def _container_path_file(type_: str, sha: str, basename: str) -> str:
    return f"{CONTAINER_CACHE_MOUNT}/{type_}/{_key(sha)}/{basename}"


def _container_path_dir(type_: str, sha: str) -> str:
    return f"{CONTAINER_CACHE_MOUNT}/{type_}/{_key(sha)}"


def _upload_blob(local_path: str, gcs_uri: str) -> None:
    bucket_name, object_name = split_gcs_uri(gcs_uri)
    client = Client()
    blob = client.bucket(bucket_name).blob(object_name)
    blob.upload_from_filename(local_path)


def _write_done_marker(type_: str, sha: str) -> None:
    bucket_name, object_name = split_gcs_uri(_done_marker_uri(type_, sha))
    client = Client()
    client.bucket(bucket_name).blob(object_name).upload_from_string(b"")


def stage_file(
    local_path: str, type_: str, *, extra_files: list[str] | None = None
) -> tuple[str, str]:
    """Stage a single file. Returns ``(sha, container_path)``.

    ``extra_files`` are uploaded alongside the primary file under the same
    content-addressed prefix without contributing to the hash — used for
    deterministic siblings like ``hg38.fa.fai`` that pysam expects next to
    the FASTA but can't write to a read-only GCSFuse mount.

    No-op if the cache already holds this hash under this basename. Both
    conditions matter: the prefix is content-addressed but the container path
    carries the local basename, so content previously staged from a file that
    has since been renamed lives beside a ``.DONE`` marker under the old name,
    and the job would be handed a path that was never uploaded.
    """
    sha = hash_file(local_path)
    basename = os.path.basename(local_path.rstrip("/"))
    container_path = _container_path_file(type_, sha, basename)
    dest = f"{cache_prefix()}/{type_}/{_key(sha)}/{basename}"
    if gcs_file_exist(_done_marker_uri(type_, sha)) and gcs_file_exist(dest):
        print(f"[stage] {type_}/{_key(sha)} (cached)")
        return sha, container_path
    print(f"[stage] {type_}/{_key(sha)} uploading {local_path} → {dest}")
    t0 = time.time()
    _upload_blob(local_path, dest)
    for extra in extra_files or []:
        extra_dest = f"{cache_prefix()}/{type_}/{_key(sha)}/{os.path.basename(extra)}"
        print(f"[stage] {type_}/{_key(sha)} uploading sibling {extra}")
        _upload_blob(extra, extra_dest)
    _write_done_marker(type_, sha)
    print(f"[stage] {type_}/{_key(sha)} uploaded in {time.time() - t0:.1f}s")
    return sha, container_path


def stage_dir(
    local_path: str, type_: str, *, filename: str | None = None
) -> tuple[str, str]:
    """Stage a directory tree. Returns ``(sha, container_path)``.

    Walks ``local_path``, uploads each file to ``<cache>/<type>/<sha>/<relpath>``,
    and writes a ``.DONE`` marker last. If ``filename`` is set, only files
    whose basename matches are hashed and uploaded — used for model trees
    that bundle unrelated analyses alongside the weight files.
    """
    print(f"[stage] hashing {local_path}" + (f" ({filename} only)" if filename else ""))
    sha = hash_dir(local_path, filename=filename)
    container_path = _container_path_dir(type_, sha)
    if gcs_file_exist(_done_marker_uri(type_, sha)):
        print(f"[stage] {type_}/{_key(sha)} (cached)")
        return sha, container_path

    files = list(_iter_dir_files(local_path, filename=filename))
    print(f"[stage] {type_}/{_key(sha)} uploading {len(files)} files from {local_path}")
    t0 = time.time()
    dest_prefix = f"{cache_prefix()}/{type_}/{_key(sha)}"
    for rel, full in files:
        _upload_blob(full, f"{dest_prefix}/{rel}")
    _write_done_marker(type_, sha)
    print(f"[stage] {type_}/{_key(sha)} uploaded in {time.time() - t0:.1f}s")
    return sha, container_path


# ---------------------------------------------------------------------------
# resume support
# ---------------------------------------------------------------------------


def detect_model_folds_gcs(gcs_models_dir: str, cross: int = 0) -> int:
    """GCS analogue of ``utils.detect_model_folds``.

    Counts sequential folds by checking for ``model_best.pth`` under
    ``<gcs_models_dir>/f{fold}c{cross}/train/`` in GCS. Used by marker mode,
    where the local models tree has no ``.pth`` files (the train mirror excludes
    them) so the local detector would always return 0.
    """
    fold = 0
    while gcs_file_exist(
        f"{gcs_models_dir.rstrip('/')}/f{fold}c{cross}/train/model_best.pth"
    ):
        fold += 1
    return fold


def check_progress_h5_gcs(gcs_uri: str, expected_status: str = "completed") -> bool:
    """GCS-aware variant of multi.check_progress_h5.

    Downloads the candidate scores.h5 to a tempfile and delegates to
    ``check_progress_h5``. Returns False if the blob doesn't exist or the
    file's progress_status doesn't match. Used by the GCP backend to skip
    shards that already finished in a previous attempt.
    """
    if not gcs_file_exist(gcs_uri):
        return False
    import tempfile

    from baskerville.helpers.gcs_utils import download_from_gcs
    from baskerville.multi import check_progress_h5

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        download_from_gcs(gcs_uri, tmp_path, bytes=True)
        return check_progress_h5(tmp_path, expected_status)
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# run marker (GCS origin of a locally-mirrored fold run)
# ---------------------------------------------------------------------------


def write_run_marker(out_dir: str, **fields) -> str:
    """Write ``{out_dir}/gcp_run.json`` recording a GCP run's GCS origin/config.

    Called by the training orchestrator so a later command (eval) can discover
    where the weights live in GCS and inherit the run's GCP settings. Returns
    the marker path. Creates ``out_dir`` if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RUN_MARKER_NAME)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2, sort_keys=True)
    return path


def read_run_marker(out_dir: str) -> dict | None:
    """Read ``{out_dir}/gcp_run.json`` if present, else None."""
    if not out_dir:
        return None
    path = os.path.join(out_dir, RUN_MARKER_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def apply_run_marker(args, marker: dict) -> list[str]:
    """Fill unset GCP args from a run marker; return the inherited field names.

    Lets a downstream command (eval) inherit the GCP config of the run that
    produced its models, so the user needn't re-specify project/region/etc.
    Explicit CLI flags always win: every field defaults to ``None`` in argparse,
    so a field is inherited only while still ``None`` — a value present on
    ``args`` was passed explicitly and is left untouched.
    """
    inherited = []
    for field in ("gcp_data_dir", "gcp_image", "gcp_project", "gcp_region", "gcp_zone"):
        if getattr(args, field, None) is None and marker.get(field) is not None:
            setattr(args, field, marker[field])
            inherited.append(field)
    return inherited


# ---------------------------------------------------------------------------
# run id
# ---------------------------------------------------------------------------


def build_run_id(*hashes: str, deterministic: bool = False) -> str:
    """Run id from input hashes.

    Default form: ``<YYYY-MM-DDTHHMMSS>-<sha8>-...`` — per-invocation unique.
    With ``deterministic=True``: ``<sha8>-...`` — same inputs always produce
    the same id, so a re-run lands in the same output prefix and can detect
    which per-shard outputs from the previous attempt are already complete.
    """
    short = "-".join(h[:8] for h in hashes if h)
    if deterministic:
        return short or "run"
    ts = time.strftime("%Y-%m-%dT%H%M%S")
    return f"{ts}-{short}" if short else ts
