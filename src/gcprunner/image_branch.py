"""Resolve a git branch to the newest built image for that branch.

Images are expected to be tagged ``commit-<full-git-sha>`` (alongside any of
``build-<run_id>``, ``latest``, or custom tags) — there is **no per-branch
tag**, and ``latest`` is a moving, cross-branch tag. So "the latest image on
branch X" is defined here as *the newest commit on that branch that has a
``commit-<sha>`` image in the registry*. Not every commit gets built, so the
newest commit and the newest built commit can differ; this picks the newest
built one.

All git + gcloud I/O lives here so ``job.resolve_image`` stays pure and
network-free. The two subprocess wrappers (:func:`_run_git`, :func:`_run_gcloud`)
are the seams tests monkeypatch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Optional

from .argparse_helpers import default_image_base

DEFAULT_BRANCH = "main"

# CI writes the full 40-char GITHUB_SHA, so git rev-list SHAs intersect directly.
_COMMIT_TAG_RE = re.compile(r"^commit-([0-9a-f]{40})$")


class ImageBranchError(RuntimeError):
    """A branch could not be resolved to a built image (bad ref, none built, or
    the git/gcloud query failed). Carries a user-actionable message."""


def _run_git(args: list[str], *, repo_dir: Optional[str] = None) -> str:
    """Run ``git <args>`` and return stdout; raise ImageBranchError on failure.

    Isolated so tests can monkeypatch the git seam without a real repo.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ImageBranchError("git not found on PATH") from e
    if proc.returncode != 0:
        raise ImageBranchError(
            f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.returncode}"
        )
    return proc.stdout


def _run_gcloud(
    args: list[str], *, gcloud_bin: Optional[str] = None, timeout: float = 120.0
) -> str:
    """Run ``gcloud <args>`` and return stdout; raise ImageBranchError on failure.

    Maps missing binary, non-zero exit, and timeout to ImageBranchError with an
    actionable message. Isolated so tests can monkeypatch the registry seam.
    """
    try:
        proc = subprocess.run(
            [gcloud_bin or "gcloud", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise ImageBranchError(
            "gcloud not found on PATH — needed to look up images by branch; "
            "pass --gcp_image to skip the lookup"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ImageBranchError(
            f"gcloud timed out after {timeout:.0f}s querying the registry"
        ) from e
    if proc.returncode != 0:
        raise ImageBranchError(
            f"gcloud {' '.join(args)} failed (check auth/network, or pass "
            f"--gcp_image): {proc.stderr.strip() or proc.returncode}"
        )
    return proc.stdout


def _package_repo_dir() -> Optional[str]:
    """The baskerville checkout git should query, not the caller's CWD.

    Fold scripts launch from a data/params directory, not the code repo, so a
    CWD-relative ``git rev-list`` can't see the branch. With the editable install
    this module lives inside the checkout, so its directory is inside the repo.
    Returns None for a non-editable install (no enclosing repo) — callers then
    fall back to CWD, the prior behavior.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        top = _run_git(["rev-parse", "--show-toplevel"], repo_dir=src_dir).strip()
    except ImageBranchError:
        return None
    return top or None


def branch_commits(
    branch: str, *, repo_dir: Optional[str] = None, max_count: int = 2000
) -> list[str]:
    """Full 40-char SHAs on ``branch``'s own line of development, newest first.

    Uses ``--first-parent`` so the walk stays on the branch and never descends
    into commits merged in from elsewhere. A feature branch that repeatedly
    merges ``main`` has all of main's history in its ancestry; without
    ``--first-parent`` the "newest built commit on the branch" can land on a
    pure-main build that lacks the branch's code (observed: a mamba branch
    resolving to a plain main image). First-parent keeps candidates to the
    commits that actually happened *on* this branch, and is equally correct for
    ``main`` itself (its first-parent line is its PR-merge mainline).

    Tries the ref in order: ``<branch>``, ``refs/remotes/origin/<branch>``,
    ``origin/<branch>`` so a not-checked-out branch still resolves from its
    remote-tracking ref. Raises ImageBranchError (suggesting ``git fetch``) if
    none resolve.
    """
    last_err: Optional[ImageBranchError] = None
    for ref in (branch, f"refs/remotes/origin/{branch}", f"origin/{branch}"):
        try:
            out = _run_git(
                ["rev-list", "--first-parent", "-n", str(max_count), ref],
                repo_dir=repo_dir,
            )
        except ImageBranchError as e:
            last_err = e
            continue
        shas = out.split()
        if shas:
            return shas
    raise ImageBranchError(
        f"could not resolve branch {branch!r} to any commits — is it fetched? "
        f"try `git fetch origin {branch}` (last error: {last_err})"
    )


def registry_commit_index(
    image_base: str, *, gcloud_bin: Optional[str] = None, timeout: float = 120.0
) -> dict[str, str]:
    """Map ``commit-<sha>`` → digest for every built image under ``image_base``.

    One ``gcloud artifacts docker images list --include-tags`` call. The JSON
    ``tags`` field may be a list or a comma-separated string depending on gcloud
    version; both are handled. Returns an empty dict if nothing matches.
    """
    out = _run_gcloud(
        [
            "artifacts",
            "docker",
            "images",
            "list",
            image_base,
            "--include-tags",
            "--format=json",
        ],
        gcloud_bin=gcloud_bin,
        timeout=timeout,
    )
    try:
        entries = json.loads(out or "[]")
    except json.JSONDecodeError as e:
        raise ImageBranchError(f"could not parse gcloud output: {e}") from e

    index: dict[str, str] = {}
    for entry in entries:
        raw_tags = entry.get("tags")
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, list):
            tags = raw_tags
        else:
            continue
        digest = entry.get("version") or ""  # e.g. "sha256:abc..."
        for tag in tags:
            m = _COMMIT_TAG_RE.match(tag)
            if m and digest:
                index[m.group(1)] = digest
    return index


def resolve_branch_image(
    branch: str,
    *,
    image_base: Optional[str] = None,
    project: Optional[str] = None,
    region: Optional[str] = None,
    repo_dir: Optional[str] = None,
    pin_digest: bool = True,
    gcloud_bin: Optional[str] = None,
) -> str:
    """Resolve ``branch`` to the concrete image of its newest built commit.

    Base precedence: ``image_base`` arg, else ``GCPRUNNER_IMAGE_BASE``, else
    :func:`~gcprunner.argparse_helpers.default_image_base` of ``project`` /
    ``region`` (set GCPRUNNER_IMAGE_BASE when the registry lives in a different
    project or region than compute). Returns a digest-pinned URI
    (``<base>@sha256:...``) when ``pin_digest`` (recommended, immune to a
    ``commit-<sha>`` tag being reassigned), otherwise ``<base>:commit-<sha>``.
    Raises ImageBranchError if no commit on the branch has a built image.
    """
    base = (
        image_base
        or os.environ.get("GCPRUNNER_IMAGE_BASE")
        or default_image_base(project, region)
    ).rstrip("/:")
    if repo_dir is None:
        repo_dir = _package_repo_dir()  # query the checkout, not the launch CWD
    # git first (cheap, local): a bad ref fails fast without the registry query.
    commits = branch_commits(branch, repo_dir=repo_dir)
    index = registry_commit_index(base, gcloud_bin=gcloud_bin)
    for sha in commits:
        digest = index.get(sha)
        if digest:
            return f"{base}@{digest}" if pin_digest else f"{base}:commit-{sha}"
    raise ImageBranchError(
        f"no built image for branch {branch!r}: scanned {len(commits)} commit(s), "
        f"none had a commit-<sha> image in {base}. Build and push an image "
        f"tagged commit-<full-sha> for this branch, or pass --gcp_image explicitly."
    )
