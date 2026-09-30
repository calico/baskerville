"""Small helper to add the gcprunner-specific CLI flags to fold scripts."""

from __future__ import annotations

import argparse
import importlib.util
import os

from .batch_spec import PROVISIONING

# Default used when neither the flag nor the corresponding env var is set.
# There is deliberately no default project: set --gcp_project or
# GCPRUNNER_PROJECT to your own (see README "GCP configuration").
DEFAULT_GCP_REGION = "us-central1"


def resolve_region(region: str | None) -> str:
    """Effective Batch region: explicit value, else GCPRUNNER_REGION, else default.

    Single source of truth for the region default so ``--gcp_region`` can stay
    ``None`` in argparse (letting GCPRUNNER_REGION apply) while every consumer —
    Job construction, run markers, the resubmit/conclude loops, status prints —
    still sees the same resolved value.
    """
    # `or` chaining (not get's default) so an empty --gcp_region or an empty
    # GCPRUNNER_REGION falls through to the default rather than yielding "".
    return region or os.environ.get("GCPRUNNER_REGION") or DEFAULT_GCP_REGION


def resolve_project(project: str | None) -> str:
    """Effective GCP project: explicit value, else GCPRUNNER_PROJECT; raise if neither."""
    resolved = project or os.environ.get("GCPRUNNER_PROJECT")
    if not resolved:
        raise ValueError(
            "GCP project not set: pass --gcp_project (project=...) or set "
            'GCPRUNNER_PROJECT. See README "GCP configuration".'
        )
    return resolved


def default_image_base(project: str | None, region: str | None) -> str:
    """Registry base used when GCPRUNNER_IMAGE_BASE is unset: an Artifact Registry
    repo named ``baskerville`` in the compute project and region."""
    return (
        f"{resolve_region(region)}-docker.pkg.dev/{resolve_project(project)}"
        "/baskerville/baskerville"
    )


def resolve_zone(zone: str | None) -> str | None:
    """Effective zone: explicit value, else GCPRUNNER_ZONE, else None (region-wide)."""
    return zone or os.environ.get("GCPRUNNER_ZONE") or None


def gcp_location_str(args) -> str:
    """Human-readable job location: an explicit zone, else the resolved region."""
    zone = resolve_zone(args.gcp_zone)
    if zone:
        return zone
    return f"{resolve_region(args.gcp_region)} (region-wide)"


def resolve_image_arg(
    args, *, allow_default: bool = True, marker: dict | None = None
) -> str | None:
    """Resolve ``--gcp_branch`` (or the 'main' default) into ``args.gcp_image``.

    Mutates ``args`` in place so every downstream consumer (per-job spec, run
    marker, eval inheritance) sees one concrete, digest-pinned image URI, and the
    git/gcloud lookup happens exactly once. Precedence:

        explicit --gcp_image > explicit --gcp_branch > GCPRUNNER_IMAGE env
        > (allow_default) latest built commit on 'main'

    ``allow_default=False`` skips the 'main' fallback so a downstream command
    (eval/ISM) can resolve an *explicit* branch before ``apply_run_marker`` (to
    beat marker inheritance) yet let the marker's image win over the default.
    ``marker`` supplies the run's project/region for the registry lookup when
    the flags are unset (the marker itself is applied to ``args`` later).
    Returns the resolved image, or None when nothing was resolved here.
    """
    if getattr(args, "gcp_image", None):
        return args.gcp_image  # explicit URI/tag wins, untouched
    branch = getattr(args, "gcp_branch", None)
    if not branch:
        if os.environ.get("GCPRUNNER_IMAGE"):
            return None  # env pin — let job.resolve_image handle it
        if not allow_default:
            return None
        from .image_branch import DEFAULT_BRANCH

        branch = DEFAULT_BRANCH
    from .image_branch import resolve_branch_image

    args.gcp_image = resolve_branch_image(
        branch,
        project=getattr(args, "gcp_project", None) or (marker or {}).get("gcp_project"),
        region=getattr(args, "gcp_region", None) or (marker or {}).get("gcp_region"),
    )
    return args.gcp_image


def announce_gcp_image(args, marker: dict | None = None) -> None:
    """Apply the 'main' default, print the resolved image, and warn on override.

    The eval/ISM scripts call this *after* ``apply_run_marker`` so a training
    image inherited from the marker wins over the default. Warns when an explicit
    ``--gcp_branch`` overrode that inherited image (eval then runs on a different
    image than training).
    """
    resolved = resolve_image_arg(args)
    if not resolved:
        return
    print(f"[gcp] image: {resolved}")
    inherited = marker.get("gcp_image") if marker else None
    if getattr(args, "gcp_branch", None) and inherited and inherited != resolved:
        print(
            f"[gcp] WARNING: --gcp_branch overrides the training image "
            f"({inherited}); this run will use a different image than training."
        )


def add_argparse_group(parser: argparse.ArgumentParser) -> None:
    """Add a ``--backend`` flag plus GCP-specific knobs to ``parser``.

    Used by every ``hound_*_folds.py`` script: when ``--backend gcp`` is
    selected, the additional flags determine the project/region/image and
    cost knobs. The default is ``slurm`` when slurmrunner is installed, else
    ``local``.
    """
    g = parser.add_argument_group("runner backend")
    g.add_argument(
        "--backend",
        choices=("local", "slurm", "gcp"),
        default="slurm" if importlib.util.find_spec("slurmrunner") else "local",
        help="Job runner: run in-process locally, submit to Slurm, or "
        "submit to GCP Batch.",
    )
    g.add_argument(
        "--gcp_project",
        default=None,
        help="GCP project ID (required for --backend gcp, or set GCPRUNNER_PROJECT).",
    )
    g.add_argument(
        "--gcp_region",
        default=None,
        help="Batch region (defaults to us-central1, or set GCPRUNNER_REGION).",
    )
    g.add_argument(
        "--gcp_zone",
        default=None,
        help="Pin the job to a single zone (e.g. us-central1-a) instead of the "
        "whole region. Use to co-locate the VM with a single-zone Rapid Cache; "
        "forfeits cross-zone GPU fallback. Omit when the cache covers every zone "
        "in the region.",
    )
    g.add_argument(
        "--gcp_image",
        default=None,
        help="Container image URI, or a bare tag expanded against the registry "
        "base (set GCPRUNNER_IMAGE_BASE when the registry lives in a different "
        "project than --gcp_project). A purely-numeric tag is prefixed with "
        "GCPRUNNER_IMAGE_TAG_PREFIX if set. Overrides "
        "--gcp_branch. When neither is set, defaults to the latest built commit "
        "on 'main' (or set GCPRUNNER_IMAGE).",
    )
    g.add_argument(
        "--gcp_branch",
        default=None,
        help="Resolve the image to the newest commit on this git branch that has "
        "a built image in the registry (correlated by commit-<sha>; there is no "
        "per-branch tag). Ignored if --gcp_image is given. When neither is set, "
        "defaults to 'main'.",
    )
    g.add_argument(
        "--gcp_output_dir",
        default=None,
        help="gs:// directory; per-job local outputs are synced here on exit.",
    )
    g.add_argument(
        "--gcp_data_dir",
        default=None,
        help="gs:// directory mounted read-only in the container (GCSFuse).",
    )
    g.add_argument(
        "--gcp_data_local",
        default="/workspace/data",
        help="Path inside the container where --gcp_data_dir is mounted.",
    )
    g.add_argument(
        "--gcp_stage_dir",
        default=None,
        help="gs:// directory rsync'd to local SSD before the job runs (training).",
    )
    g.add_argument(
        "--gcp_stage_local",
        default="/workspace/stage",
        help="Local path for --gcp_stage_dir.",
    )
    g.add_argument(
        "--gcp_provisioning",
        choices=PROVISIONING,
        default="standard",
        help="VM purchase option. standard = on-demand; spot = reclaimable at "
        "~30s notice for ~40%% off; flex_start = queue for capacity, then run "
        "uninterrupted for up to 7 days at the deepest discount (recommended "
        "for H100/A3) [Default: %(default)s].",
    )
    g.add_argument(
        "--gcp_retry",
        type=int,
        default=None,
        help="Batch task retries for ANY failure (crash/hardware/preemption); "
        "each runs on a fresh VM. Overrides the per-backend default (Spot jobs "
        "retry a few times, on-demand none); 0 disables.",
    )
    g.add_argument(
        "--gcp_service_account",
        default=None,
        help="Service account email for the Batch VMs.",
    )
    g.add_argument(
        "--gcp_fetch_output",
        default=None,
        help=(
            "Local directory to mirror merged per-fold results to (skips per-job "
            "intermediates). Unset = use the script's local output dir; empty "
            "string = skip local fetch."
        ),
    )


class _PartialRunner:
    """Slurmrunner-shaped facade that pre-applies GCP-only kwargs to ``Job``.

    Lets fold scripts keep their existing ``runner.Job(...)`` /
    ``runner.multi_run(...)`` call sites with no further changes.
    """

    def __init__(self, mod, extra_job_kwargs: dict):
        self._mod = mod
        self._extra = extra_job_kwargs

    def Job(self, *args, **kwargs):
        return self._mod.Job(*args, **{**self._extra, **kwargs})

    def __getattr__(self, name):
        return getattr(self._mod, name)


def make_runner(args, *, slurm_module=None):
    """Return a runner facade matching the slurmrunner public API.

    For ``--backend gcp`` the returned object's ``Job`` constructor has the
    GCP-specific kwargs (project, region, image, provisioning, mounts, …) already
    bound, so existing call sites that pass the slurmrunner kwargs work
    unchanged. For ``--backend slurm`` returns the slurmrunner module
    directly (or whatever was passed as ``slurm_module``, which may be None
    if slurmrunner is not installed; callers already handle that). For
    ``--backend local`` returns ``None``, which callers treat as "run jobs
    in-process locally" — the same path as a missing slurmrunner.
    """
    if args.backend == "local":
        return None
    if args.backend == "gcp":
        # Backstop: resolve --gcp_branch / the 'main' default for any script that
        # didn't do it earlier. Idempotent — a no-op once args.gcp_image is set,
        # so scripts that resolve before apply_run_marker (train/eval/ISM) keep
        # their marker precedence and this only fills the gap for the rest.
        if not getattr(args, "gcp_image", None):
            resolved = resolve_image_arg(args)
            if resolved:
                print(f"[gcp] image: {resolved}")

        import importlib

        mod = importlib.import_module("gcprunner")
        return _PartialRunner(mod, gcp_job_kwargs(args))
    if args.backend == "slurm":
        if slurm_module is not None:
            return slurm_module
        try:
            import slurmrunner

            return slurmrunner
        except ImportError:
            print("[slurm] slurmrunner not installed, running locally")
            return None
    raise ValueError(f"unknown backend {args.backend!r}")


def gcp_job_kwargs(args) -> dict:
    """Translate the parsed argparse namespace into kwargs for ``gcprunner.Job``."""
    kwargs: dict = {
        "project": args.gcp_project,
        "region": args.gcp_region,
        "image": args.gcp_image,
        "provisioning": args.gcp_provisioning,
    }
    if getattr(args, "gcp_zone", None):
        kwargs["zone"] = args.gcp_zone
    if getattr(args, "gcp_retry", None) is not None:
        kwargs["retry_count"] = args.gcp_retry
    if args.gcp_output_dir:
        kwargs["output_dir_gcs"] = args.gcp_output_dir
    if args.gcp_service_account:
        kwargs["service_account"] = args.gcp_service_account

    from .batch_spec import DataMount

    mounts: list[DataMount] = []
    if args.gcp_data_dir:
        mounts.append(DataMount(args.gcp_data_dir, args.gcp_data_local, mode="fuse"))
    if args.gcp_stage_dir:
        mounts.append(
            DataMount(args.gcp_stage_dir, args.gcp_stage_local, mode="stage_local")
        )
    if mounts:
        kwargs["data_mounts"] = mounts
    return kwargs
