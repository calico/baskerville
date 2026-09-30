"""GCP Batch runner with a slurmrunner-shaped API.

Mirrors the public surface of slurmrunner so call sites can switch backend with
a single import shim:

    if args.backend == "gcp":
        import gcprunner as runner
    else:
        import slurmrunner as runner

Then ``runner.Job(...)`` and ``runner.multi_run(...)`` work the same in either
backend.
"""

from .job import Job, resolve_image
from .image_branch import resolve_branch_image, ImageBranchError
from .runner import (
    multi_run,
    multi_update_status,
    get_job_summary,
    list_active_jobs,
    cancel_active_jobs,
)
from .gpus import GPU_PROFILES, resolve_gpu
from . import run_identity

__all__ = [
    "Job",
    "resolve_image",
    "resolve_branch_image",
    "ImageBranchError",
    "multi_run",
    "multi_update_status",
    "get_job_summary",
    "list_active_jobs",
    "cancel_active_jobs",
    "GPU_PROFILES",
    "resolve_gpu",
    "run_identity",
]
