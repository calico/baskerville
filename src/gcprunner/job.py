"""Job class for managing individual GCP Batch jobs.

Public API is intentionally a superset of slurmrunner.Job — the slurmrunner
keyword args (cmd, name, out_file, err_file, sb_file, queue, cpu, mem, time,
gpu) are all accepted so call sites need not change.
"""

from __future__ import annotations

import logging
import os
import re
import time as _time
import uuid
from typing import Optional

from .argparse_helpers import (
    default_image_base,
    resolve_project,
    resolve_region,
    resolve_zone,
)
from .batch_spec import DataMount, JobSpec, build_batch_job_dict
from .gpus import resolve_gpu

logger = logging.getLogger(__name__)

# Prefix prepended to a purely-numeric image tag (a CI build number) so callers
# can pass just the number. This is the fixed convention for this repo's images;
# override per-call via image_tag_prefix= or GCPRUNNER_IMAGE_TAG_PREFIX.
DEFAULT_IMAGE_TAG_PREFIX = ""


def resolve_image(
    image: Optional[str],
    *,
    region: str,
    project: str,
    image_base: Optional[str] = None,
    image_tag_prefix: Optional[str] = None,
) -> str:
    """Resolve an image reference to a full Artifact Registry URI.

    A value containing ``/`` is treated as a complete URI and returned verbatim.
    A bare tag is expanded against the registry base (``image_base`` /
    ``GCPRUNNER_IMAGE_BASE``, else ``<region>-docker.pkg.dev/<project>/
    baskerville/baskerville``). A purely-numeric tag is first prefixed
    (``image_tag_prefix`` / ``GCPRUNNER_IMAGE_TAG_PREFIX`` /
    ``DEFAULT_IMAGE_TAG_PREFIX``); pass ``image_tag_prefix=""`` or set the env
    var empty to disable that. Shared by Job and the run-marker writer so both
    record the same effective image.
    """
    raw_image = image or os.environ.get("GCPRUNNER_IMAGE") or "latest"
    if "/" in raw_image:
        return raw_image

    tag = raw_image
    # `is not None` (not `or`) so an explicit "" disables the prefix rather than
    # falling through to the env/default value.
    prefix = (
        image_tag_prefix
        if image_tag_prefix is not None
        else os.environ.get("GCPRUNNER_IMAGE_TAG_PREFIX", DEFAULT_IMAGE_TAG_PREFIX)
    )
    if prefix and tag.isdigit():
        tag = f"{prefix}{tag}"

    base = (
        image_base
        or os.environ.get("GCPRUNNER_IMAGE_BASE")
        or default_image_base(project, region)
    )
    return f"{base.rstrip('/:')}:{tag}"


# Map Batch state strings → slurmrunner-style strings so callers don't branch.
_STATE_MAP = {
    "STATE_UNSPECIFIED": "PENDING",
    "QUEUED": "PENDING",
    "SCHEDULED": "PENDING",
    "RUNNING": "RUNNING",
    "SUCCEEDED": "COMPLETED",
    "FAILED": "FAILED",
    "DELETION_IN_PROGRESS": "RUNNING",
    "CANCELLATION_IN_PROGRESS": "RUNNING",
    "CANCELLED": "CANCELLED",
}


def _parse_slurm_time(t: Optional[str]) -> Optional[int]:
    """Parse slurmrunner's ``D-HH:MM:SS`` (or ``HH:MM:SS``) into seconds."""
    if t is None:
        return None
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = (int(x) for x in parts)
    elif len(parts) == 2:
        h = 0
        m, s = (int(x) for x in parts)
    else:
        raise ValueError(f"cannot parse time string {t!r}")
    return days * 86400 + h * 3600 + m * 60 + s


def _job_id_safe(name: str) -> str:
    """Batch job IDs: lowercase letters, digits, dashes; <=63 chars; start letter."""
    base = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    if not base or not base[0].isalpha():
        base = "j-" + base
    base = base[:50] or "job"
    return f"{base}-{uuid.uuid4().hex[:8]}"


class Job:
    """A single GCP Batch job.

    Slurmrunner-compatible kwargs (``queue``, ``mem`` in MB, ``time`` as
    ``D-HH:MM:SS``, ``gpu`` count) are all accepted. Cloud-specific knobs are
    available as additional kwargs.

    Args:
        cmd: Shell command to run inside the container.
        name: Human-readable job name.
        out_file: Optional gs:// (or local) path for stdout copy.
        err_file: Optional gs:// (or local) path for stderr copy.
        sb_file: Ignored; accepted for slurmrunner API parity.
        queue: GPU alias (e.g. ``"l4"``, ``"a100-80"``, ``"rtx4090"``).
        cpu: vCPU count.
        mem: Memory in MB.
        time: Max runtime, slurm format (``"7-0:0:0"``).
        gpu: GPU count override (defaults to the queue profile's count).
        image: Container image URI, or a bare tag expanded against the registry
            base (``image_base`` / ``GCPRUNNER_IMAGE_BASE``). Falls back to
            ``GCPRUNNER_IMAGE`` when unset.
        image_base: Registry path prefix (host/project/repo/name, no tag) used
            to expand a bare ``image`` tag. Defaults to ``GCPRUNNER_IMAGE_BASE``;
            when unset, falls back to ``<region>-docker.pkg.dev/<project>/
            baskerville/baskerville`` (registry == compute project).
        image_tag_prefix: Prefix prepended to a purely-numeric ``image`` tag so
            e.g. ``29162067376`` → ``build-29162067376`` with prefix
            ``"build-"``. Defaults to ``GCPRUNNER_IMAGE_TAG_PREFIX``, else no
            prefix. Non-numeric tags and the ``latest`` default are left
            unchanged.
        provisioning: VM purchase option — ``"standard"`` (on-demand, default),
            ``"spot"``, or ``"flex_start"``. See ``batch_spec.PROVISIONING``.
        region: Batch region (e.g. ``us-central1``).
        project: GCP project ID. Defaults to ``GCPRUNNER_PROJECT``; raises if neither is set.
        data_mounts: Iterable of ``DataMount`` for GCSFuse / staged data.
        env: Extra environment variables.
        output_dir_gcs: Optional gs:// dir; container's local output dir is
            synced here on exit (success or crash).
        service_account, network, subnetwork: Compute Engine knobs.
        boot_disk_gb: VM boot disk size in GiB (default 100). Bump for jobs that
            write large temporary files to the container filesystem (e.g. the
            spec eval preds/targets Zarr store at small --step).
        labels: Extra Batch labels for run identity / filtering (e.g.
            ``{"gcprunner_run": run_id}``). Merged with the auto ``gcprunner_name``
            label; keys/values are sanitized to GCP's label rules at build time.
    """

    def __init__(
        self,
        cmd: str,
        name: str,
        out_file: Optional[str] = None,
        err_file: Optional[str] = None,
        sb_file: Optional[str] = None,  # noqa: ARG002 — slurmrunner parity
        queue: str = "l4",
        cpu: int = 1,
        mem: Optional[int] = None,
        time: Optional[str] = None,
        gpu: int = 0,
        *,
        image: Optional[str] = None,
        image_base: Optional[str] = None,
        image_tag_prefix: Optional[str] = None,
        provisioning: str = "standard",
        region: Optional[str] = None,
        zone: Optional[str] = None,
        retry_count: Optional[int] = None,
        project: Optional[str] = None,
        data_mounts: Optional[list[DataMount]] = None,
        env: Optional[dict[str, str]] = None,
        output_dir_gcs: Optional[str] = None,
        output_dir_local: str = "/workspace/out",
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        subnetwork: Optional[str] = None,
        boot_disk_gb: int = 100,
        labels: Optional[dict[str, str]] = None,
    ):
        self.cmd = cmd
        self.name = name
        self.out_file = out_file
        self.err_file = err_file

        profile = resolve_gpu(queue)
        if gpu > 0 and profile.accelerator_count and gpu != profile.accelerator_count:
            logger.warning(
                "Job %s requested gpu=%d but queue %r profile is %d-GPU; using profile count",
                name,
                gpu,
                queue,
                profile.accelerator_count,
            )

        self.project = resolve_project(project)
        resolved_region = resolve_region(region)
        resolved_zone = resolve_zone(zone)
        if resolved_zone and not resolved_zone.startswith(f"{resolved_region}-"):
            raise ValueError(
                f"zone {resolved_zone!r} is not within region {resolved_region!r} "
                f"(expected a '{resolved_region}-<x>' zone)."
            )

        resolved_image = resolve_image(
            image,
            region=resolved_region,
            project=self.project,
            image_base=image_base,
            image_tag_prefix=image_tag_prefix,
        )

        self.spec = JobSpec(
            name=name,
            cmd=cmd,
            image=resolved_image,
            profile=profile,
            cpu=cpu,
            mem_mb=mem,
            max_runtime_seconds=_parse_slurm_time(time),
            provisioning=provisioning,
            region=resolved_region,
            zone=resolved_zone,
            retry_count=retry_count,
            data_mounts=list(data_mounts or []),
            env=dict(env or {}),
            out_gcs=out_file if (out_file and out_file.startswith("gs://")) else None,
            err_gcs=err_file if (err_file and err_file.startswith("gs://")) else None,
            output_dir_gcs=output_dir_gcs,
            output_dir_local=output_dir_local,
            service_account=service_account,
            network=network,
            subnetwork=subnetwork,
            boot_disk_gb=boot_disk_gb,
            labels=dict(labels or {}),
        )

        # Resolved at launch time
        self.id: Optional[str] = None  # full job resource name
        self.short_id: Optional[str] = None  # just the job id portion
        self.status: Optional[str] = None
        self._client = None

    @property
    def queue(self) -> str:
        return self.spec.profile.machine_type

    @property
    def gpu(self) -> int:
        return self.spec.profile.accelerator_count

    def to_batch_dict(self) -> dict:
        """Return the Batch CreateJobRequest payload as a plain dict."""
        if not self.spec.image:
            raise ValueError(
                f"Job {self.name}: container image not set. "
                "Pass image=... or set GCPRUNNER_IMAGE."
            )
        return build_batch_job_dict(self.spec)

    def _get_client(self):
        if self._client is None:
            from google.cloud import batch_v1

            self._client = batch_v1.BatchServiceClient()
        return self._client

    def launch(self) -> None:
        """Submit the job to GCP Batch."""
        from google.cloud import batch_v1

        job_dict = self.to_batch_dict()
        job_proto = batch_v1.Job(job_dict)

        short = _job_id_safe(self.name)
        request = batch_v1.CreateJobRequest(
            parent=f"projects/{self.project}/locations/{self.spec.region}",
            job=job_proto,
            job_id=short,
        )
        client = self._get_client()
        try:
            created = client.create_job(request=request)
        except Exception as e:
            logger.error("Failed to launch job %s: %s", self.name, e)
            raise

        self.id = created.name
        self.short_id = short
        self.status = "PENDING"
        logger.info("Launched job %s as %s", self.name, self.id)

    def update_status(self, max_attempts: int = 3, sleep_attempt: int = 5) -> bool:
        """Refresh ``self.status`` from the Batch API.

        Returns True if status was found, False otherwise.
        """
        if self.id is None:
            return False
        from google.cloud import batch_v1

        client = self._get_client()
        for attempt in range(max_attempts):
            if attempt > 0:
                _time.sleep(sleep_attempt)
            try:
                got = client.get_job(request=batch_v1.GetJobRequest(name=self.id))
                state_name = batch_v1.JobStatus.State(got.status.state).name
                self.status = _STATE_MAP.get(state_name, state_name)
                return True
            except Exception as e:
                logger.warning(
                    "get_job failed for %s (attempt %d): %s", self.name, attempt + 1, e
                )
        return False

    def clean(self) -> None:
        """Delete the Batch job resource (no-op if not launched)."""
        if self.id is None:
            return
        from google.cloud import batch_v1

        try:
            self._get_client().delete_job(
                request=batch_v1.DeleteJobRequest(name=self.id)
            )
        except Exception as e:
            logger.warning("delete_job failed for %s: %s", self.name, e)

    def __repr__(self) -> str:
        return f"Job(name={self.name!r}, id={self.short_id!r}, status={self.status!r})"
