"""Multi-job runner — slurmrunner.runner-shaped API for GCP Batch."""

from __future__ import annotations

import logging
import sys
import time as _time
from typing import List, Optional

from .job import Job

logger = logging.getLogger(__name__)


def multi_run(
    jobs: List[Job],
    max_proc: Optional[int] = None,
    verbose: bool = False,
    launch_sleep: int = 2,
    update_sleep: int = 30,
    *,
    raise_on_failure: bool = True,
) -> None:
    """Launch and poll a list of jobs, capping concurrency at ``max_proc``.

    Mirrors ``slurmrunner.multi_run``: jobs run independently, no dependency
    graph. Batch already reaped each VM, so there's nothing to clean up here.
    A Spot job that still fails after its Batch retries is resubmitted once as
    standard.

    Raises ``RuntimeError``, naming the jobs, if any fails to launch or reaches
    a terminal state other than ``COMPLETED`` (Batch already applied its in-task
    retries by then). This aborts the caller before a downstream step — e.g.
    ``finalize_fold`` — trips over the missing output. Training passes
    ``raise_on_failure=False`` to decide whether to resume from GCS checkpoints.
    Submission errors always raise after the other jobs finish.
    """
    if not jobs:
        logger.info("No jobs to run")
        return

    total = len(jobs)
    next_job = 0
    active: List[Job] = []
    failed: List[str] = []
    submission_errors: List[str] = []
    if max_proc is None:
        max_proc = total
    if max_proc < 1:
        raise ValueError("max_proc must be positive")

    logger.info("multi_run: %d jobs, max concurrent %d", total, max_proc)

    while next_job < total or active:
        while next_job < total and len(active) < max_proc:
            current = jobs[next_job]
            next_job += 1
            try:
                current.launch()
            except Exception as e:
                logger.error("Failed to launch %s: %s", current.name, e)
                current.status = "FAILED"
                failed.append(current.name)
                submission_errors.append(f"{current.name}: {e}")
                continue
            active.append(current)
            _time.sleep(launch_sleep)
            if verbose:
                print(f"Launched: {current.name} (id: {current.short_id})")
                print(current.cmd, file=sys.stderr)

        if active:
            _time.sleep(update_sleep)
            multi_update_status(active)
            still: List[Job] = []
            for job in active:
                if job.status in {"PENDING", "RUNNING"}:
                    still.append(job)
                    continue
                if job.status == "FAILED" and job.spec.provisioning == "spot":
                    # Spot retries exhausted (typically a preemption storm):
                    # resubmit once on-demand, which Batch never reclaims.
                    logger.warning(
                        "Job %s failed on Spot; resubmitting as standard", job.name
                    )
                    job.spec.provisioning = "standard"
                    try:
                        job.launch()
                    except Exception as e:
                        logger.error("Failed to relaunch %s: %s", job.name, e)
                        job.status = "FAILED"
                        failed.append(job.name)
                        submission_errors.append(f"{job.name}: {e}")
                        continue
                    if verbose:
                        print(
                            f"Relaunched as standard: {job.name} (id: {job.short_id})"
                        )
                    still.append(job)
                    continue
                if verbose:
                    print(f"Completed: {job.name} - {job.status}")
                logger.info("Job %s -> %s", job.name, job.status)
                if job.status != "COMPLETED":
                    failed.append(job.name)
            active = still

    summary = get_job_summary(jobs)
    logger.info("All %d jobs done: %s", total, summary)
    if verbose:
        print(f"[multi_run] {total} jobs done: {summary}")
    if submission_errors or (failed and raise_on_failure):
        message = f"{len(failed)} job(s) failed: {', '.join(sorted(failed))}"
        if submission_errors:
            message += ". Submission errors: " + "; ".join(submission_errors)
        raise RuntimeError(message)


def multi_update_status(
    jobs: List[Job], max_attempts: int = 3, sleep_attempt: int = 5
) -> None:
    """Refresh status for many jobs.

    Batch has no batched status RPC, so this just calls each job's
    ``update_status`` in turn. Kept as a separate function to match
    slurmrunner's API.
    """
    for j in jobs:
        j.update_status(max_attempts=max_attempts, sleep_attempt=sleep_attempt)


def get_job_summary(jobs: List[Job]) -> dict:
    summary: dict = {}
    for j in jobs:
        s = j.status or "UNKNOWN"
        summary[s] = summary.get(s, 0) + 1
    return summary


def list_active_jobs(project: str, region: str = "us-central1") -> List[dict]:
    """Cost-watchdog helper: list non-terminal Batch jobs in the project/region.

    Returns a list of ``{"name": str, "state": str}`` dicts. Use this to
    confirm nothing is left running and billing.
    """
    from google.cloud import batch_v1

    client = batch_v1.BatchServiceClient()
    parent = f"projects/{project}/locations/{region}"
    out: List[dict] = []
    for job in client.list_jobs(request=batch_v1.ListJobsRequest(parent=parent)):
        state_name = batch_v1.JobStatus.State(job.status.state).name
        if state_name in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            continue
        out.append({"name": job.name, "state": state_name, "labels": dict(job.labels)})
    return out


def cancel_active_jobs(
    project: str,
    region: str,
    name_substr: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
) -> List[str]:
    """Cancel (SIGTERM, not delete) active Batch jobs matching a name and/or labels.

    A job is cancelled iff it matches ``name_substr`` (when given) AND every
    key/value in ``labels`` (when given). At least one filter must be provided —
    a bare call would otherwise cancel every active job. Label values are
    sanitized to GCP's label rules before comparison, so callers can pass raw
    identifiers (e.g. a run_id).

    Cancel is preferred over delete: it sends the container a SIGTERM, so the
    entrypoint's EXIT trap runs and uploads the final partial output + logs to
    GCS before the VM is reaped. The job resource stays around (state CANCELLED)
    for inspection. Idempotent — already-terminal jobs are skipped, so re-running
    is safe. Returns the resource names of the jobs a cancel was requested for.
    """
    if name_substr is None and not labels:
        raise ValueError("cancel_active_jobs requires name_substr and/or labels")
    from google.cloud import batch_v1

    from .batch_spec import _label_safe

    want_labels = {k: _label_safe(v) for k, v in (labels or {}).items()}
    client = batch_v1.BatchServiceClient()
    parent = f"projects/{project}/locations/{region}"
    cancelled: List[str] = []
    for job in client.list_jobs(request=batch_v1.ListJobsRequest(parent=parent)):
        state_name = batch_v1.JobStatus.State(job.status.state).name
        if state_name in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            continue
        if name_substr is not None and name_substr not in job.name:
            continue
        if want_labels:
            job_labels = dict(job.labels)
            if any(job_labels.get(k) != v for k, v in want_labels.items()):
                continue
        try:
            client.cancel_job(request=batch_v1.CancelJobRequest(name=job.name))
            cancelled.append(job.name)
        except Exception as e:  # pragma: no cover - network/SDK error path
            logger.warning("cancel_job failed for %s: %s", job.name, e)
    return cancelled
