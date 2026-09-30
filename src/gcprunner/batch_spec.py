"""Build GCP Batch job specifications as plain dicts.

Returning a dict (rather than a ``google.cloud.batch_v1`` proto) keeps this
module unit-testable without the SDK installed and makes the spec easy to log
or persist. The Job class converts the dict to the proto types at submit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .gpus import GpuProfile

# Auto retry count for Spot jobs; see build_batch_job_dict for the rationale.
SPOT_DEFAULT_RETRIES = 3

# VM purchase options.
#   standard   — on-demand. Never reclaimed; full price.
#   spot       — reclaimable at ~30s notice, any time. ~40% off.
#   flex_start — Dynamic Workload Scheduler. The request queues until capacity
#                is granted, then the VM is NOT reclaimable for up to 7 days.
#                Cheapest of the three, and the option Google recommends for A3
#                (H100): an A3 job left STANDARD gets the same 7-day cap without
#                the discount. Consumes the project's *preemptible* GPU quota.
PROVISIONING = ("standard", "spot", "flex_start")


@dataclass
class DataMount:
    """Describes how a GCS path should be made available inside the container.

    mode == "fuse": Cloud Storage FUSE read-only mount at ``local_path``.
    mode == "stage_local": rsync the prefix to local SSD at ``local_path``
        before the user command runs (one-shot copy; container is ephemeral).
    """

    gcs_uri: str
    local_path: str
    mode: str = "fuse"  # "fuse" or "stage_local"

    def __post_init__(self):
        if self.mode not in {"fuse", "stage_local"}:
            raise ValueError(f"invalid mount mode {self.mode!r}")
        if not self.gcs_uri.startswith("gs://"):
            raise ValueError(f"data mount gcs_uri must be gs://…, got {self.gcs_uri!r}")


@dataclass
class JobSpec:
    name: str
    cmd: str
    image: str
    profile: GpuProfile
    cpu: int = 1
    mem_mb: Optional[int] = None
    max_runtime_seconds: Optional[int] = None
    provisioning: str = "standard"  # one of PROVISIONING
    region: str = "us-central1"
    # Optional single-zone pin. When set, the job is restricted to this zone
    # (allowed_locations = zones/<zone>) instead of the whole region. Needed to
    # co-locate the VM with a zonal Rapid Cache / dataset; the cost is no
    # cross-zone fallback when that zone is GPU-exhausted.
    zone: Optional[str] = None
    # Batch task retries. None = auto (Spot → SPOT_DEFAULT_RETRIES, on-demand →
    # none); an explicit int (incl. 0 to disable) is honored as-is.
    retry_count: Optional[int] = None
    data_mounts: list[DataMount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    out_gcs: Optional[str] = None  # gs:// path to upload stdout
    err_gcs: Optional[str] = None  # gs:// path to upload stderr
    output_dir_gcs: Optional[str] = None  # gs:// dir to upload run outputs
    output_dir_local: str = "/workspace/out"
    service_account: Optional[str] = None
    network: Optional[str] = None
    subnetwork: Optional[str] = None
    # Boot disk size in GiB. Default 100 covers cuda-devel + cudnn + triton
    # image extractions; Batch's implicit default (~30 GiB) is not enough.
    boot_disk_gb: int = 100
    # Extra Batch labels for run identity / filtering (e.g. gcprunner_run). Merged
    # with the auto gcprunner_name label; keys/values are sanitized at build time.
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.provisioning not in PROVISIONING:
            raise ValueError(
                f"invalid provisioning {self.provisioning!r}; expected one of "
                f"{list(PROVISIONING)}"
            )


def build_batch_job_dict(spec: JobSpec) -> dict:
    """Build a Batch CreateJobRequest payload as a plain dict.

    The structure mirrors the REST/proto schema described at
    https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs#Job
    """
    env = dict(spec.env)
    env["GCPRUNNER_OUTPUT_DIR_LOCAL"] = spec.output_dir_local
    if spec.output_dir_gcs:
        env["GCPRUNNER_OUTPUT_DIR_GCS"] = spec.output_dir_gcs
    if spec.out_gcs:
        env["GCPRUNNER_STDOUT_GCS"] = spec.out_gcs
    if spec.err_gcs:
        env["GCPRUNNER_STDERR_GCS"] = spec.err_gcs

    # Encode mount specs as env vars so entry.sh can stage/mount before
    # invoking the user command. Format: "<mode>\t<gcs_uri>\t<local_path>",
    # one per line. Tab-separated to survive paths with spaces.
    if spec.data_mounts:
        lines = [f"{m.mode}\t{m.gcs_uri}\t{m.local_path}" for m in spec.data_mounts]
        env["GCPRUNNER_DATA_MOUNTS"] = "\n".join(lines)

    # Privileged required for gcsfuse (uses /dev/fuse). Privileged also gives
    # the container access to host /dev/nvidia* device nodes, so on GPU jobs
    # we just bind-mount the host's NVIDIA userspace (libs + nvidia-smi) and
    # entry.sh prepends them to LD_LIBRARY_PATH/PATH. We do NOT use
    # --gpus / --runtime=nvidia: Batch's COS-GPU image is in CDI mode, where
    # both options are rejected by docker.
    # --privileged: gcsfuse (/dev/fuse) + host NVIDIA device nodes (see below).
    # --shm-size: PyTorch DataLoader workers exchange tensors via /dev/shm, and
    # Docker's 64 MB default triggers "Bus error / DataLoader worker killed".
    # Size it to a generous fraction of the VM (it's a tmpfs *limit*, RAM is only
    # consumed as used) so multi-worker loading has room.
    shm_mib = spec.profile.mem_mib // 2
    container: dict = {
        "image_uri": spec.image,
        # entry.sh wraps the user command with stage/mount/upload/log logic.
        "entrypoint": "/usr/local/bin/gcprunner-entry",
        "commands": ["/bin/bash", "-c", spec.cmd],
        "options": f"--privileged --shm-size={shm_mib}m",
    }
    if spec.profile.accelerator_count:
        container["volumes"] = [
            "/var/lib/nvidia/lib64:/usr/local/nvidia/lib64",
            "/var/lib/nvidia/bin:/usr/local/nvidia/bin",
        ]

    cpu_milli = spec.cpu * 1000
    # memoryMib is the per-task memory requirement. Unset, Batch defaults it to
    # 2000 MiB (~1.95 GiB) regardless of machine type — far too little for
    # training and misleading in the console. Every job here is single-task, so
    # default to the profile's usable VM memory; an explicit mem overrides.
    mem_mib = spec.mem_mb if spec.mem_mb is not None else spec.profile.mem_mib
    compute_resource: dict = {"cpu_milli": cpu_milli, "memory_mib": mem_mib}

    runnable = {
        "container": container,
        "environment": {"variables": env},
    }

    task_spec: dict = {
        "runnables": [runnable],
        "compute_resource": compute_resource,
    }
    if spec.max_runtime_seconds is not None:
        task_spec["max_run_duration"] = f"{spec.max_runtime_seconds}s"
    # max_retry_count alone (no lifecycle policy) retries ANY non-zero exit. We
    # don't narrow to the Spot preemption code 50001: a preempted VM can vanish
    # mid-task without surfacing it, and a full retry is safe anyway (eval shards
    # recompute, training restores from checkpoint). A clean exit (incl. loss_nan
    # exiting 0) is never retried.
    if spec.retry_count is not None:
        retries = spec.retry_count
    else:
        retries = SPOT_DEFAULT_RETRIES if spec.provisioning == "spot" else 0
    if retries > 0:
        task_spec["max_retry_count"] = retries

    instance_policy: dict = {
        "machine_type": spec.profile.machine_type,
        "boot_disk": {"size_gb": spec.boot_disk_gb, "type_": "pd-balanced"},
    }
    if spec.profile.accelerator_type:
        instance_policy["accelerators"] = [
            {
                "type_": spec.profile.accelerator_type,
                "count": spec.profile.accelerator_count,
            }
        ]
    if spec.provisioning != "standard":
        instance_policy["provisioning_model"] = spec.provisioning.upper()
    if spec.provisioning == "flex_start":
        # Flex Start is only granted to jobs that opt out of reservations.
        instance_policy["reservation"] = "NO_RESERVATION"

    network_interface: Optional[dict] = None
    if spec.network or spec.subnetwork:
        network_interface = {}
        if spec.network:
            network_interface["network"] = spec.network
        if spec.subnetwork:
            network_interface["subnetwork"] = spec.subnetwork

    # By default allow Batch to pick any zone in the region — pinning to a
    # single zone blocks provisioning when that zone is GPU-exhausted
    # (CODE_GCE_ZONE_RESOURCE_POOL_EXHAUSTED), with no fallback. A zone pin is
    # opt-in (spec.zone) for jobs that must sit beside a zonal resource such as
    # a Rapid Cache or the dataset bucket's cache.
    if spec.zone:
        allowed_locations = [f"zones/{spec.zone}"]
    else:
        allowed_locations = [f"regions/{spec.region}"]
    instance_template: dict = {"policy": instance_policy}
    # installGpuDrivers belongs on InstancePolicyOrTemplate, NOT inside
    # accelerators[] (where it was historically — that location is silently
    # ignored in v1). Without this, the VM comes up with no NVIDIA driver
    # and torch.cuda.is_available() is False.
    if spec.profile.accelerator_type:
        instance_template["install_gpu_drivers"] = True
    allocation_policy: dict = {
        "instances": [instance_template],
        "location": {"allowed_locations": allowed_locations},
    }
    if spec.service_account:
        allocation_policy["service_account"] = {"email": spec.service_account}
    if network_interface:
        allocation_policy["network"] = {"network_interfaces": [network_interface]}

    # Caller labels first, then gcprunner_name last so it's always name-derived
    # (a caller cannot shadow it).
    labels = {_label_key_safe(k): _label_safe(v) for k, v in spec.labels.items()}
    labels["gcprunner_name"] = _label_safe(spec.name)

    job: dict = {
        "task_groups": [
            {
                "task_count": 1,
                "parallelism": 1,
                "task_spec": task_spec,
            }
        ],
        "allocation_policy": allocation_policy,
        "logs_policy": {"destination": "CLOUD_LOGGING"},
        "labels": labels,
    }
    return job


def _label_safe(s: str) -> str:
    """GCP label values: lowercase letters, digits, dashes, underscores; <=63 chars."""
    out = []
    for ch in s.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:63]


def _label_key_safe(s: str) -> str:
    """GCP label keys: like _label_safe but must start with a lowercase letter."""
    k = _label_safe(s)
    if not k or not k[0].isalpha():
        k = "k_" + k
    return k[:63]
