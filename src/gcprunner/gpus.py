"""GPU profiles for GCP Batch.

Each alias resolves to a tuple of (machine_type, accelerator_type,
accelerator_count, mem_mib). New aliases can be added here without
touching call sites — the analogue of slurmrunner's ``GPU_TRANSLATIONS``.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuProfile:
    machine_type: str
    accelerator_type: Optional[str]
    accelerator_count: int
    # Usable memory per task (MiB). This is the VM's physical RAM minus headroom
    # the GCE/Batch agent + OS reserve — requesting the full physical amount gets
    # the job rejected. Used as the default Batch ``computeResource.memoryMib``
    # when a caller doesn't pass an explicit ``mem``: every job here is
    # single-task, so the one task should get (nearly) the whole VM rather than
    # Batch's 2000 MiB default. ~91-94% of physical, matching the SNP convention
    # (30000 on a 32 GiB g2-standard-8).
    mem_mib: int


GPU_PROFILES: dict[str, GpuProfile] = {
    # CPU-only
    "cpu": GpuProfile("n2-standard-8", None, 0, 30000),  # 32 GiB
    "cpu-large": GpuProfile("n2-standard-32", None, 0, 122000),  # 128
    # T4 — cheap inference
    "t4": GpuProfile("n1-standard-8", "nvidia-tesla-t4", 1, 28000),  # 30
    # L4 — modern Ada inference; recommended default for SNP / ISM / eval
    "l4": GpuProfile("g2-standard-8", "nvidia-l4", 1, 30000),  # 32 GiB
    "l4-large": GpuProfile("g2-standard-32", "nvidia-l4", 1, 122000),  # 128
    # V100
    "v100": GpuProfile("n1-standard-8", "nvidia-tesla-v100", 1, 28000),  # 30
    # A100 40GB
    "a100-40": GpuProfile("a2-highgpu-1g", "nvidia-tesla-a100", 1, 80000),  # 85 GiB
    "a100-40x4": GpuProfile("a2-highgpu-4g", "nvidia-tesla-a100", 4, 330000),  # 340
    # A100 80GB
    "a100-80": GpuProfile("a2-ultragpu-1g", "nvidia-a100-80gb", 1, 160000),  # 170 GiB
    # H100 80GB — accelerator-optimized A3
    "h100": GpuProfile("a3-highgpu-1g", "nvidia-h100-80gb", 1, 224000),  # 234 GiB
    "h100x8": GpuProfile("a3-highgpu-8g", "nvidia-h100-80gb", 8, 1800000),  # 1872
}


# Slurm-style aliases (so existing scripts that pass --queue rtx4090 etc. fall
# back to a sensible cloud equivalent). Maps slurmrunner alias → gcprunner
# alias. Anything not found here is passed through and resolved against
# GPU_PROFILES; unknowns raise.
SLURM_ALIASES: dict[str, str] = {
    "rtx4090": "l4",
    "titan_rtx": "l4",
    "titan": "l4",
    "p100": "t4",
    "tesla": "t4",
    "geforce": "t4",
    "gtx1080": "t4",
    "quadro": "v100",
    "standard": "cpu",
    "cpu_compute": "cpu",
}


def resolve_gpu(queue: str) -> GpuProfile:
    """Resolve a queue name to a GpuProfile.

    Accepts slurmrunner-style aliases (rtx4090, titan_rtx, …) and gcprunner
    canonical names (l4, a100-80, h100, …).
    """
    name = SLURM_ALIASES.get(queue, queue)
    if name not in GPU_PROFILES:
        raise ValueError(
            f"Unknown GPU/queue alias {queue!r}. "
            f"Known: {sorted(GPU_PROFILES)} (or slurm aliases: {sorted(SLURM_ALIASES)})"
        )
    return GPU_PROFILES[name]
