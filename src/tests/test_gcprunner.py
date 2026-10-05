"""Unit tests for gcprunner — no GCP SDK calls, purely spec construction."""

import pytest

from types import SimpleNamespace

from gcprunner import run_identity
from gcprunner.argparse_helpers import gcp_job_kwargs
from gcprunner.batch_spec import (
    PROVISIONING,
    DataMount,
    JobSpec,
    build_batch_job_dict,
)
from gcprunner.gpus import GPU_PROFILES, resolve_gpu
from gcprunner.job import Job, _STATE_MAP, _parse_slurm_time, _job_id_safe
from gcprunner.runner import multi_run


def test_resolve_gpu_canonical():
    p = resolve_gpu("l4")
    assert p.machine_type == "g2-standard-8"
    assert p.accelerator_type == "nvidia-l4"
    assert p.accelerator_count == 1


def test_resolve_gpu_slurm_alias():
    assert resolve_gpu("rtx4090").accelerator_type == "nvidia-l4"
    assert resolve_gpu("titan_rtx").accelerator_type == "nvidia-l4"
    assert resolve_gpu("standard").accelerator_type is None


def test_resolve_gpu_unknown():
    with pytest.raises(ValueError):
        resolve_gpu("definitely-not-a-gpu")


def test_data_mount_validation():
    with pytest.raises(ValueError):
        DataMount("not-a-uri", "/x", mode="fuse")
    with pytest.raises(ValueError):
        DataMount("gs://b/x", "/x", mode="bogus")


def test_parse_slurm_time():
    assert _parse_slurm_time(None) is None
    assert _parse_slurm_time("0:30:00") == 30 * 60
    assert _parse_slurm_time("1:00:00") == 3600
    assert _parse_slurm_time("7-0:0:0") == 7 * 86400
    assert _parse_slurm_time("1-2:3:4") == 86400 + 2 * 3600 + 3 * 60 + 4


def test_job_id_safe_alphanum():
    out = _job_id_safe("Hound-SNP_fold0_job17")
    assert all(c.isalnum() or c == "-" for c in out)
    assert out[0].isalpha()
    assert len(out) <= 63


def test_build_batch_job_dict_merges_labels():
    """Caller labels are merged + sanitized; gcprunner_name stays name-derived."""
    spec = JobSpec(
        name="b32-train-f0c0",
        cmd="python -m foo",
        image="us-central1-docker.pkg.dev/p/img:tag",
        profile=GPU_PROFILES["l4"],
        labels={
            "gcprunner_run": "2c5d7474-1e4a5bd1",
            "gcprunner_kind": "train",
            "gcprunner_fold": "f0c0",
        },
    )
    labels = build_batch_job_dict(spec)["labels"]
    assert labels["gcprunner_run"] == "2c5d7474-1e4a5bd1"
    assert labels["gcprunner_kind"] == "train"
    assert labels["gcprunner_fold"] == "f0c0"
    # name-derived label always present and sanitized
    assert labels["gcprunner_name"] == "b32-train-f0c0"


def test_build_batch_job_dict_sanitizes_labels():
    """Bad keys/values are coerced to GCP's rules; name label can't be shadowed."""
    spec = JobSpec(
        name="Job/Name",
        cmd="python -m foo",
        image="us-central1-docker.pkg.dev/p/img:tag",
        profile=GPU_PROFILES["l4"],
        labels={
            "1bad": "MixedCase/Value",  # key starts with a digit; value has /,caps
            "gcprunner_name": "attempted-override",  # must not win
        },
    )
    labels = build_batch_job_dict(spec)["labels"]
    assert "k_1bad" in labels
    assert labels["k_1bad"] == "mixedcase_value"
    # gcprunner_name is forced from spec.name, never the caller's value
    assert labels["gcprunner_name"] == "job_name"


def test_run_id_from_gcs_dir():
    assert run_identity.run_id_from_gcs_dir("gs://b/train/2c5d7474-1e4a5bd1") == (
        "2c5d7474-1e4a5bd1"
    )
    # trailing slash is stripped
    assert run_identity.run_id_from_gcs_dir("gs://b/eval/runX/") == "runX"


def test_identity_labels_sanitize_and_optional_fold():
    # fold omitted → run-scoped filter (no FOLD_KEY)
    run_scoped = run_identity.identity_labels("MyRun-2026", "train")
    assert run_scoped == {
        run_identity.RUN_KEY: "myrun-2026",  # value sanitized (lowercased)
        run_identity.KIND_KEY: "train",
    }
    full = run_identity.identity_labels("MyRun-2026", "train", "f0c0")
    assert full[run_identity.FOLD_KEY] == "f0c0"


def test_identity_labels_match_what_the_builder_stores():
    """A filter from identity_labels equals the labels build_batch_job_dict stores.

    This is the invariant that makes label matching correct even when a run_id
    isn't already GCP-safe: both sides sanitize identically.
    """
    run_id = "MyRun-2026"  # not label-safe (uppercase)
    spec = JobSpec(
        name="job",
        cmd="python -m foo",
        image="us-central1-docker.pkg.dev/p/img:tag",
        profile=GPU_PROFILES["l4"],
        labels=run_identity.identity_labels(run_id, "train", "f0c0"),
    )
    stored = build_batch_job_dict(spec)["labels"]
    want = run_identity.identity_labels(run_id, "train")  # run-scoped filter
    assert all(stored.get(k) == v for k, v in want.items())


def test_job_passes_labels_to_spec(monkeypatch):
    """Job(labels=...) flows into the JobSpec (no network)."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu", labels={"gcprunner_run": "runX"})
    assert j.spec.labels == {"gcprunner_run": "runX"}


def test_build_batch_job_dict_minimal():
    spec = JobSpec(
        name="snp-f0-c0-job1",
        cmd="python -m foo --bar 1",
        image="us-central1-docker.pkg.dev/p/img:tag",
        profile=GPU_PROFILES["l4"],
        cpu=4,
        mem_mb=30000,
        max_runtime_seconds=3600,
        provisioning="spot",
    )
    d = build_batch_job_dict(spec)
    tg = d["task_groups"][0]
    runnable = tg["task_spec"]["runnables"][0]
    assert runnable["container"]["entrypoint"] == "/usr/local/bin/gcprunner-entry"
    assert runnable["container"]["commands"] == [
        "/bin/bash",
        "-c",
        "python -m foo --bar 1",
    ]
    assert tg["task_spec"]["compute_resource"]["cpu_milli"] == 4000
    assert tg["task_spec"]["compute_resource"]["memory_mib"] == 30000
    assert tg["task_spec"]["max_run_duration"] == "3600s"

    inst = d["allocation_policy"]["instances"][0]["policy"]
    assert inst["machine_type"] == "g2-standard-8"
    assert inst["provisioning_model"] == "SPOT"
    assert inst["accelerators"][0]["type_"] == "nvidia-l4"
    assert inst["accelerators"][0]["count"] == 1


def test_build_batch_job_dict_no_spot_no_gpu():
    spec = JobSpec(
        name="cpu-job",
        cmd="echo hi",
        image="img",
        profile=GPU_PROFILES["cpu"],
        cpu=2,
        provisioning="standard",
    )
    d = build_batch_job_dict(spec)
    inst = d["allocation_policy"]["instances"][0]["policy"]
    assert "accelerators" not in inst
    assert "provisioning_model" not in inst


def test_flex_start_blocks_reservations():
    """FLEX_START is only granted to jobs that opt out of reservations."""
    spec = JobSpec(
        name="h100-job",
        cmd="hound_train",
        image="img",
        profile=GPU_PROFILES["h100"],
        provisioning="flex_start",
    )
    inst = build_batch_job_dict(spec)["allocation_policy"]["instances"][0]["policy"]
    assert inst["provisioning_model"] == "FLEX_START"
    assert inst["reservation"] == "NO_RESERVATION"


def test_invalid_provisioning_rejected():
    with pytest.raises(ValueError, match="invalid provisioning"):
        JobSpec(
            name="j",
            cmd="c",
            image="i",
            profile=GPU_PROFILES["cpu"],
            provisioning="dws",
        )


@pytest.mark.parametrize("provisioning", PROVISIONING)
def test_batch_job_dict_converts_to_proto(provisioning):
    """The dict is only ever consumed as a Job proto, so every key must be a real
    field of the installed SDK — an unknown key or enum raises here, not at submit."""
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    spec = JobSpec(
        name="train",
        cmd="train.py",
        image="img",
        profile=GPU_PROFILES["h100"],
        cpu=8,
        data_mounts=[DataMount("gs://b/data", "/workspace/data", mode="fuse")],
        output_dir_gcs="gs://b/out",
        provisioning=provisioning,
    )
    policy = (
        batch_v1.Job(build_batch_job_dict(spec)).allocation_policy.instances[0].policy
    )
    assert policy.provisioning_model.name == (
        "PROVISIONING_MODEL_UNSPECIFIED"
        if provisioning == "standard"
        else provisioning.upper()
    )
    assert policy.reservation == (
        "NO_RESERVATION" if provisioning == "flex_start" else ""
    )


def test_memory_defaults_to_profile_when_unset():
    """No explicit mem → memoryMib is the profile's usable VM memory, not 2000."""
    spec = JobSpec(
        name="train",
        cmd="hound_train",
        image="img",
        profile=GPU_PROFILES["a100-80"],
        cpu=8,
        # mem_mb left None — the training case
    )
    cr = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]["compute_resource"]
    assert cr["memory_mib"] == GPU_PROFILES["a100-80"].mem_mib
    assert cr["memory_mib"] != 2000  # not Batch's tiny default


def test_explicit_memory_overrides_profile():
    spec = JobSpec(
        name="snp",
        cmd="hound_snp",
        image="img",
        profile=GPU_PROFILES["l4"],
        cpu=4,
        mem_mb=30000,
    )
    cr = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]["compute_resource"]
    assert cr["memory_mib"] == 30000


def test_container_sets_shm_size():
    """DataLoader workers need a real /dev/shm, not Docker's 64 MB default."""
    spec = JobSpec(
        name="train",
        cmd="hound_train",
        image="img",
        profile=GPU_PROFILES["a100-40"],
        cpu=8,
    )
    opts = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]["runnables"][0][
        "container"
    ]["options"]
    assert "--privileged" in opts
    assert f"--shm-size={GPU_PROFILES['a100-40'].mem_mib // 2}m" in opts


def test_build_batch_job_dict_data_mounts_round_trip():
    mounts = [
        DataMount("gs://b/data", "/workspace/data", mode="fuse"),
        DataMount("gs://b/train", "/workspace/stage", mode="stage_local"),
    ]
    spec = JobSpec(
        name="train",
        cmd="train.py",
        image="img",
        profile=GPU_PROFILES["a100-80"],
        cpu=8,
        data_mounts=mounts,
        output_dir_gcs="gs://b/out",
        out_gcs="gs://b/out/stdout.log",
        err_gcs="gs://b/out/stderr.log",
    )
    d = build_batch_job_dict(spec)
    env = d["task_groups"][0]["task_spec"]["runnables"][0]["environment"]["variables"]
    assert env["GCPRUNNER_OUTPUT_DIR_GCS"] == "gs://b/out"
    assert env["GCPRUNNER_STDOUT_GCS"] == "gs://b/out/stdout.log"
    assert env["GCPRUNNER_STDERR_GCS"] == "gs://b/out/stderr.log"
    lines = env["GCPRUNNER_DATA_MOUNTS"].splitlines()
    assert lines == [
        "fuse\tgs://b/data\t/workspace/data",
        "stage_local\tgs://b/train\t/workspace/stage",
    ]


def test_job_requires_project(monkeypatch):
    """No built-in project: Job raises when neither project= nor the env var is set."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GCPRUNNER_PROJECT"):
        Job("echo hi", "n", queue="cpu")


def test_job_default_image_from_env_project(monkeypatch):
    """GCPRUNNER_PROJECT supplies the project and the default registry base."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu")
    assert j.project == "my-gcp-project"
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville:latest"
    )


def test_job_image_tag_expansion(monkeypatch):
    """Ensure tag provided in constructor expands to a full GAR URI."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu", image="dev-tag", project="my-project")
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:dev-tag"
    )


def test_job_env_image_tag_expansion(monkeypatch):
    """Ensure tag provided in environment variable expands to a full GAR URI."""
    monkeypatch.setenv("GCPRUNNER_IMAGE", "dev-tag")
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu", project="my-project")
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:dev-tag"
    )


def test_job_image_base_kwarg_expands_tag(monkeypatch):
    """A bare tag expands against image_base, decoupled from the compute project."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    base = "us-west1-docker.pkg.dev/my-registry-project/baskerville/baskerville"
    j = Job(
        "echo hi",
        "n",
        queue="cpu",
        image="build-123",
        image_base=base,
        project="my-gcp-project",
        region="us-west1",
    )
    # registry stays in my-registry-project even though compute is my-gcp-project
    assert j.spec.image == f"{base}:build-123"


def test_job_image_base_env_expands_tag(monkeypatch):
    """GCPRUNNER_IMAGE_BASE supplies the registry base for a bare tag."""
    base = "us-west1-docker.pkg.dev/my-registry-project/baskerville/baskerville"
    monkeypatch.setenv("GCPRUNNER_IMAGE_BASE", base)
    monkeypatch.setenv("GCPRUNNER_IMAGE", "build-123")
    monkeypatch.setenv("GCPRUNNER_PROJECT", "my-gcp-project")
    j = Job("echo hi", "n", queue="cpu")
    assert j.spec.image == f"{base}:build-123"


def test_job_numeric_tag_gets_prefix(monkeypatch):
    """A purely-numeric tag is prefixed (build number shorthand)."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    base = "us-west1-docker.pkg.dev/my-registry-project/baskerville/baskerville"
    j = Job(
        "echo hi",
        "n",
        queue="cpu",
        image="29162067376",
        image_base=base,
        image_tag_prefix="build-",
        project="my-gcp-project",
    )
    assert j.spec.image == f"{base}:build-29162067376"


def test_job_numeric_tag_default_prefix(monkeypatch):
    """With no env/kwarg, a numeric tag is used as-is."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_TAG_PREFIX", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu", image="29162067376", project="my-project")
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:29162067376"
    )


def test_job_tag_prefix_env_disables(monkeypatch):
    """An empty GCPRUNNER_IMAGE_TAG_PREFIX disables the default, leaving the tag bare."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.setenv("GCPRUNNER_IMAGE_TAG_PREFIX", "")
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job("echo hi", "n", queue="cpu", image="123", project="my-project")
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:123"
    )


def test_job_tag_prefix_kwarg_empty_disables(monkeypatch):
    """image_tag_prefix="" disables prefixing even with the default/env in play."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.setenv("GCPRUNNER_IMAGE_TAG_PREFIX", "build-")  # would otherwise apply
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    j = Job(
        "echo hi",
        "n",
        queue="cpu",
        image="123",
        image_tag_prefix="",
        project="my-project",
    )
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:123"
    )


def test_job_tag_prefix_env_and_skips_nonnumeric(monkeypatch):
    """GCPRUNNER_IMAGE_TAG_PREFIX applies to numeric tags but not named ones."""
    monkeypatch.setenv("GCPRUNNER_IMAGE_TAG_PREFIX", "build-")
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)
    prefix = "us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville"
    # numeric → prefixed
    j = Job("echo hi", "n", queue="cpu", image="123")
    assert j.spec.image == f"{prefix}:build-123"
    # already-prefixed / named tag → untouched
    j2 = Job("echo hi", "n", queue="cpu", image="build-123")
    assert j2.spec.image == f"{prefix}:build-123"
    # the "latest" default is not numeric → untouched
    j3 = Job("echo hi", "n", queue="cpu")
    assert j3.spec.image == f"{prefix}:latest"


def test_job_full_image_uri_ignores_base(monkeypatch):
    """A full image URI (contains '/') is used verbatim regardless of image_base."""
    monkeypatch.setenv("GCPRUNNER_IMAGE_BASE", "ignored-docker.pkg.dev/p/r/i")
    full = (
        "us-west1-docker.pkg.dev/my-registry-project/baskerville/baskerville:build-123"
    )
    j = Job("echo hi", "n", queue="cpu", image=full)
    assert j.spec.image == full


def test_job_region_env_applies_when_unset(monkeypatch):
    """region=None lets GCPRUNNER_REGION take effect (Fix: no shadowing default)."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.setenv("GCPRUNNER_REGION", "us-west1")
    j = Job("echo hi", "n", queue="cpu", image="dev-tag", project="my-project")
    assert (
        j.spec.image
        == "us-west1-docker.pkg.dev/my-project/baskerville/baskerville:dev-tag"
    )


def test_job_empty_region_env_falls_back_to_default(monkeypatch):
    """An empty GCPRUNNER_REGION falls back to the default, not "" (broken URI)."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    monkeypatch.setenv("GCPRUNNER_REGION", "")
    j = Job("echo hi", "n", queue="cpu", image="dev-tag", project="my-project")
    assert (
        j.spec.image
        == "us-central1-docker.pkg.dev/my-project/baskerville/baskerville:dev-tag"
    )


def test_job_slurmrunner_compat_kwargs(monkeypatch):
    """Slurmrunner-style kwargs (queue alias, mem in MB, time string) all work."""
    monkeypatch.setenv("GCPRUNNER_IMAGE", "img:tag")
    j = Job(
        cmd="python foo.py",
        name="snp-job1",
        out_file="gs://b/out.log",
        err_file="gs://b/err.log",
        sb_file="/tmp/ignored.sb",
        queue="rtx4090",
        cpu=4,
        mem=30000,
        time="7-0:0:0",
        gpu=1,
    )
    d = j.to_batch_dict()
    assert (
        d["allocation_policy"]["instances"][0]["policy"]["machine_type"]
        == "g2-standard-8"
    )
    assert d["task_groups"][0]["task_spec"]["max_run_duration"] == f"{7 * 86400}s"


def test_label_safe_truncation():
    from gcprunner.batch_spec import _label_safe

    assert _label_safe("Hound-SNP fold/0").startswith("hound-snp_fold_0")
    long = _label_safe("x" * 100)
    assert len(long) == 63


# ---------------------------------------------------------------------------
# zone pinning
# ---------------------------------------------------------------------------


def test_zone_pin_allowed_locations():
    spec = JobSpec(
        name="t",
        cmd="c",
        image="img",
        profile=GPU_PROFILES["a100-80"],
        region="us-central1",
        zone="us-central1-a",
    )
    d = build_batch_job_dict(spec)
    assert d["allocation_policy"]["location"]["allowed_locations"] == [
        "zones/us-central1-a"
    ]


def test_region_default_allowed_locations():
    spec = JobSpec(
        name="t", cmd="c", image="img", profile=GPU_PROFILES["l4"], region="us-central1"
    )
    d = build_batch_job_dict(spec)
    assert d["allocation_policy"]["location"]["allowed_locations"] == [
        "regions/us-central1"
    ]


def test_ai_zone_name_passes_validation():
    j = Job(
        "c", "n", queue="l4", region="us-central1", zone="us-central1-ai1a", image="i"
    )
    assert j.spec.zone == "us-central1-ai1a"


def test_zone_region_mismatch_raises():
    with pytest.raises(ValueError):
        Job("c", "n", queue="l4", region="us-central1", zone="us-west1-a", image="i")


# ---------------------------------------------------------------------------
# task retry
# ---------------------------------------------------------------------------


def test_retry_count_retries_any_failure():
    spec = JobSpec(
        name="train",
        cmd="hound_train",
        image="img",
        profile=GPU_PROFILES["a100-80"],
        retry_count=3,
        provisioning="standard",
    )
    ts = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]
    assert ts["max_retry_count"] == 3
    # no narrowing lifecycle → retries any non-zero exit
    assert "lifecycle_policies" not in ts


def test_spot_default_retries_all_failures():
    spec = JobSpec(
        name="snp",
        cmd="hound_snp",
        image="img",
        profile=GPU_PROFILES["l4"],
        provisioning="spot",
    )
    ts = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]
    assert ts["max_retry_count"] == 3
    # no exit-code narrowing → Batch retries any non-zero exit (incl. preemption
    # that doesn't surface as 50001)
    assert "lifecycle_policies" not in ts


def test_no_retry_on_demand_without_count():
    spec = JobSpec(
        name="snp",
        cmd="hound_snp",
        image="img",
        profile=GPU_PROFILES["l4"],
        provisioning="standard",
    )
    ts = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]
    assert "max_retry_count" not in ts
    assert "lifecycle_policies" not in ts


def test_explicit_zero_disables_spot_retries():
    spec = JobSpec(
        name="snp",
        cmd="hound_snp",
        image="img",
        profile=GPU_PROFILES["l4"],
        provisioning="spot",
        retry_count=0,
    )
    ts = build_batch_job_dict(spec)["task_groups"][0]["task_spec"]
    assert "max_retry_count" not in ts


def test_retry_count_passthrough_from_job():
    j = Job(
        "c", "n", queue="a100-80", retry_count=3, image="i", provisioning="standard"
    )
    assert j.spec.retry_count == 3


# ---------------------------------------------------------------------------
# --gcp_retry flag wiring
# ---------------------------------------------------------------------------


def _gcp_args(**overrides):
    """Minimal argparse-like namespace for gcp_job_kwargs."""
    base = dict(
        gcp_provisioning="standard",
        gcp_project="proj",
        gcp_region="us-central1",
        gcp_image="img",
        gcp_zone=None,
        gcp_retry=None,
        gcp_output_dir=None,
        gcp_service_account=None,
        gcp_data_dir=None,
        gcp_data_local="/workspace/data",
        gcp_stage_dir=None,
        gcp_stage_local="/workspace/stage",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gcp_retry_flag_passthrough():
    kwargs = gcp_job_kwargs(_gcp_args(gcp_retry=7))
    assert kwargs["retry_count"] == 7


def test_gcp_retry_unset_omits_kwarg():
    kwargs = gcp_job_kwargs(_gcp_args())
    assert "retry_count" not in kwargs


# ---------------------------------------------------------------------------
# multi_run failure handling
# ---------------------------------------------------------------------------


class _FakeJob:
    """Minimal Job stand-in for multi_run: terminal status set on update."""

    def __init__(self, name, final_status, provisioning="standard"):
        self.name = name
        self._final = final_status
        self.status = "PENDING"
        self.short_id = name
        self.cmd = "echo hi"
        self.spec = SimpleNamespace(provisioning=provisioning)
        self.launched = []

    def launch(self):
        self.launched.append(self.spec.provisioning)
        self.status = "PENDING"

    def update_status(self, *args, **kwargs):
        self.status = self._final


def test_multi_run_raises_on_failure():
    jobs = [
        _FakeJob("snp-f2c0-job20", "COMPLETED"),
        _FakeJob("snp-f2c0-job21", "FAILED"),
        _FakeJob("snp-f2c0-job22", "COMPLETED"),
    ]
    with pytest.raises(RuntimeError, match="snp-f2c0-job21"):
        multi_run(jobs, max_proc=3, launch_sleep=0, update_sleep=0)


def test_multi_run_all_completed_ok():
    jobs = [_FakeJob(f"j{i}", "COMPLETED") for i in range(3)]
    multi_run(jobs, max_proc=3, launch_sleep=0, update_sleep=0)  # no raise


def test_multi_run_spot_failure_falls_back_to_standard():
    class _PreemptedOnceJob(_FakeJob):
        def update_status(self, *args, **kwargs):
            self.status = "FAILED" if self.spec.provisioning == "spot" else "COMPLETED"

    job = _PreemptedOnceJob("j0", None, provisioning="spot")
    multi_run([job], launch_sleep=0, update_sleep=0)  # no raise
    assert job.launched == ["spot", "standard"]


def test_multi_run_spot_fallback_fails_once():
    job = _FakeJob("j0", "FAILED", provisioning="spot")
    with pytest.raises(RuntimeError, match="j0"):
        multi_run([job], launch_sleep=0, update_sleep=0)
    assert job.launched == ["spot", "standard"]


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_multi_run_cancelled_spot_job_is_not_relaunched(raise_on_failure):
    job = _FakeJob("j0", _STATE_MAP["CANCELLED"], provisioning="spot")
    kwargs = dict(launch_sleep=0, update_sleep=0, raise_on_failure=raise_on_failure)
    if raise_on_failure:
        with pytest.raises(RuntimeError, match="j0"):
            multi_run([job], **kwargs)
    else:
        multi_run([job], **kwargs)
    assert job.launched == ["spot"]
    assert job.status == "CANCELLED"


def test_multi_run_raises_on_launch_failure():
    class _UnlaunchableJob(_FakeJob):
        def launch(self):
            raise RuntimeError("CreateJob rejected")

    jobs = [_FakeJob("j0", "COMPLETED"), _UnlaunchableJob("j1", "FAILED")]
    with pytest.raises(RuntimeError, match="j1"):
        multi_run(jobs, max_proc=2, launch_sleep=0, update_sleep=0)


# ---------------------------------------------------------------------------
# branch → image resolution (image_branch + resolve_image_arg)
# ---------------------------------------------------------------------------

import json as _json

from gcprunner import image_branch as ib
from gcprunner.argparse_helpers import make_runner, resolve_image_arg


def _sha(c):
    return c * 40


def _mock_seams(monkeypatch, commits, index_entries):
    """Stub the two subprocess seams: git rev-list output and gcloud JSON."""
    monkeypatch.setattr(ib, "_run_git", lambda args, **k: "\n".join(commits) + "\n")
    monkeypatch.setattr(ib, "_run_gcloud", lambda args, **k: _json.dumps(index_entries))
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_REGION", raising=False)


def test_resolve_branch_picks_newest_built_commit(monkeypatch):
    # branch history newest→oldest is [a, b, c]; only b and c are built → pick b.
    _mock_seams(
        monkeypatch,
        commits=[_sha("a"), _sha("b"), _sha("c")],
        index_entries=[
            {
                "tags": ["commit-" + _sha("b"), "build-1", "latest"],
                "version": "sha256:bbb",
            },
            {"tags": "commit-" + _sha("c") + ",dev", "version": "sha256:ccc"},
        ],
    )
    base = "us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville"
    assert ib.resolve_branch_image("main") == f"{base}@sha256:bbb"


def test_resolve_branch_no_pin_uses_commit_tag(monkeypatch):
    _mock_seams(
        monkeypatch,
        commits=[_sha("a")],
        index_entries=[{"tags": ["commit-" + _sha("a")], "version": "sha256:aaa"}],
    )
    assert (
        ib.resolve_branch_image(
            "main", project="other-project", region="us-west1", pin_digest=False
        )
        == "us-west1-docker.pkg.dev/other-project/baskerville/baskerville"
        f":commit-{_sha('a')}"
    )


def test_resolve_branch_honors_image_base_env(monkeypatch):
    _mock_seams(
        monkeypatch,
        commits=[_sha("a")],
        index_entries=[{"tags": ["commit-" + _sha("a")], "version": "sha256:aaa"}],
    )
    monkeypatch.setenv("GCPRUNNER_IMAGE_BASE", "us-east1-docker.pkg.dev/p/r/img")
    assert (
        ib.resolve_branch_image("main") == "us-east1-docker.pkg.dev/p/r/img@sha256:aaa"
    )


def test_resolve_branch_requires_project_without_base(monkeypatch):
    _mock_seams(monkeypatch, commits=[_sha("a")], index_entries=[])
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GCPRUNNER_PROJECT"):
        ib.resolve_branch_image("main")


def test_resolve_image_arg_uses_marker_project_region(monkeypatch):
    """An explicit --gcp_branch resolved before the marker is applied still
    queries the training run's registry (project/region from the marker)."""
    _mock_seams(
        monkeypatch,
        commits=[_sha("a")],
        index_entries=[{"tags": ["commit-" + _sha("a")], "version": "sha256:aaa"}],
    )
    monkeypatch.delenv("GCPRUNNER_PROJECT", raising=False)
    args = SimpleNamespace(
        gcp_image=None, gcp_branch="feat", gcp_project=None, gcp_region=None
    )
    marker = {"gcp_project": "train-proj", "gcp_region": "us-west1"}
    assert resolve_image_arg(args, allow_default=False, marker=marker) == (
        "us-west1-docker.pkg.dev/train-proj/baskerville/baskerville@sha256:aaa"
    )


def test_resolve_branch_no_built_image_raises(monkeypatch):
    _mock_seams(monkeypatch, commits=[_sha("a"), _sha("b")], index_entries=[])
    with pytest.raises(ib.ImageBranchError, match="no built image for branch"):
        ib.resolve_branch_image("feature")


def test_branch_commits_walks_first_parent_only(monkeypatch):
    # Regression: a branch that merges main has main's commits in its full
    # ancestry, so a plain `git rev-list` could resolve to a pure-main build
    # lacking the branch's code. branch_commits must pass --first-parent to keep
    # the walk on the branch's own line of development.
    seen = {}

    def _capture(args, **k):
        seen["args"] = args
        return _sha("a") + "\n"

    monkeypatch.setattr(ib, "_run_git", _capture)
    ib.branch_commits("mamba3")
    assert "--first-parent" in seen["args"]
    assert "rev-list" in seen["args"]


def test_branch_commits_unresolvable_ref_raises(monkeypatch):
    def _boom(args, **k):
        raise ib.ImageBranchError("unknown revision")

    monkeypatch.setattr(ib, "_run_git", _boom)
    with pytest.raises(ib.ImageBranchError, match="could not resolve branch"):
        ib.branch_commits("nope")


def test_resolve_image_arg_explicit_image_wins(monkeypatch):
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: "SHOULD-NOT-RUN")
    args = SimpleNamespace(gcp_image="my/img:tag", gcp_branch="x")
    assert resolve_image_arg(args) == "my/img:tag"
    assert args.gcp_image == "my/img:tag"


def test_resolve_image_arg_explicit_branch(monkeypatch):
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: f"resolved:{b}")
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    args = SimpleNamespace(gcp_image=None, gcp_branch="feature")
    assert resolve_image_arg(args) == "resolved:feature"
    assert args.gcp_image == "resolved:feature"


def test_resolve_image_arg_default_main(monkeypatch):
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: f"resolved:{b}")
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    args = SimpleNamespace(gcp_image=None, gcp_branch=None)
    assert resolve_image_arg(args) == "resolved:main"
    assert args.gcp_image == "resolved:main"


def test_resolve_image_arg_no_default_is_noop(monkeypatch):
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: "SHOULD-NOT-RUN")
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    args = SimpleNamespace(gcp_image=None, gcp_branch=None)
    assert resolve_image_arg(args, allow_default=False) is None
    assert args.gcp_image is None


def test_resolve_image_arg_env_pin_is_noop(monkeypatch):
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: "SHOULD-NOT-RUN")
    monkeypatch.setenv("GCPRUNNER_IMAGE", "env/img:tag")
    args = SimpleNamespace(gcp_image=None, gcp_branch=None)
    assert resolve_image_arg(args) is None
    assert args.gcp_image is None


def test_make_runner_gcp_backstops_default_image(monkeypatch):
    # A GCP script that never called resolve_image_arg itself still gets the
    # branch/main default filled in by make_runner (covers snp/grad/distill).
    monkeypatch.setattr(ib, "resolve_branch_image", lambda b, **k: f"resolved:{b}")
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    args = _gcp_args(backend="gcp", gcp_image=None, gcp_branch=None)
    make_runner(args)
    assert args.gcp_image == "resolved:main"


def test_make_runner_gcp_skips_when_image_already_set(monkeypatch):
    # Idempotent: an image resolved earlier (e.g. eval's marker phase) is left
    # untouched — the backstop must not re-resolve or clobber it.
    def _boom(b, **k):
        raise AssertionError("resolve_branch_image should not run")

    monkeypatch.setattr(ib, "resolve_branch_image", _boom)
    args = _gcp_args(backend="gcp", gcp_image="pinned:img", gcp_branch=None)
    make_runner(args)
    assert args.gcp_image == "pinned:img"


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_multi_run_limits_concurrency_and_launches_once(limit):
    active = set()
    launched = []

    class Job(_FakeJob):
        def launch(self):
            assert self.name not in launched
            launched.append(self.name)
            if self.name == "bad-launch":
                raise RuntimeError("rejected")
            active.add(self.name)
            assert len(active) <= limit
            self.polls = 0

        def update_status(self, **kwargs):
            self.polls += 1
            if self.polls == 2:
                active.remove(self.name)
                self.status = self._final
            else:
                self.status = "RUNNING"

    jobs = [Job(str(i), "COMPLETED") for i in range(5)]
    jobs.insert(1, Job("bad-launch", "FAILED"))
    jobs.insert(3, Job("bad-task", "FAILED"))
    with pytest.raises(RuntimeError, match="bad-launch, bad-task"):
        multi_run(jobs, max_proc=limit, launch_sleep=0, update_sleep=0)
    assert launched == [job.name for job in jobs]
    assert not active
    assert jobs[1].status == "FAILED"


@pytest.mark.parametrize("limit", [0, -1])
def test_multi_run_rejects_invalid_concurrency(limit):
    with pytest.raises(ValueError, match="max_proc"):
        multi_run([_FakeJob("job", "COMPLETED")], max_proc=limit)


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_submission_errors_always_raise_after_other_jobs_finish(raise_on_failure):
    class UnlaunchableJob(_FakeJob):
        def launch(self):
            raise RuntimeError("503 CreateJob unavailable")

    jobs = [
        _FakeJob("first", "COMPLETED"),
        UnlaunchableJob("not-submitted", "FAILED"),
        _FakeJob("last", "COMPLETED"),
    ]
    with pytest.raises(RuntimeError, match="not-submitted: 503 CreateJob unavailable"):
        multi_run(
            jobs,
            max_proc=2,
            launch_sleep=0,
            update_sleep=0,
            raise_on_failure=raise_on_failure,
        )
    assert [job.status for job in jobs] == ["COMPLETED", "FAILED", "COMPLETED"]


def test_backend_default_follows_slurmrunner(monkeypatch):
    import argparse
    import importlib.util

    from gcprunner.argparse_helpers import add_argparse_group

    for spec, expected in ((None, "local"), (object(), "slurm")):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, s=spec: s)
        parser = argparse.ArgumentParser()
        add_argparse_group(parser)
        assert parser.parse_args([]).backend == expected
