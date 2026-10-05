#!/usr/bin/env python
"""Offline tests for transfer-learning param staging in hound_train_folds (GCP).

No real GCS or Batch calls — staging is monkeypatched and the staged params
content is captured for assertions.
"""

import json
from types import SimpleNamespace

import pytest

from baskerville.scripts import hound_train_folds as htf


class MockArgs:
    """Minimal args for _stage_train_params (mirrors the argparse defaults)."""

    def __init__(self, **kwargs):
        self.params_file = "params.json"
        self.transfer = None
        for key, value in kwargs.items():
            setattr(self, key, value)


def _write_params(path):
    """A params.json with a model section make_rep_params can inject into."""
    with open(path, "w") as f:
        f.write(
            '{\n    "train": {},\n    "model": {\n        "seq_length": 16\n    }\n}\n'
        )


def _patch_stage(monkeypatch, *, transfer_container=None):
    """Capture (type, basename, content) for every stage_file/stage_dir call."""
    staged_files = []  # (type_, content_str)

    def fake_stage_file(local_path, type_, **kwargs):
        with open(local_path) as f:
            content = f.read()
        staged_files.append((type_, content))
        # container path is content-addressed; a stable stand-in is fine here
        idx = len(staged_files) - 1
        return f"sha{idx}", f"/workspace/cache/{type_}/sha{idx}/params.json"

    def fake_stage_dir(local_path, type_, **kwargs):
        return "transfersha", transfer_container

    monkeypatch.setattr(htf.stage_cache, "stage_file", fake_stage_file)
    monkeypatch.setattr(htf.stage_cache, "stage_dir", fake_stage_dir)
    return staged_files


def test_transfer_embeds_per_fold_pretrained_model(tmp_path, monkeypatch):
    params = tmp_path / "params.json"
    _write_params(params)
    transfer_container = "/workspace/cache/models/abc"
    staged = _patch_stage(monkeypatch, transfer_container=transfer_container)

    args = MockArgs(params_file=str(params), transfer="/path/to/transfer")
    fold_crosses = ["f0c0", "f1c0"]
    mapping = htf._stage_train_params(args, fold_crosses, transfer_container)

    # one staged params (and one map entry) per fold
    assert set(mapping) == set(fold_crosses)
    assert len(staged) == len(fold_crosses)

    # each fold's staged params carries the matching foundation weight path
    for (type_, content), fold_cross in zip(staged, fold_crosses):
        assert type_ == "params"
        params_json = json.loads(content)
        assert (
            params_json["model"]["pretrained_model"]
            == f"{transfer_container}/{fold_cross}/train/model_best.pth"
        )


def test_no_transfer_stages_params_once(tmp_path, monkeypatch):
    params = tmp_path / "params.json"
    _write_params(params)
    staged = _patch_stage(monkeypatch)

    args = MockArgs(params_file=str(params), transfer=None)
    fold_crosses = ["f0c0", "f1c0", "f2c0"]
    mapping = htf._stage_train_params(args, fold_crosses, None)

    # staged exactly once; every fold maps to the same container path
    assert len(staged) == 1
    assert staged[0][0] == "params"
    assert "pretrained_model" not in json.loads(staged[0][1])["model"]
    assert set(mapping) == set(fold_crosses)
    assert len(set(mapping.values())) == 1


# ---------------------------------------------------------------------------
# --conclude: stop a run and pull it down
# ---------------------------------------------------------------------------


def test_cancel_active_jobs_filters_by_substr_and_skips_terminal(monkeypatch):
    """cancel_active_jobs cancels only active jobs whose name matches the substr."""
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    from gcprunner import runner

    running = batch_v1.JobStatus.State.RUNNING
    succeeded = batch_v1.JobStatus.State.SUCCEEDED

    def _job(name, state, labels=None):
        status = type("Status", (), {"state": state})()
        return type(
            "Job", (), {"name": name, "status": status, "labels": labels or {}}
        )()

    jobs = [
        _job("projects/p/locations/r/jobs/fold-train-f0c0-aaa", running),
        _job("projects/p/locations/r/jobs/fold-train-f1c0-bbb", running),
        _job("projects/p/locations/r/jobs/other-job-ccc", running),  # wrong prefix
        _job("projects/p/locations/r/jobs/fold-train-f2c0-ddd", succeeded),  # terminal
    ]
    cancelled = []

    class FakeClient:
        def list_jobs(self, request=None):
            return jobs

        def cancel_job(self, request=None):
            cancelled.append(request.name)

    monkeypatch.setattr(batch_v1, "BatchServiceClient", lambda: FakeClient())

    out = runner.cancel_active_jobs("p", "r", "fold-train-")

    expected = {
        "projects/p/locations/r/jobs/fold-train-f0c0-aaa",
        "projects/p/locations/r/jobs/fold-train-f1c0-bbb",
    }
    assert set(out) == expected
    assert set(cancelled) == expected


def test_cancel_active_jobs_filters_by_labels(monkeypatch):
    """cancel_active_jobs(labels=...) cancels only jobs matching every label."""
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    from gcprunner import runner

    running = batch_v1.JobStatus.State.RUNNING
    succeeded = batch_v1.JobStatus.State.SUCCEEDED

    def _job(name, state, labels):
        status = type("Status", (), {"state": state})()
        return type("Job", (), {"name": name, "status": status, "labels": labels})()

    # GCP stores label values lowercased (via _label_safe at submit time), so
    # fixture jobs use the sanitized form here; the filter below passes the
    # raw mixed-case run_id to also exercise cancel_active_jobs' own sanitizing.
    jobs = [
        # target run + kind: cancel
        _job("j/a", running, {"gcprunner_run": "runa", "gcprunner_kind": "train"}),
        _job("j/b", running, {"gcprunner_run": "runa", "gcprunner_kind": "train"}),
        # same run but eval kind: leave alone
        _job("j/c", running, {"gcprunner_run": "runa", "gcprunner_kind": "eval"}),
        # different run: leave alone
        _job("j/d", running, {"gcprunner_run": "runb", "gcprunner_kind": "train"}),
        # right labels but terminal: skip
        _job("j/e", succeeded, {"gcprunner_run": "runa", "gcprunner_kind": "train"}),
        # unlabeled (pre-label code): leave alone
        _job("j/f", running, {}),
    ]
    cancelled = []

    class FakeClient:
        def list_jobs(self, request=None):
            return jobs

        def cancel_job(self, request=None):
            cancelled.append(request.name)

    monkeypatch.setattr(batch_v1, "BatchServiceClient", lambda: FakeClient())

    out = runner.cancel_active_jobs(
        "p", "r", labels={"gcprunner_run": "runA", "gcprunner_kind": "train"}
    )
    assert set(out) == {"j/a", "j/b"}
    assert set(cancelled) == {"j/a", "j/b"}


def test_cancel_active_jobs_requires_a_filter():
    """A bare call (no name_substr, no labels) must refuse to cancel everything."""
    pytest.importorskip("google.cloud.batch_v1")
    from gcprunner import runner

    with pytest.raises(ValueError):
        runner.cancel_active_jobs("p", "r")


def test_list_active_jobs_surfaces_labels(monkeypatch):
    """list_active_jobs returns each job's labels and still drops terminal jobs."""
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    from gcprunner import runner

    running = batch_v1.JobStatus.State.RUNNING
    cancelled_state = batch_v1.JobStatus.State.CANCELLED

    def _job(name, state, labels):
        status = type("Status", (), {"state": state})()
        return type("Job", (), {"name": name, "status": status, "labels": labels})()

    jobs = [
        _job("j/a", running, {"gcprunner_run": "runA"}),
        _job("j/b", cancelled_state, {"gcprunner_run": "runA"}),  # terminal → dropped
    ]

    class FakeClient:
        def list_jobs(self, request=None):
            return jobs

    monkeypatch.setattr(batch_v1, "BatchServiceClient", lambda: FakeClient())

    out = runner.list_active_jobs("p", "r")
    assert len(out) == 1
    assert out[0]["name"] == "j/a"
    assert out[0]["labels"] == {"gcprunner_run": "runA"}


def test_conclude_cancels_run_and_full_pulls(tmp_path, monkeypatch):
    """_conclude_gcp_run cancels with the run's name prefix and pulls incl. .pth."""
    import gcprunner

    calls = {}

    # no marker -> resolve GCS dir from the explicit --gcp_output_dir
    monkeypatch.setattr(htf.stage_cache, "read_run_marker", lambda out_dir: None)

    def fake_cancel(project, region, name_substr=None, labels=None):
        calls["cancel"] = (project, region, name_substr, labels)
        return []  # nothing active -> skips the wait-for-stop poll loop

    monkeypatch.setattr(gcprunner, "cancel_active_jobs", fake_cancel)

    def fake_mirror(gcs_dir, local_dir, exclude=htf._MIRROR_EXCLUDE):
        calls["mirror"] = (gcs_dir, local_dir, exclude)

    monkeypatch.setattr(htf, "_mirror_once", fake_mirror)
    monkeypatch.setattr(htf, "_warn_if_orchestrator_alive", lambda out_dir: None)

    args = MockArgs(
        out_dir=str(tmp_path),
        gcp_output_dir="gs://bucket/train/run123",
        gcp_project="proj",
        gcp_region="us-west1",
        name="fold",
    )
    htf._conclude_gcp_run(args)

    # cancel is scoped by run identity (run_id + kind), NOT the --name substring
    project, region, name_substr, labels = calls["cancel"]
    assert (project, region) == ("proj", "us-west1")
    assert name_substr is None
    assert labels == {"gcprunner_run": "run123", "gcprunner_kind": "train"}
    gcs_dir, local_dir, exclude = calls["mirror"]
    assert gcs_dir == "gs://bucket/train/run123"
    assert local_dir == str(tmp_path)
    assert exclude is None  # full pull, unlike the periodic mirror


def _patch_gcp_training_env(monkeypatch, mirror_calls):
    """Stub out marker/resolve/summary so _run_gcp_training only exercises the
    mirror daemon lifecycle. Records every _mirror_once call's exclude arg."""
    monkeypatch.setattr(
        htf.stage_cache, "write_run_marker", lambda *a, **k: "marker.json"
    )
    monkeypatch.setattr(htf, "resolve_region", lambda x: x or "us-west1")
    monkeypatch.setattr(htf, "resolve_project", lambda x: x or "proj")
    monkeypatch.setattr(htf, "resolve_zone", lambda x: x)
    monkeypatch.setattr(htf, "resolve_image", lambda img, **k: img)
    monkeypatch.setattr(htf, "_print_conclude_summary", lambda out_dir: None)
    # a non-None stop() so the final-mirror branch in _run_gcp_training runs
    monkeypatch.setattr(htf, "_start_output_mirror", lambda *a, **k: lambda: None)

    def fake_mirror(gcs_dir, local_dir, exclude=htf._MIRROR_EXCLUDE):
        mirror_calls.append(exclude)

    monkeypatch.setattr(htf, "_mirror_once", fake_mirror)


def _gcp_training_args(tmp_path):
    return MockArgs(
        out_dir=str(tmp_path),
        gcp_output_dir="gs://bucket/train/run123",
        gcp_data_dir="gs://bucket/data",
        gcp_image="img",
        gcp_project="proj",
        gcp_region="us-west1",
        gcp_zone=None,
    )


@pytest.mark.parametrize(
    "loop_result, expected_exclude",
    [(True, None), (False, htf._MIRROR_EXCLUDE)],
)
def test_final_mirror_scope_by_outcome(
    tmp_path, monkeypatch, loop_result, expected_exclude
):
    """Success (loop True) -> full pull incl. .pth (exclude=None); an
    aborted/incomplete run (loop False) -> light catch-up keeping the .pth
    exclude, weights stay in GCS."""
    mirror_calls = []
    _patch_gcp_training_env(monkeypatch, mirror_calls)
    monkeypatch.setattr(htf, "_resubmit_loop", lambda *a, **k: loop_result)

    htf._run_gcp_training(_gcp_training_args(tmp_path), {}, {}, [], [])

    assert mirror_calls == [expected_exclude]


def test_interrupt_light_mirror_only(tmp_path, monkeypatch):
    """Ctrl-C mid-run (loop raises) -> finally does the light catch-up only,
    never a full weight pull, and the interrupt still propagates."""
    mirror_calls = []
    _patch_gcp_training_env(monkeypatch, mirror_calls)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(htf, "_resubmit_loop", boom)

    with pytest.raises(KeyboardInterrupt):
        htf._run_gcp_training(_gcp_training_args(tmp_path), {}, {}, [], [])

    assert mirror_calls == [htf._MIRROR_EXCLUDE]


def test_run_marker_records_resolved_env_config(tmp_path, monkeypatch):
    """The run marker captures the *resolved* project/region/image, so a
    downstream command inherits them even when the run was configured via env."""
    # config supplied entirely through the environment; args left at None
    monkeypatch.setenv("GCPRUNNER_PROJECT", "my-gcp-project")
    monkeypatch.setenv("GCPRUNNER_REGION", "us-west1")
    monkeypatch.setenv(
        "GCPRUNNER_IMAGE_BASE",
        "us-west1-docker.pkg.dev/my-registry-project/baskerville/baskerville",
    )
    monkeypatch.setenv("GCPRUNNER_IMAGE_TAG_PREFIX", "build-")
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_ZONE", raising=False)

    # skip the mirror daemon and the resubmit loop; we only exercise marker write
    monkeypatch.setattr(htf, "_start_output_mirror", lambda *a, **k: None)
    monkeypatch.setattr(htf, "_resubmit_loop", lambda *a, **k: None)

    args = MockArgs(
        out_dir=str(tmp_path),
        gcp_output_dir="gs://bucket/train/run123",
        gcp_data_dir="gs://bucket/data",
        gcp_image="29162067376",  # bare CI build number
        gcp_project=None,
        gcp_region=None,
        gcp_zone=None,
    )
    htf._run_gcp_training(args, {}, {}, [], [])

    with open(tmp_path / htf.stage_cache.RUN_MARKER_NAME) as f:
        marker = json.load(f)
    assert marker["gcp_project"] == "my-gcp-project"
    assert marker["gcp_region"] == "us-west1"
    # bare numeric tag → prefixed + expanded against the base, stored as a full URI
    assert marker["gcp_image"] == (
        "us-west1-docker.pkg.dev/my-registry-project/baskerville/"
        "baskerville:build-29162067376"
    )


def test_run_marker_records_branch_resolved_digest(tmp_path, monkeypatch):
    """A branch-resolved digest URI (set by resolve_image_arg in main before the
    marker is written) round-trips through the marker verbatim, so eval inherits
    the exact same image — the reproducibility chain for --gcp_branch."""
    monkeypatch.delenv("GCPRUNNER_IMAGE", raising=False)
    monkeypatch.delenv("GCPRUNNER_IMAGE_BASE", raising=False)
    monkeypatch.setattr(htf, "_start_output_mirror", lambda *a, **k: None)
    monkeypatch.setattr(htf, "_resubmit_loop", lambda *a, **k: None)

    digest_uri = (
        "us-west1-docker.pkg.dev/my-registry-project/baskerville/"
        "baskerville@sha256:deadbeef"
    )
    args = MockArgs(
        out_dir=str(tmp_path),
        gcp_output_dir="gs://bucket/train/run123",
        gcp_data_dir="gs://bucket/data",
        gcp_image=digest_uri,  # already resolved from --gcp_branch in main()
        gcp_project="proj",
        gcp_region="us-west1",
        gcp_zone=None,
    )
    htf._run_gcp_training(args, {}, {}, [], [])

    with open(tmp_path / htf.stage_cache.RUN_MARKER_NAME) as f:
        marker = json.load(f)
    assert marker["gcp_image"] == digest_uri  # '/' URI stored verbatim

    # eval inherits the exact digest via apply_run_marker
    eval_args = MockArgs(
        gcp_image=None,
        gcp_data_dir=None,
        gcp_project=None,
        gcp_region=None,
        gcp_zone=None,
    )
    inherited = htf.stage_cache.apply_run_marker(eval_args, marker)
    assert "gcp_image" in inherited
    assert eval_args.gcp_image == digest_uri


# ---------------------------------------------------------------------------
# run-identity labels: detection + job construction
# ---------------------------------------------------------------------------


def test_active_fold_crosses_is_run_scoped(monkeypatch):
    """Only folds whose active job carries this run_id + kind=train count."""
    import gcprunner

    jobs = [
        {
            "name": "j/a",
            "labels": {
                "gcprunner_run": "runa",
                "gcprunner_kind": "train",
                "gcprunner_fold": "f0c0",
            },
        },
        {
            "name": "j/b",
            "labels": {
                "gcprunner_run": "runa",
                "gcprunner_kind": "train",
                "gcprunner_fold": "f1c0",
            },
        },
        # same run, but an eval job → ignored
        {
            "name": "j/c",
            "labels": {
                "gcprunner_run": "runa",
                "gcprunner_kind": "eval",
                "gcprunner_fold": "f2c0",
            },
        },
        # a different run → ignored (this is the collision the change fixes)
        {
            "name": "j/d",
            "labels": {
                "gcprunner_run": "runb",
                "gcprunner_kind": "train",
                "gcprunner_fold": "f2c0",
            },
        },
        # pre-label job (no labels) → ignored
        {"name": "j/e", "labels": {}},
    ]
    monkeypatch.setattr(gcprunner, "list_active_jobs", lambda p, r: jobs)

    active = htf._active_fold_crosses("p", "r", ["f0c0", "f1c0", "f2c0"], "runa")
    assert active == {"f0c0", "f1c0"}


def test_active_fold_crosses_sanitizes_run_id(monkeypatch):
    """A run_id that isn't already GCP-label-safe still matches its stored labels.

    Regression: detection must sanitize the run_id the same way the spec builder
    did, or a --gcp_output_dir with e.g. an uppercase basename silently fails to
    detect its own active jobs and double-launches.
    """
    import gcprunner
    from gcprunner import run_identity

    # what the builder would have stored (sanitized) for this run
    stored = run_identity.identity_labels("MyRun-2026", "train", "f0c0")
    monkeypatch.setattr(
        gcprunner, "list_active_jobs", lambda p, r: [{"name": "j", "labels": stored}]
    )

    # caller passes the RAW (unsanitized) run_id
    active = htf._active_fold_crosses("p", "r", ["f0c0"], "MyRun-2026")
    assert active == {"f0c0"}


def test_build_gcp_train_job_sets_identity_labels(monkeypatch):
    """_build_gcp_train_job stamps run/kind/fold labels from the output dir."""
    import gcprunner

    # slurmrunner may be absent in the test env; the real runtime wraps gcprunner.
    monkeypatch.setattr(htf, "slurmrunner", gcprunner)

    args = MockArgs(
        gcp_output_dir="gs://bucket/train/run123",
        name="b32",
        queue="l4",
        whole=False,
        gcp_retry_count=0,
        gcp_data_dir="gs://bucket/data",
        gcp_data_local="/workspace/data",
    )
    job = htf._build_gcp_train_job(
        args,
        {"num_gpu": 1},
        "/workspace/cache/params/x/params.json",
        ["/workspace/data/hg38"],
        "f0c0",
    )
    assert job.spec.labels == {
        "gcprunner_run": "run123",
        "gcprunner_kind": "train",
        "gcprunner_fold": "f0c0",
    }


@pytest.mark.parametrize(
    "initial, outcomes, expected_complete",
    [
        (("new", None), [("incomplete", 0), ("complete", None)], True),
        (("new", None), [("new", None)], False),
        (("new", None), [("incomplete", 0), ("incomplete", 0)], False),
        (("incomplete", 0), [("incomplete", 0)], False),
        (("incomplete", 3), [("incomplete", 2)], False),
        (("incomplete", 3), [("incomplete", None)], False),
        (("incomplete", None), [("incomplete", 3)], False),
        (("incomplete", 3), [("incomplete", 4), ("complete", None)], True),
        (("incomplete", 3), [("complete", None)], True),
    ],
)
def test_resubmit_after_batch_failure(
    monkeypatch, initial, outcomes, expected_complete
):
    from gcprunner import runner

    state = {"value": initial, "attempts": 0}

    class Job:
        name = short_id = "train-f0c0"
        cmd = "hound_train"
        status = "PENDING"
        spec = SimpleNamespace(provisioning="standard")

        def launch(self):
            state["attempts"] += 1

        def update_status(self, **kwargs):
            state["value"] = outcomes[state["attempts"] - 1]
            self.status = "COMPLETED" if state["value"][0] == "complete" else "FAILED"

    monkeypatch.setattr(htf, "slurmrunner", runner)
    monkeypatch.setattr(runner._time, "sleep", lambda _: None)
    monkeypatch.setattr(htf, "_active_fold_crosses", lambda *a: set())
    monkeypatch.setattr(htf, "_fold_state", lambda *a: state["value"])
    monkeypatch.setattr(htf, "_build_gcp_train_job", lambda *a: Job())
    args = MockArgs(
        gcp_output_dir="gs://b/run",
        gcp_project="p",
        gcp_region="r",
        gcp_max_rounds=10,
        processes=1,
    )
    completed = htf._resubmit_loop(args, {}, {"f0c0": "params"}, [], ["f0c0"])
    assert completed is expected_complete
    assert state["attempts"] == len(outcomes)


def test_active_job_lookup_failure_propagates(monkeypatch):
    import gcprunner

    def fail(*args):
        raise RuntimeError("lookup unavailable")

    monkeypatch.setattr(gcprunner, "list_active_jobs", fail)
    with pytest.raises(RuntimeError, match="lookup unavailable"):
        htf._active_fold_crosses("p", "r", ["f0c0"], "run")


@pytest.mark.parametrize(
    "complete, epochs_max, expected",
    [
        ({"epoch": 49, "reason": "max_epochs"}, 50, ("complete", None)),
        ({"epoch": 49, "reason": "max_epochs"}, 70, ("incomplete", 50)),
        ({"epoch": 49, "reason": "early_stop"}, 70, ("complete", None)),
        ({"epoch": 49}, 70, ("complete", None)),  # pre-reason marker
    ],
)
def test_fold_state_extends_max_epochs(monkeypatch, complete, epochs_max, expected):
    base = "gs://b/run/f0c0/train"
    files = {f"{base}/COMPLETE", f"{base}/checkpoint.pth"}
    monkeypatch.setattr(htf, "gcs_file_exist", lambda uri: uri in files)
    monkeypatch.setattr(
        htf,
        "read_json_gcs",
        lambda uri: complete if uri.endswith("COMPLETE") else {"epoch": 50},
    )
    assert htf._fold_state("gs://b/run", "f0c0", set(), epochs_max) == expected


def test_extend_inherits_run_dir_and_image(tmp_path, monkeypatch):
    htf.stage_cache.write_run_marker(
        str(tmp_path),
        output_dir_gcs="gs://b/train/old-run",
        gcp_image="reg/img@sha256:abc",
        gcp_project="p",
        gcp_region="r",
    )
    seen = {}

    def stop(args):
        seen.update(vars(args))
        raise SystemExit(0)

    monkeypatch.setattr(htf, "resolve_image_arg", stop)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hound_train_folds",
            "--backend",
            "gcp",
            "--extend",
            "--gcp_branch",
            "main",
            "-o",
            str(tmp_path),
            "params.json",
            "hg38",
        ],
    )
    with pytest.raises(SystemExit):
        htf.main()
    assert seen["gcp_output_dir"] == "gs://b/train/old-run"
    assert seen["gcp_image"] == "reg/img@sha256:abc"
