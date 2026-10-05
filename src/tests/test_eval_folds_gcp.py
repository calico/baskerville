#!/usr/bin/env python
"""Offline tests for the GCP backend + folds.json alignment in hound_eval_folds.

No real GCS or Batch calls — staging/runner/fetch are monkeypatched and the
built job commands are captured for assertions.
"""

import json
import os

import pytest

from baskerville import dataset
from baskerville.scripts import hound_eval_folds as hef


class MockArgs:
    """Mock arguments object for eval_folds (mirrors the argparse defaults)."""

    def __init__(self, **kwargs):
        # eval options
        self.aggregate_genes = False
        self.out_dir = "models"
        self.rank_corr = False
        self.rc = False
        self.save = False
        self.shifts = "0"
        self.step = 1
        self.test_only = False
        self.valid_only = False
        self.targets_gene_file = None
        # spec streaming knobs (hound_eval_spec pass-through)
        self.band = None
        self.seq_chunk = None
        self.ram = False
        self.scratch_dir = None
        # replication options
        self.crosses = 1
        self.conda_env = "torch2.6"
        self.fold_subset = None
        self.name = "fold"
        self.processes = 1
        self.queue = "l4"
        self.restart = False
        self.spec = False
        self.params_file = "params.json"
        self.data_dirs = ["/data/hg38"]
        # runner backend (add_argparse_group)
        self.backend = "gcp"
        self.gcp_project = None
        self.gcp_region = None
        self.gcp_zone = None
        self.gcp_image = None
        self.gcp_output_dir = None
        self.gcp_data_dir = "gs://bucket/data"
        self.gcp_data_local = "/workspace/data"
        self.gcp_stage_dir = None
        self.gcp_stage_local = "/workspace/stage"
        self.gcp_provisioning = "standard"
        self.gcp_service_account = None
        self.gcp_fetch_output = ""  # skip local fetch in tests

        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeJob:
    """Captures the cmd + kwargs each Job() call was built with."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs


def _make_models_dir(base, num_folds, test_fold, valid_fold, crosses=1):
    """Create f{fi}c{ci}/train/{model_best.pth,folds.json} under base/models."""
    out_dir = os.path.join(base, "models")
    for ci in range(crosses):
        for fi in range(num_folds):
            train_dir = os.path.join(out_dir, f"f{fi}c{ci}", "train")
            os.makedirs(train_dir, exist_ok=True)
            open(os.path.join(train_dir, "model_best.pth"), "wb").close()
            with open(os.path.join(train_dir, "folds.json"), "w") as f:
                json.dump(
                    {
                        "fold": fi,
                        "cross": ci,
                        "num_folds": num_folds,
                        "test_fold": test_fold,
                        "valid_fold": valid_fold,
                        "train_folds": [
                            i
                            for i in range(num_folds)
                            if i not in (test_fold, valid_fold)
                        ],
                    },
                    f,
                )
    return out_dir


def _make_mirrored_models_dir(
    base, num_folds, test_fold, valid_fold, gcs_dir, crosses=1
):
    """Mimic a GCP-trained tree mirrored back locally: folds.json + a gcp_run.json
    marker, but NO model_best.pth (the train mirror excludes .pth)."""
    out_dir = os.path.join(base, "models")
    for ci in range(crosses):
        for fi in range(num_folds):
            train_dir = os.path.join(out_dir, f"f{fi}c{ci}", "train")
            os.makedirs(train_dir, exist_ok=True)
            with open(os.path.join(train_dir, "folds.json"), "w") as f:
                json.dump(
                    {
                        "fold": fi,
                        "cross": ci,
                        "num_folds": num_folds,
                        "test_fold": test_fold,
                        "valid_fold": valid_fold,
                    },
                    f,
                )
    hef.stage_cache.write_run_marker(
        out_dir,
        backend="gcp",
        output_dir_gcs=gcs_dir,
        gcp_data_dir="gs://bucket/data",
        gcp_image="img:tag",
        gcp_project="proj",
        gcp_region="us-west1",
        gcp_zone=None,
    )
    return out_dir


def _patch_gcp(monkeypatch, num_folds):
    """Stub out all GCS/staging/runner calls; return the captured-jobs list."""
    captured = []

    monkeypatch.setattr(
        hef.stage_cache,
        "stage_file",
        lambda p, t, **k: ("sha", f"/workspace/cache/{t}/x"),
    )
    monkeypatch.setattr(
        hef.stage_cache,
        "stage_dir",
        lambda p, t, **k: ("modelsha", "/workspace/cache/models/abcd"),
    )
    monkeypatch.setattr(
        hef,
        "read_json_gcs",
        lambda uri: {f"fold{i}": {} for i in range(num_folds)},
    )
    # True for model weights (so marker mode finds them in GCS), False for the
    # acc.txt done-check (so jobs aren't skipped as already complete).
    # fold_utils.model_present uses its own module's gcs_file_exist, so patch both.
    _model_check = lambda uri: uri.rstrip("/").endswith("model_best.pth")
    monkeypatch.setattr(hef, "gcs_file_exist", _model_check)
    monkeypatch.setattr(hef.fold_utils, "gcs_file_exist", _model_check)
    monkeypatch.setattr(hef, "download_folder_from_gcs", lambda *a, **k: None)

    class FakeRunner:
        Job = FakeJob

        @staticmethod
        def multi_run(jobs, **kwargs):
            captured.extend(jobs)

    monkeypatch.setattr(hef, "make_runner", lambda *a, **k: FakeRunner)
    monkeypatch.setattr(hef, "announce_gcp_image", lambda *a, **k: None)
    return captured


def test_gcp_command_uses_container_paths_and_mounts(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(out_dir=out_dir, test_only=True)  # one eval job per replicate
    hef.eval_folds(args)

    # 2 folds × test_only → one job each (ei == test_fold)
    assert len(captured) == 2
    job = captured[0]

    # bare container command — no conda activation, writes under /workspace/out
    assert "conda activate" not in job.cmd
    assert "echo $HOSTNAME" not in job.cmd
    assert job.cmd.startswith("hound_eval ")
    assert " -o /workspace/out/" in job.cmd
    # container-staged params + model + mounted data, not local paths
    assert "/workspace/cache/models/abcd/f0c0/train/model_best.pth" in job.cmd
    assert "/workspace/cache/params/x" in job.cmd
    assert "/workspace/data/hg38" in job.cmd

    # output upload + both fuse mounts (dataset + content cache) are wired up
    assert job.kwargs["output_dir_gcs"].startswith("gs://")
    mounts = job.kwargs["data_mounts"]
    assert len(mounts) == 2
    local_targets = {m.local_path for m in mounts}
    assert "/workspace/data" in local_targets
    assert hef.stage_cache.CONTAINER_CACHE_MOUNT in local_targets

    # run-identity labels: kind=eval, run == basename of the output dir, fold set
    labels = job.kwargs["labels"]
    assert labels["gcprunner_kind"] == "eval"
    assert labels["gcprunner_run"] == os.path.basename(job.kwargs["output_dir_gcs"])
    assert labels["gcprunner_fold"].startswith("f")


def test_valid_only_honors_folds_json(tmp_path, monkeypatch):
    # 4 folds, training held out fold0 as test and fold3 as valid (e.g. cross>0).
    # The old convention (valid == fi+1 == 1) would pick the wrong fold.
    out_dir = _make_models_dir(tmp_path, num_folds=4, test_fold=0, valid_fold=3)
    captured = _patch_gcp(monkeypatch, num_folds=4)

    args = MockArgs(out_dir=out_dir, fold_subset=1, valid_only=True)
    hef.eval_folds(args)

    assert len(captured) == 1
    assert " --split fold3" in captured[0].cmd
    assert " --split fold1" not in captured[0].cmd


def test_default_evaluates_all_folds(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=3, test_fold=0, valid_fold=2)
    captured = _patch_gcp(monkeypatch, num_folds=3)

    args = MockArgs(out_dir=out_dir, fold_subset=1)  # one replicate, all eval folds
    hef.eval_folds(args)

    splits = sorted(c.cmd.split(" --split fold")[1].split()[0] for c in captured)
    assert splits == ["0", "1", "2"]


def test_spec_gcp_uses_queue(tmp_path, monkeypatch, capsys):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(
        out_dir=out_dir, fold_subset=1, spec=True, test_only=True, queue="l4"
    )
    hef.eval_folds(args)

    spec_jobs = [j for j in captured if j.cmd.startswith("hound_eval_spec")]
    assert spec_jobs
    for j in spec_jobs:
        # spec now streams (a few GB), so it honors --queue like coverage eval
        assert j.kwargs["queue"] == "l4"
        assert j.kwargs["cpu"] == 8
        assert " --ncpus 8" in j.cmd
        # bumped boot disk: the streamed store is tight on the 100 GiB default
        assert j.kwargs["boot_disk_gb"] == 200

    # coverage eval jobs also honor the user's --queue
    eval_jobs = [j for j in captured if j.cmd.startswith("hound_eval ")]
    assert eval_jobs
    for j in eval_jobs:
        assert j.kwargs["queue"] == "l4"

    # no auto-pin note is emitted anymore
    assert "pinned to --queue" not in capsys.readouterr().out


def test_read_fold_splits_from_json(tmp_path):
    train_dir = tmp_path / "f0c1" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "folds.json").write_text(
        json.dumps({"test_fold": 0, "valid_fold": 2, "num_folds": 4})
    )
    assert hef._read_fold_splits(str(train_dir), 4, 0, 1) == (0, 2)


def test_read_fold_splits_fallback_warns(tmp_path, capsys):
    train_dir = tmp_path / "f1c0" / "train"
    train_dir.mkdir(parents=True)  # no folds.json

    test_fold, valid_fold = hef._read_fold_splits(str(train_dir), 4, 1, 0)

    expected = dataset.compute_fold_splits(4, 1, 0)
    assert (test_fold, valid_fold) == (expected["test"], expected["valid"])
    assert "not found" in capsys.readouterr().out


def test_gcp_requires_data_dir(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    _patch_gcp(monkeypatch, num_folds=2)
    args = MockArgs(out_dir=out_dir, gcp_data_dir=None)
    with pytest.raises(ValueError, match="gcp_data_dir"):
        hef.eval_folds(args)


# --------------------------------------------------------------------------
# marker mode: GCP-trained models read straight from GCS (no staging)
# --------------------------------------------------------------------------


def test_marker_mode_reads_gcs_without_staging(tmp_path, monkeypatch):
    gcs_dir = "gs://my-bucket/output/train/deadbeef"
    out_dir = _make_mirrored_models_dir(
        tmp_path, num_folds=2, test_fold=0, valid_fold=1, gcs_dir=gcs_dir
    )
    captured = _patch_gcp(monkeypatch, num_folds=2)
    # staging the models tree must NOT happen in marker mode
    monkeypatch.setattr(
        hef.stage_cache,
        "stage_dir",
        lambda *a, **k: pytest.fail("stage_dir called in marker mode"),
    )

    # marker supplies the GCP config — omit it from the CLI (region unset)
    args = MockArgs(
        out_dir=out_dir,
        test_only=True,
        gcp_data_dir=None,
        gcp_image=None,
        gcp_project=None,
        gcp_region=None,
    )
    hef.eval_folds(args)

    assert len(captured) == 2
    job = captured[0]
    # weights come from the mounted GCS dir, not the content cache
    assert "/workspace/models/f0c0/train/model_best.pth" in job.cmd
    assert "/workspace/cache/models" not in job.cmd

    # three fuse mounts: dataset + cache + GCS models
    mounts = job.kwargs["data_mounts"]
    assert len(mounts) == 3
    assert gcs_dir in {m.gcs_uri for m in mounts}
    assert hef.fold_utils.GCP_MODELS_MOUNT in {m.local_path for m in mounts}


def test_marker_mode_inherits_gcp_config(tmp_path, monkeypatch):
    gcs_dir = "gs://my-bucket/output/train/abc123"
    out_dir = _make_mirrored_models_dir(
        tmp_path, num_folds=2, test_fold=0, valid_fold=1, gcs_dir=gcs_dir
    )
    _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(
        out_dir=out_dir,
        test_only=True,
        gcp_data_dir=None,
        gcp_image=None,
        gcp_project=None,
        gcp_region=None,  # unset → inherited from marker
    )
    hef.eval_folds(args)

    assert args.gcp_data_dir == "gs://bucket/data"
    assert args.gcp_image == "img:tag"
    assert args.gcp_project == "proj"
    assert args.gcp_region == "us-west1"


def test_marker_mode_explicit_flag_wins(tmp_path, monkeypatch):
    gcs_dir = "gs://my-bucket/output/train/abc123"
    out_dir = _make_mirrored_models_dir(
        tmp_path, num_folds=2, test_fold=0, valid_fold=1, gcs_dir=gcs_dir
    )
    _patch_gcp(monkeypatch, num_folds=2)

    # explicitly-passed flags must not be overwritten by the marker. region now
    # defaults to None, so an explicit value is distinguishable and wins too.
    args = MockArgs(
        out_dir=out_dir,
        test_only=True,
        gcp_data_dir="gs://other/data",
        gcp_image=None,
        gcp_project=None,
        gcp_region="europe-west4",
    )
    hef.eval_folds(args)

    assert args.gcp_data_dir == "gs://other/data"
    assert args.gcp_region == "europe-west4"


def test_run_marker_roundtrip(tmp_path):
    out_dir = str(tmp_path / "models")
    path = hef.stage_cache.write_run_marker(
        out_dir, backend="gcp", output_dir_gcs="gs://b/run", gcp_zone=None
    )
    assert path.endswith(hef.stage_cache.RUN_MARKER_NAME)
    marker = hef.stage_cache.read_run_marker(out_dir)
    assert marker["backend"] == "gcp"
    assert marker["output_dir_gcs"] == "gs://b/run"
    assert marker["gcp_zone"] is None
    # absent marker → None
    assert hef.stage_cache.read_run_marker(str(tmp_path / "nope")) is None


def _gcp_output_dir(tmp_path, monkeypatch, **kwargs):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)
    hef.eval_folds(MockArgs(out_dir=out_dir, test_only=True, **kwargs))
    return captured[0].kwargs["output_dir_gcs"]


def test_gcp_output_dir_keyed_on_options(tmp_path, monkeypatch):
    """Output-changing options start fresh; memory/IO knobs still resume."""
    base = _gcp_output_dir(tmp_path, monkeypatch)
    assert _gcp_output_dir(tmp_path, monkeypatch) == base
    assert _gcp_output_dir(tmp_path, monkeypatch, rc=True) != base
    assert _gcp_output_dir(tmp_path, monkeypatch, shifts="0,1") != base
    assert _gcp_output_dir(tmp_path, monkeypatch, band=8, ram=True) == base
    # adding --spec to a finished eval run resumes its eval jobs
    assert _gcp_output_dir(tmp_path, monkeypatch, spec=True) == base
