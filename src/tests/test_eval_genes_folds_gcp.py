#!/usr/bin/env python
"""Offline tests for the GCP backend + folds.json alignment in
hound_eval_genes_folds.

No real GCS or Batch calls — staging/runner/fetch are monkeypatched and the
built job commands are captured for assertions.
"""

import json
import os

import pytest

from baskerville import dataset
from baskerville.scripts import hound_eval_genes_folds as hegf


class MockArgs:
    """Mock arguments object for eval_genes_folds (mirrors argparse defaults)."""

    def __init__(self, **kwargs):
        # eval options
        self.genes_gtf = "genes.gtf"
        self.out_dir = "models"
        self.pseudo_qtl = None
        self.rc = False
        self.save_span = False
        self.seq_step = 1
        self.shifts = "0"
        self.span = False
        self.targets_file = None
        self.valid = False
        # replication options
        self.crosses = 1
        self.conda_env = "torch2.6"
        self.fold_subset = None
        self.local = False
        self.name = "evalg"
        self.processes = 1
        self.queue = "l4"
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
    hegf.stage_cache.write_run_marker(
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
        hegf.stage_cache,
        "stage_file",
        lambda p, t, **k: ("sha", f"/workspace/cache/{t}/x"),
    )
    monkeypatch.setattr(
        hegf.stage_cache,
        "stage_dir",
        lambda p, t, **k: ("modelsha", "/workspace/cache/models/abcd"),
    )
    monkeypatch.setattr(
        hegf,
        "read_json_gcs",
        lambda uri: {f"fold{i}": {} for i in range(num_folds)},
    )
    # True for model weights (so marker mode finds them in GCS), False for the
    # gene_metrics.tsv done-check (so jobs aren't skipped as already complete).
    # fold_utils.model_present uses its own module's gcs_file_exist, so patch both.
    _model_check = lambda uri: uri.rstrip("/").endswith("model_best.pth")
    monkeypatch.setattr(hegf, "gcs_file_exist", _model_check)
    monkeypatch.setattr(hegf.fold_utils, "gcs_file_exist", _model_check)
    monkeypatch.setattr(hegf, "download_folder_from_gcs", lambda *a, **k: None)

    class FakeRunner:
        Job = FakeJob

        @staticmethod
        def multi_run(jobs, **kwargs):
            captured.extend(jobs)

    monkeypatch.setattr(hegf, "make_runner", lambda *a, **k: FakeRunner)
    monkeypatch.setattr(hegf, "announce_gcp_image", lambda *a, **k: None)
    return captured


def test_gcp_command_uses_container_paths_and_mounts(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(out_dir=out_dir)  # one evalg job per replicate (test fold)
    hegf.eval_genes_folds(args)

    # 2 replicates → one gene-eval job each
    assert len(captured) == 2
    job = captured[0]

    # bare container command — no conda activation, writes under /workspace/out
    assert "conda activate" not in job.cmd
    assert "echo $HOSTNAME" not in job.cmd
    assert job.cmd.startswith("hound_eval_genes ")
    assert " -o /workspace/out/" in job.cmd
    # container-staged params + model + gtf + mounted data, not local paths
    assert "/workspace/cache/models/abcd/f0c0/train/model_best.pth" in job.cmd
    assert "/workspace/cache/params/x" in job.cmd
    assert "/workspace/cache/gtf/x" in job.cmd
    assert "/workspace/data/hg38" in job.cmd

    # output upload + both fuse mounts (dataset + content cache) are wired up
    assert job.kwargs["output_dir_gcs"].startswith("gs://")
    mounts = job.kwargs["data_mounts"]
    assert len(mounts) == 2
    local_targets = {m.local_path for m in mounts}
    assert "/workspace/data" in local_targets
    assert hegf.stage_cache.CONTAINER_CACHE_MOUNT in local_targets


def test_default_evaluates_test_fold(tmp_path, monkeypatch):
    # training held out fold0 as test and fold3 as valid (e.g. cross>0).
    out_dir = _make_models_dir(tmp_path, num_folds=4, test_fold=0, valid_fold=3)
    captured = _patch_gcp(monkeypatch, num_folds=4)

    args = MockArgs(out_dir=out_dir, fold_subset=1)
    hegf.eval_genes_folds(args)

    assert len(captured) == 1
    assert " --split fold0" in captured[0].cmd
    assert " --split fold3" not in captured[0].cmd


def test_valid_flag_uses_valid_fold(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=4, test_fold=0, valid_fold=3)
    captured = _patch_gcp(monkeypatch, num_folds=4)

    args = MockArgs(out_dir=out_dir, fold_subset=1, valid=True)
    hegf.eval_genes_folds(args)

    assert len(captured) == 1
    assert " --split fold3" in captured[0].cmd
    assert " --split fold0" not in captured[0].cmd


def test_span_gcp_forced_to_l4_large(tmp_path, monkeypatch, capsys):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(out_dir=out_dir, span=True, queue="l4")
    hegf.eval_genes_folds(args)

    assert captured
    for job in captured:
        # span jobs forced to the high-mem GPU VM regardless of --queue
        assert job.kwargs["queue"] == "l4-large"
        assert " --span" in job.cmd

    assert "pinned to --queue l4-large" in capsys.readouterr().out


def test_no_span_honors_queue(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    captured = _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(out_dir=out_dir, span=False, queue="l4")
    hegf.eval_genes_folds(args)

    assert captured
    for job in captured:
        assert job.kwargs["queue"] == "l4"
        assert " --span" not in job.cmd


def test_local_backend_builds_command(tmp_path, monkeypatch):
    """Local (slurm/exec_par) path: absolute paths, conda activation, gene done-check."""
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)

    # a minimal local data dir with fold-labeled statistics.json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "statistics.json", "w") as f:
        json.dump({"fold0": {}, "fold1": {}}, f)

    captured = []
    monkeypatch.setattr(hegf.utils, "exec_par", lambda jobs, **k: captured.extend(jobs))

    args = MockArgs(
        out_dir=out_dir,
        data_dirs=[str(data_dir)],
        backend="slurm",
        local=True,
        gcp_data_dir=None,
    )
    hegf.eval_genes_folds(args)

    assert len(captured) == 2  # one per replicate, on its test fold
    cmd = captured[0]
    assert "conda activate" in cmd
    assert "hound_eval_genes" in cmd
    assert " --split fold0" in cmd
    assert f"{out_dir}/f0c0/evalg" in cmd  # local output dir
    assert f"{out_dir}/f0c0/train/model_best.pth" in cmd  # local weights
    assert cmd.rstrip().endswith(os.path.abspath(args.genes_gtf))  # gtf positional last


def test_gcp_requires_data_dir(tmp_path, monkeypatch):
    out_dir = _make_models_dir(tmp_path, num_folds=2, test_fold=0, valid_fold=1)
    _patch_gcp(monkeypatch, num_folds=2)
    args = MockArgs(out_dir=out_dir, gcp_data_dir=None)
    with pytest.raises(ValueError, match="gcp_data_dir"):
        hegf.eval_genes_folds(args)


def test_missing_out_dir_fails_fast(tmp_path, monkeypatch):
    _patch_gcp(monkeypatch, num_folds=2)
    args = MockArgs(out_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError, match="not found"):
        hegf.eval_genes_folds(args)


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
        hegf.stage_cache,
        "stage_dir",
        lambda *a, **k: pytest.fail("stage_dir called in marker mode"),
    )

    # marker supplies the GCP config — omit it from the CLI (region unset)
    args = MockArgs(
        out_dir=out_dir,
        gcp_data_dir=None,
        gcp_image=None,
        gcp_project=None,
        gcp_region=None,
    )
    hegf.eval_genes_folds(args)

    assert len(captured) == 2
    job = captured[0]
    # weights come from the mounted GCS dir, not the content cache
    assert "/workspace/models/f0c0/train/model_best.pth" in job.cmd
    assert "/workspace/cache/models" not in job.cmd

    # three fuse mounts: dataset + cache + GCS models
    mounts = job.kwargs["data_mounts"]
    assert len(mounts) == 3
    assert gcs_dir in {m.gcs_uri for m in mounts}
    assert hegf._GCP_MODELS_MOUNT in {m.local_path for m in mounts}


def test_marker_mode_inherits_gcp_config(tmp_path, monkeypatch):
    gcs_dir = "gs://my-bucket/output/train/abc123"
    out_dir = _make_mirrored_models_dir(
        tmp_path, num_folds=2, test_fold=0, valid_fold=1, gcs_dir=gcs_dir
    )
    _patch_gcp(monkeypatch, num_folds=2)

    args = MockArgs(
        out_dir=out_dir,
        gcp_data_dir=None,
        gcp_image=None,
        gcp_project=None,
        gcp_region=None,  # unset → inherited from marker
    )
    hegf.eval_genes_folds(args)

    assert args.gcp_data_dir == "gs://bucket/data"
    assert args.gcp_image == "img:tag"
    assert args.gcp_project == "proj"
    assert args.gcp_region == "us-west1"
