import os
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd


def epoch_stats(epoch_line):
    estats = {}
    for kv in epoch_line.split(" - ")[1:]:
        if kv.count(":") == 1:
            k, v = kv.split(":")
            estats[k.strip()] = float(v)
    return estats


def verify_training(
    model_dir,
    expect_coverage=True,
    expect_gene=False,
):
    """Verify training completed successfully.

    Args:
        model_dir: Directory containing log.txt and model_best.pth
        expect_coverage: Whether to expect coverage metrics (train_r, valid_r)
        expect_gene: Whether to expect gene metrics (train_r_gene, valid_r_gene)
    """
    train_log_file = f"{model_dir}/log.txt"
    epoch_num = 0

    for line in open(train_log_file):
        line = line.strip()
        if line.startswith("Data"):
            epoch_num += 1
            estats = epoch_stats(line)

            # Always check loss
            assert not np.isnan(estats["train_loss"])

            # Check coverage metrics
            if expect_coverage:
                assert "train_r" in estats, "Missing coverage metrics in log"
                assert not np.isnan(estats["train_r"])
                assert not np.isnan(estats["valid_loss"])
                assert not np.isnan(estats["valid_r"])
            else:
                assert "train_r" not in estats, "Unexpected coverage metrics"

            # Check gene metrics
            if expect_gene:
                assert "train_r_gene" in estats, "Missing gene metrics in log"
                assert not np.isnan(estats["train_r_gene"])
                assert not np.isnan(estats["valid_r_gene"])
            else:
                assert "train_r_gene" not in estats, "Unexpected gene metrics"

    assert epoch_num >= 1, "No 'Data' line found in log"

    # Check the per-epoch metrics table
    metrics_df = pd.read_csv(f"{model_dir}/metrics.tsv", sep="\t")
    assert {"train", "valid"} <= set(metrics_df.split)
    metric = "r" if expect_coverage else "loss"
    assert ((metrics_df.group == "all") & (metrics_df.metric == metric)).any()

    # Check model file exists
    model_file = f"{model_dir}/model_best.pth"
    assert os.path.exists(model_file)


def test_train(model_dir):
    """Test basic training with coverage data."""
    verify_training(model_dir, expect_coverage=True, expect_gene=False)


def test_transfer(transfer_dir):
    """Test transfer learning (single epoch)."""
    verify_training(transfer_dir, expect_coverage=True, expect_gene=False)


def test_train_distill(static_model_dir, data_me_dir):
    """Test distillation training using a trained model as teacher."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    output_dir = f"{test_dir}/data/sc3_distill"

    # Create teachers directory structure (mimic hound_train_folds output)
    teachers_dir = f"{test_dir}/data/sc3_teachers"
    if os.path.exists(teachers_dir):
        shutil.rmtree(teachers_dir)
    os.makedirs(f"{teachers_dir}/f0c0/train", exist_ok=True)

    # Copy model and params to teacher location
    shutil.copy(
        f"{static_model_dir}/sc3_model.pth", f"{teachers_dir}/f0c0/train/model_best.pth"
    )
    shutil.copy(params_file, f"{teachers_dir}/f0c0/params.json")

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Run distillation training
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train_distill",
        "-o",
        output_dir,
        params_file,
        teachers_dir,
        data_me_dir,
    ]
    subprocess.run(cmd, check=True)

    # Verify training (distillation has different log format, check manually)
    train_log_file = f"{output_dir}/log.txt"
    found_data_line = False
    for line in open(train_log_file):
        line = line.strip()
        if line.startswith("Data"):
            found_data_line = True
            estats = epoch_stats(line)
            assert not np.isnan(estats["train_loss"]), "train_loss is NaN"
            assert not np.isnan(estats["train_r"]), "train_r is NaN"

    assert found_data_line, "No 'Data' line found in log"
    model_file = f"{output_dir}/model_best.pth"
    assert os.path.exists(model_file), f"Model file not found: {model_file}"

    # Cleanup
    if os.path.exists(teachers_dir):
        shutil.rmtree(teachers_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_train_gene(model_gene_dir):
    """Test training with both coverage and gene expression heads."""
    verify_training(model_gene_dir, expect_coverage=True, expect_gene=True)


def test_train_gene_only(model_gene_only_dir):
    """Test training with only gene expression head (no coverage)."""
    verify_training(model_gene_only_dir, expect_coverage=False, expect_gene=True)


def test_train_gene_mixed(model_gene_mixed_dir):
    """Test a gene head on dataset 0 beside a coverage-only dataset 1."""
    seen = set()
    for line in open(f"{model_gene_mixed_dir}/log.txt"):
        line = line.strip()
        if line.startswith("Data"):
            seen.add(line.split()[1])
            estats = epoch_stats(line)
            assert not np.isnan(estats["train_loss"])
            assert not np.isnan(estats["valid_r"])
            if line.startswith("Data 0"):
                assert not np.isnan(estats["valid_r_gene"])
            else:
                assert "train_r_gene" not in estats, "Unexpected gene metrics"
    assert seen == {"0", "1"}
    assert os.path.exists(f"{model_gene_mixed_dir}/model_best.pth")


def test_train_cov_only_with_gene_data(model_cov_with_gene_data_dir):
    """Test training coverage-only model on dataset with gene data (backward compat)."""
    verify_training(
        model_cov_with_gene_data_dir, expect_coverage=True, expect_gene=False
    )
