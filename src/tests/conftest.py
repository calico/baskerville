import json
import os
import pathlib
import pytest
import shutil
import subprocess

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _placeholder_gcp_config(monkeypatch):
    """gcprunner has no built-in project/buckets; give every test placeholders.

    Tests that exercise the unset case delenv these explicitly.
    """
    monkeypatch.setenv("GCPRUNNER_PROJECT", "my-gcp-project")
    monkeypatch.setenv("GCPRUNNER_CACHE_PREFIX", "gs://my-bucket/cache")
    monkeypatch.setenv("GCPRUNNER_OUTPUT_PREFIX", "gs://my-bucket/output")


def _hound_data(tmp_path_factory, name, targets_file, *extra):
    """Run hound_data on a 10% sample of sc3, returning the folds directory."""
    folds_dir = str(tmp_path_factory.mktemp(name) / "folds")
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_data",
        "--folds",
        "8",
        "--sample_pct",
        "0.1",
        "-l",
        "16384",
        "--local",
        "-o",
        folds_dir,
        "-w",
        "32",
        "--fasta",
        f"{DATA_DIR}/sc3.fa.gz",
        "--targets_cov",
        f"{DATA_DIR}/{targets_file}",
        *extra,
    ]
    subprocess.run(cmd, check=True)
    return folds_dir


def _hound_train(tmp_path_factory, name, params_file, *data_dirs):
    """Run hound_train, returning the output directory."""
    output_dir = str(tmp_path_factory.mktemp(name))
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        str(params_file),
        *data_dirs,
    ]
    subprocess.run(cmd, check=True)
    return output_dir


@pytest.fixture(scope="session")
def data_me_dir(tmp_path_factory):
    folds_dir = _hound_data(tmp_path_factory, "data_me", "targets_sc3_me.txt")
    tvt_dir = str(pathlib.Path(folds_dir).parent / "tvt")
    make_tvt(folds_dir, tvt_dir)
    return tvt_dir


@pytest.fixture(scope="session")
def data_ac_dir(tmp_path_factory):
    folds_dir = _hound_data(tmp_path_factory, "data_ac", "targets_sc3_ac.txt")
    tvt_dir = str(pathlib.Path(folds_dir).parent / "tvt")
    make_tvt(folds_dir, tvt_dir)
    return tvt_dir


@pytest.fixture(scope="session")
def data_gene_folds_dir(tmp_path_factory):
    """Create test data with gene expression, yielding raw folds directory."""
    return _hound_data(
        tmp_path_factory,
        "data_gene",
        "targets_sc3_me.txt",
        "--gtf",
        f"{DATA_DIR}/sc3_genes.gtf",
        "--targets_gene",
        f"{DATA_DIR}/targets_sc3_gene.txt",
    )


@pytest.fixture(scope="session")
def data_gene_dir(data_gene_folds_dir):
    """Convert folds dataset to train/valid/test split."""
    tvt_dir = str(pathlib.Path(data_gene_folds_dir).parent / "tvt")
    make_tvt_gene(data_gene_folds_dir, tvt_dir)
    return tvt_dir


@pytest.fixture(scope="session")
def static_model_dir():
    """Use the pre-trained static model for faster tests."""
    assert (DATA_DIR / "sc3_model.pth").exists(), "Static model file not found"
    return str(DATA_DIR)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, data_me_dir):
    return _hound_train(
        tmp_path_factory, "sc3_model", DATA_DIR / "params_sc3.json", data_me_dir
    )


@pytest.fixture(scope="session")
def transfer_dir(tmp_path_factory, data_ac_dir, static_model_dir):
    with open(DATA_DIR / "params_sc3.json") as f:
        params = json.load(f)
    params["model"]["pretrained_model"] = f"{static_model_dir}/sc3_model.pth"
    params_file = tmp_path_factory.mktemp("params") / "params_sc3_ac.json"
    with open(params_file, "w") as f:
        json.dump(params, f)
    return _hound_train(tmp_path_factory, "sc3_transfer", params_file, data_ac_dir)


@pytest.fixture(scope="session")
def model_gene_dir(tmp_path_factory, data_gene_dir):
    """Train model with both coverage and gene heads."""
    return _hound_train(
        tmp_path_factory,
        "sc3_model_gene",
        DATA_DIR / "params_sc3_gene.json",
        data_gene_dir,
    )


@pytest.fixture(scope="session")
def model_gene_only_dir(tmp_path_factory, data_gene_dir):
    """Train model with only gene head (no coverage)."""
    return _hound_train(
        tmp_path_factory,
        "sc3_model_gene_only",
        DATA_DIR / "params_sc3_gene_only.json",
        data_gene_dir,
    )


@pytest.fixture(scope="session")
def model_cov_with_gene_data_dir(tmp_path_factory, data_gene_dir):
    """Train coverage-only model on dataset that also has gene data (backward compat)."""
    return _hound_train(
        tmp_path_factory,
        "sc3_model_cov_with_gene_data",
        DATA_DIR / "params_sc3.json",
        data_gene_dir,
    )


@pytest.fixture(scope="session")
def model_gene_mixed_dir(tmp_path_factory, data_gene_dir, data_me_dir):
    """Train a gene head on dataset 0 only, beside a coverage-only dataset 1."""
    return _hound_train(
        tmp_path_factory,
        "sc3_model_gene_mixed",
        DATA_DIR / "params_sc3_gene_mixed.json",
        data_gene_dir,
        data_me_dir,
    )


def make_tvt_gene(folds_dir, tvt_dir):
    """
    Create train/valid/test splits for data with gene expression.

    Args:
        folds_dir (str): Path to the data directory.
        tvt_dir (str): Path to the output directory for train/valid/test splits.
    """
    # First do the standard tvt split
    make_tvt(folds_dir, tvt_dir)

    # Copy gene-related files
    if os.path.exists(f"{folds_dir}/targets_gene.txt"):
        shutil.copy(f"{folds_dir}/targets_gene.txt", f"{tvt_dir}/targets_gene.txt")
    if os.path.exists(f"{folds_dir}/genes.txt"):
        shutil.copy(f"{folds_dir}/genes.txt", f"{tvt_dir}/genes.txt")
    if os.path.exists(f"{folds_dir}/seqs_gene"):
        shutil.copytree(f"{folds_dir}/seqs_gene", f"{tvt_dir}/seqs_gene")


def make_tvt(folds_dir, tvt_dir):
    """
    Create train/valid/test splits for the given data directory.

    Args:
        folds_dir (str): Path to the data directory.
        tvt_dir (str): Path to the output directory for train/valid/test splits.
    """
    # read data parameters
    data_stats_file = f"{folds_dir}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # sequences per fold
    fold_seqs = []
    dfi = 0
    fold_label = f"fold{dfi}_seqs"
    while fold_label in data_stats:
        fold_seqs.append(data_stats[fold_label])
        del data_stats[fold_label]
        dfi += 1
        fold_label = f"fold{dfi}_seqs"
    num_folds = dfi

    # split folds into train/valid/test
    test_fold = 0
    valid_fold = 1
    train_folds = [
        fold for fold in range(num_folds) if fold not in [valid_fold, test_fold]
    ]

    os.makedirs(tvt_dir)

    # dump data stats
    data_stats["test_seqs"] = fold_seqs[test_fold]
    data_stats["valid_seqs"] = fold_seqs[valid_fold]
    data_stats["train_seqs"] = sum([fold_seqs[tf] for tf in train_folds])
    for split in ["train", "valid", "test"]:
        assert data_stats[f"{split}_seqs"] > 0, f"Empty {split} split in {folds_dir}"
    with open(f"{tvt_dir}/statistics.json", "w") as data_stats_open:
        json.dump(data_stats, data_stats_open, indent=4)

    # set sequence tvt
    seqs_bed_out = open(f"{tvt_dir}/sequences.bed", "w")
    for line in open(f"{folds_dir}/sequences.bed"):
        a = line.split()
        sfi = int(a[-1].replace("fold", ""))
        if sfi == test_fold:
            a[-1] = "test"
        elif sfi == valid_fold:
            a[-1] = "valid"
        else:
            a[-1] = "train"
        print("\t".join(a), file=seqs_bed_out)
    seqs_bed_out.close()

    # copy targets
    shutil.copy(f"{folds_dir}/targets.txt", f"{tvt_dir}/targets.txt")

    # sym link tfrecords
    data_examples_dir = f"{folds_dir}/examples"
    tvt_examples_dir = f"{tvt_dir}/examples"
    os.mkdir(tvt_examples_dir)

    # test examples
    data_test_dir = f"{data_examples_dir}/fold{test_fold}.zarr"
    tvt_test_dir = f"{tvt_examples_dir}/test.zarr"
    os.symlink(data_test_dir, tvt_test_dir)

    # valid examples
    data_valid_dir = f"{data_examples_dir}/fold{valid_fold}.zarr"
    tvt_valid_dir = f"{tvt_examples_dir}/valid.zarr"
    os.symlink(data_valid_dir, tvt_valid_dir)

    # train examples
    for tfi in train_folds:
        data_train_dir = f"{data_examples_dir}/fold{tfi}.zarr"
        tvt_train_dir = f"{tvt_examples_dir}/train{tfi}.zarr"
        os.symlink(data_train_dir, tvt_train_dir)
