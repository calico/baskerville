import json
import os
import pathlib
import pytest
import shutil
import subprocess

DEBUG = 0


@pytest.fixture(autouse=True)
def _placeholder_gcp_config(monkeypatch):
    """gcprunner has no built-in project/buckets; give every test placeholders.

    Tests that exercise the unset case delenv these explicitly.
    """
    monkeypatch.setenv("GCPRUNNER_PROJECT", "my-gcp-project")
    monkeypatch.setenv("GCPRUNNER_CACHE_PREFIX", "gs://my-bucket/cache")
    monkeypatch.setenv("GCPRUNNER_OUTPUT_PREFIX", "gs://my-bucket/output")


@pytest.fixture(scope="session")
def data_me_dir():
    # inputs
    data_files_dir = str(pathlib.Path(__file__).parent / "data")
    fasta_file = f"{data_files_dir}/sc3.fa.gz"
    targets_file = f"{data_files_dir}/targets_sc3_me.txt"
    folds_dir = f"{data_files_dir}/data_folds"
    tvt_dir = f"{data_files_dir}/data_tvt"

    # clean
    if os.path.exists(folds_dir):
        shutil.rmtree(folds_dir)

    # hound_data
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_data",
        "--folds",
        "8",
        "-l",
        "16384",
        "--local",
        "-o",
        folds_dir,
        "-w",
        "32",
        "--fasta",
        fasta_file,
        "--targets_cov",
        targets_file,
    ]
    subprocess.run(cmd, check=True)

    # split into train/valid/test
    make_tvt(folds_dir, tvt_dir)

    yield tvt_dir

    if not DEBUG:
        # cleanup
        if os.path.exists(folds_dir):
            shutil.rmtree(folds_dir)
        if os.path.exists(tvt_dir):
            shutil.rmtree(tvt_dir)


@pytest.fixture(scope="session")
def data_ac_dir():
    # inputs
    data_files_dir = str(pathlib.Path(__file__).parent / "data")
    fasta_file = f"{data_files_dir}/sc3.fa.gz"
    targets_file = f"{data_files_dir}/targets_sc3_ac.txt"
    folds_dir = f"{data_files_dir}/data_folds"
    tvt_dir = f"{data_files_dir}/data_tvt"

    # clean
    if os.path.exists(folds_dir):
        shutil.rmtree(folds_dir)

    # hound_data
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_data",
        "--folds",
        "8",
        "-l",
        "16384",
        "--local",
        "-o",
        folds_dir,
        "-w",
        "32",
        "--fasta",
        fasta_file,
        "--targets_cov",
        targets_file,
    ]
    subprocess.run(cmd, check=True)

    # split into train/valid/test
    make_tvt(folds_dir, tvt_dir)

    yield tvt_dir

    if not DEBUG:
        # cleanup
        if os.path.exists(folds_dir):
            shutil.rmtree(folds_dir)
        if os.path.exists(tvt_dir):
            shutil.rmtree(tvt_dir)


@pytest.fixture(scope="session")
def model_dir(data_me_dir):
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    output_dir = f"{test_dir}/data/sc3_model"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_file,
        data_me_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        # cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def static_model_dir():
    """Use the pre-trained static model for faster tests."""
    test_dir = str(pathlib.Path(__file__).parent)
    model_dir = f"{test_dir}/data"

    # Verify the static model file exists
    model_file = f"{model_dir}/sc3_model.pth"
    assert os.path.exists(model_file), f"Static model file not found: {model_file}"

    return model_dir


@pytest.fixture(scope="session")
def transfer_dir(data_ac_dir, model_dir):
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    params_transfer_file = f"{test_dir}/data/params_sc3_ac.json"
    output_dir = f"{test_dir}/data/sc3_transfer"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # copy params file, adding pretained model path
    model_file = f"{model_dir}/model_best.pth"
    params_pretrained(params_file, params_transfer_file, model_file)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_transfer_file,
        data_ac_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        # cleanup
        os.remove(params_transfer_file)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def params_pretrained(params_file, params_transfer_file, pretrained_model):
    """Copy params file, adding pretained model path.

    Args:
        params_file (str): Path to the original params file.
        params_transfer_file (str): Path to the transfer params file.
        pretrained_model (str): Path to the pretrained model, if any.
    """
    with open(params_transfer_file, "w") as params_transfer_open:
        for line in open(params_file):
            print(line, file=params_transfer_open, end="")
            if line.strip() == '"model": {':
                if pretrained_model is not None:
                    print(
                        f'        "pretrained_model": "{pretrained_model}",',
                        file=params_transfer_open,
                    )
            elif line.split()[0] == '"train_epochs_max":':
                print(f'        "train_epochs_max": 1,', file=params_transfer_open)


@pytest.fixture(scope="session")
def model_gene_dir(data_gene_dir):
    """Train model with both coverage and gene heads."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3_gene.json"
    output_dir = f"{test_dir}/data/sc3_model_gene"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_file,
        data_gene_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def model_gene_only_dir(data_gene_dir):
    """Train model with only gene head (no coverage)."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3_gene_only.json"
    output_dir = f"{test_dir}/data/sc3_model_gene_only"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_file,
        data_gene_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def model_cov_with_gene_data_dir(data_gene_dir):
    """Train coverage-only model on dataset that also has gene data (backward compat)."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"  # Coverage-only params
    output_dir = f"{test_dir}/data/sc3_model_cov_with_gene_data"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_file,
        data_gene_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def model_gene_mixed_dir(data_gene_dir, data_me_dir):
    """Train a gene head on dataset 0 only, beside a coverage-only dataset 1."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3_gene_mixed.json"
    output_dir = f"{test_dir}/data/sc3_model_gene_mixed"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_train",
        "-o",
        output_dir,
        params_file,
        data_gene_dir,
        data_me_dir,
    ]
    subprocess.run(cmd, check=True)

    yield output_dir

    if not DEBUG:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def data_gene_dir(data_gene_folds_dir):
    """Convert folds dataset to train/valid/test split."""
    data_files_dir = str(pathlib.Path(__file__).parent / "data")
    tvt_dir = f"{data_files_dir}/data_tvt_gene"

    # clean
    if os.path.exists(tvt_dir):
        shutil.rmtree(tvt_dir)

    # split into train/valid/test
    make_tvt_gene(data_gene_folds_dir, tvt_dir)

    yield tvt_dir

    if not DEBUG:
        # cleanup tvt_dir only (folds cleaned by parent fixture)
        if os.path.exists(tvt_dir):
            shutil.rmtree(tvt_dir)


@pytest.fixture(scope="session")
def data_gene_folds_dir():
    """Create test data with gene expression, yielding raw folds directory."""
    # inputs
    data_files_dir = str(pathlib.Path(__file__).parent / "data")
    fasta_file = f"{data_files_dir}/sc3.fa.gz"
    targets_file = f"{data_files_dir}/targets_sc3_me.txt"
    targets_gene_file = f"{data_files_dir}/targets_sc3_gene.txt"
    gtf_file = f"{data_files_dir}/sc3_genes.gtf"
    folds_dir = f"{data_files_dir}/data_folds_gene"

    # clean
    if os.path.exists(folds_dir):
        shutil.rmtree(folds_dir)

    # hound_data with gene options
    cmd = [
        "python",
        "-m",
        "baskerville.scripts.hound_data",
        "--folds",
        "8",
        "-l",
        "16384",
        "--local",
        "-o",
        folds_dir,
        "-w",
        "32",
        "--fasta",
        fasta_file,
        "--targets_cov",
        targets_file,
        "--gtf",
        gtf_file,
        "--targets_gene",
        targets_gene_file,
    ]
    subprocess.run(cmd, check=True)

    yield folds_dir

    if not DEBUG:
        # cleanup
        if os.path.exists(folds_dir):
            shutil.rmtree(folds_dir)


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

    # clear existing directory
    if os.path.isdir(tvt_dir):
        shutil.rmtree(tvt_dir)

    # make data directory
    os.makedirs(tvt_dir, exist_ok=True)

    # dump data stats
    data_stats["test_seqs"] = fold_seqs[test_fold]
    data_stats["valid_seqs"] = fold_seqs[valid_fold]
    data_stats["train_seqs"] = sum([fold_seqs[tf] for tf in train_folds])
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
