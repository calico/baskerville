# Cross-fold Training and Evaluation

## Introduction

This guide demonstrates how to train and evaluate Borzoi-like regulatory sequence machine learning models using cross-fold validation. Cross-fold validation is helpful for robust model evaluation.

## Overview

Cross-fold validation involves:

1. Creating a dataset divided into multiple folds
2. Training models where each fold serves as the test set
3. Evaluating all models across all folds
4. Aggregating results for comprehensive performance assessment

## Dataset Construction with Folds

### Basic Cross-fold Dataset

To create a dataset with cross-fold validation, use the `-f` option with `hound_data` to specify the number of folds:

```bash
hound_data -l 16384 --local --stride 4109 -f 6 -o data_folds src/tests/data/sc3.fa.gz -w 32 src/tests/data/targets_sc3_me.txt
```

This command creates a dataset with 6 folds, instead of the traditional train/valid/test split. Key parameters:

- `-f 6`: Creates 6 cross-validation folds
- `-l 16384`: Sequence length (appropriate for yeast)
- `--stride 4109`: Stride for sequence sampling (~1/4 sequence length + 15bp)
- `-w 32`: Bin width for target aggregation
- `--local`: Run data processing jobs locally

### Understanding Fold Structure

When using folds, the dataset structure changes:

- **Traditional**: `train`, `valid`, `test` splits
- **Cross-fold**: `fold0`, `fold1`, `fold2`, `fold3`, `fold4`, `fold5` (for 6 folds)

Each fold contains roughly equal amounts of genomic sequence, and the division respects genomic context to avoid data leakage.

### Advanced Options

You can combine fold creation with other data processing options:

```bash
# With blacklist and unmappable regions
hound_data -l 16384 --local --stride 4109 -f 5 \
  -b blacklist.bed -u unmappable.bed --umap_clip 0.5 \
  -o data_folds src/tests/data/sc3.fa.gz -w 32 src/tests/data/targets_sc3_me.txt
```

## Cross-fold Training

### Model Architecture

Define your model architecture in JSON format, same as single training:

```json
{
  "train": {
    "batch_size": 16,
    "optimizer": "adamw",
    "beta1": 0.9,
    "beta2": 0.99,
    "learning_rate": 0.02,
    "warmup_steps": 4,
    "train_epochs_max": 100,
    "weight_decay": 1.0e-2,
    "global_clipnorm": 1.0,
    "loss": "poisson_mn",
    "total_weight": 0.25,
    "weight_exp": 8,
    "weight_range": 10,
    "mix_dtype": "bfloat16",
    "num_workers": 4
  },
  "model": {
    "seq_length": 16384,
    "trunk": [
      {
        "name": "ConvDNA",
        "out_channels": 64,
        "kernel_size": 7,
        "pool_size": 2
      },
      {
        "name": "ConvTower",
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "divisible_by": 16,
        "act_func": "silu",
        "norm_type": "batch",
        "pool_size": 2,
        "repeat": 4
      },
      {
        "name": "HydraTower",
        "channels": 128,
        "d_state": 1,
        "headdim": 64,
        "dropout": 0.1,
        "repeat": 4
      }
    ],
    "head": {
      "name": "Final",
      "in_channels": 64,
      "out_channels": 2,
      "act_func": "silu"
    }
  }
}
```

### Basic Training

Train models across all folds locally using:

```bash
hound_train_folds --backend local -o train_folds params.json data_folds
```

This command:

- Trains one model per fold (6 models for 6 folds)
- Each model uses 4 folds for training, 1 fold for validation, and the remaining fold for testing
- Passes `--fold {fold} --cross {cross}` to `hound_train`, which resolves the
  train/valid/test fold sets directly from the `examples/fold*.zarr` files in
  the data directory. (No per-replicate `data{di}/` directories or symbolic
  links are created — older versions built those; the fold split is now derived
  at load time.)

The split for each replicate is `test = fold`, `valid = (fold + 1 + cross) % num_folds`,
and `train =` everything else (plus `examples/free.zarr` if present). The exact
assignment is logged to stdout and written to `{rep_dir}/train/folds.json` for
each model.

### Training Options

```bash
# Specify number of parallel processes (probably not, if local)
hound_train_folds --backend local -p 4 -o train_folds params.json data_folds

# Train only a subset of folds
hound_train_folds --backend local -f 2 -o train_folds params.json data_folds

# Multiple cross-validation rounds for more robust results
hound_train_folds --backend local -c 2 -o train_folds params.json data_folds

# Setup data directories only (without training)
hound_train_folds --backend local --setup -o train_folds params.json data_folds
```

### Stopping / concluding a run early (GCP)

Many experiments reveal themselves as not worth finishing after a few epochs.
To stop a GCP run and bring the progress home in one step:

1. `Ctrl-C` the orchestrator (the running `hound_train_folds` process). This
   stops the resubmit loop so it won't relaunch anything.
2. Recall that same launch command (up-arrow) and append `--conclude`:

```bash
hound_train_folds --backend gcp --gcp_data_dir gs://<bucket>/<prefix> \
  -o models_basic params_basic.json homo_sapiens mus_musculus --conclude
```

`--conclude`:

- Cancels the run's active Batch jobs **gracefully** (SIGTERM), so each worker's
  exit trap uploads its final partial state and logs to GCS before the VM dies.
- Waits for the jobs to stop, then mirrors the **full** GCS run dir — including
  the `.pth` weights the periodic mirror skips — into the local `-o` dir.
- Prints a per-fold snapshot (last epoch + which weights landed locally).

It's **non-destructive**: the checkpoints stay in GCS, so the run is still
resumable later by re-running the plain launch command (no `--conclude`).

> **Order matters.** Cancel the Batch jobs _without_ first stopping the
> orchestrator and the resubmit loop just relaunches them. Always `Ctrl-C` the
> orchestrator first. `--conclude` warns you if it detects one still running.

The GCS location and project/region are read from the `gcp_run.json` marker the
launch wrote into `-o`. If that marker is missing (e.g. an older run), pass
`--gcp_output_dir gs://...` explicitly (find it with `gcloud batch jobs describe`).

### Run identity and concurrent runs (GCP)

A run is identified by its **content hash**, not by `--name`. The GCS output dir
is `…/train/<run_id>` where `run_id = <params_sha8>-<data_sha8>` (plus the
transfer-weights hash when `--transfer` is used). Two launches with different
params or data get different `run_id`s and are fully independent; re-launching
with the _same_ params/data lands in the same dir and resumes.

Each Batch job carries this identity as labels
(`gcprunner_run`/`gcprunner_kind`/`gcprunner_fold`). The launcher uses those
labels — not the job name — to decide which folds are already running, and
`--conclude` cancels strictly by `gcprunner_run`. So:

- **`--name` is purely cosmetic** — a human-readable prefix for the job names and
  the `gcprunner_name` label. It has no effect on which run is which.
- **Distinct runs coexist** in the same project/region even if they share a
  `--name` (the default is `fold`). A `dp25`-params run and a `b32`-params run
  launched from different directories won't block or cancel each other.
- If a launch reports _"N fold(s) still running from another invocation"_, an
  earlier launch of **this same run** (same params/data → same `run_id`) still
  has active Batch jobs. Confirm with
  `gcloud batch jobs list --project <p> --location <region> --format='table(name,status.state,labels)'`
  and either let them finish or `--conclude` this run.

### Understanding the Output Structure

After training, the output directory contains:

```
train_folds/
├── f0c0/           # Fold 0, Cross 0
│   ├── train/      # Training logs, best model, and folds.json
│   └── params.json
├── f1c0/           # Fold 1, Cross 0
│   ├── train/
│   └── params.json
└── ...
```

Each `f{fold}c{cross}` directory represents:

- `fold`: Which fold is used as the test set
- `cross`: Cross-validation round (for multiple rounds)

## Cross-fold Evaluation

### Basic Evaluation

Evaluate all trained models across all folds:

```bash
hound_eval_folds --backend local -o train_folds params.json data_folds
```

This evaluates each model on all folds, providing comprehensive performance metrics.

### Evaluation Options

```bash
# Enable reverse complement ensembling
hound_eval_folds --backend local --rc -o train_folds params.json data_folds

# Save predictions and targets for detailed analysis
hound_eval_folds --backend local --save -o train_folds params.json data_folds

# Compute rank correlations (slower but more informative)
hound_eval_folds --backend local --rank -o train_folds params.json data_folds

# Evaluate only test sets (faster)
hound_eval_folds --backend local --test -o train_folds params.json data_folds

# Include specificity analysis
hound_eval_folds --backend local --spec -o train_folds params.json data_folds
```

### Understanding Evaluation Output

After evaluation, each fold directory contains:

```
train_folds/f0c0/
├── train/          # Training artifacts
├── eval/           # Evaluation results
│   ├── fold0/      # Model evaluated on fold 0
│   ├── fold1/      # Model evaluated on fold 1
│   ├── ...
│   ├── test -> fold0  # Symlink to test fold
│   └── test.out
└── spec/           # Specificity analysis (if --spec used)
```

Key files:

- `eval/fold{X}/acc.txt`: Accuracy metrics for each target
- `eval/test/acc.txt`: Test set performance (most important)
- `spec/acc.txt`: Specificity analysis results

## Analyzing Results

### Aggregating Cross-fold Results

To get overall performance, you need to aggregate results across folds. The test performance for each fold is in:

```bash
# View test performance for each fold
cat train_folds/f*/eval/test/acc.txt

# Extract key metrics across all folds
for fold in train_folds/f*c0/eval/test/; do
    echo "Fold: $(basename $(dirname $(dirname $fold)))"
    head -2 "$fold/acc.txt"
    echo
done
```

### Performance Metrics

Each `acc.txt` file contains metrics like:

- **PearsonR**: Pearson correlation coefficient
- **R2**: Coefficient of determination

### Cross-fold Statistics

Calculate summary statistics across folds:

```python
import pandas as pd
import glob

# Load results from all folds
results = []
for acc_file in glob.glob('train_folds/f*/eval/test/acc.txt'):
    fold_id = acc_file.split('/')[1]  # e.g., 'f0c0'
    df = pd.read_csv(acc_file, sep='\t')
    df['fold'] = fold_id
    results.append(df)

# Combine and analyze
all_results = pd.concat(results)
summary = all_results.groupby('identifier')[['PearsonR', 'R2']].agg(['mean', 'std'])
print(summary)
```

## Advanced Workflows

### Transfer Learning with Folds

Start from a pre-trained model. The transfer directory is a cross-fold model set
with `f{fold}c{cross}/train/model_best.pth` for each replicate; each new model is
seeded from the matching fold's weights.

```bash
hound_train_folds --backend local --transfer pretrained_model_dir \
  -o train_folds params.json data_folds
```

On GCP this works the same way — the foundation `model_best.pth` weights are staged
to the content cache automatically (content-addressed, so re-runs skip the upload)
and mounted read-only on the workers, replacing the old manual tar +
`gcloud compute scp` procedure:

```bash
hound_train_folds --backend gcp --gcp_data_dir gs://<bucket>/<prefix> \
  --transfer /path/to/pretrained_models \
  -o models_transfer params_transfer.json data/hg38 data/mm10
```

## Computational Considerations

- Cross-fold training requires ~N times more computation (N = number of folds)
- This document uses `--backend local` because it will work for everyone, but jobs can also go to GCP Batch with `--backend gcp` (see [gcprunner](gcprunner.md)), or to Slurm with `--backend slurm`, which requires the non-public `slurmrunner` package and is the default when it is installed.
