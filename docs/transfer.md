# Training and evaluating models

## Introduction

This guide demonstrates how to train a Borzoi-like regulatory sequence machine learning model, transferred from a pre-trained model using our command-line interface.

## Necessary files

The simplest dataset construction requires a genome FASTA file and some genomic tracks in BigWig (or our bespoke HDF5-based format.) Make a table describing the tracks, which includes the following columns.

```
	identifier	file	clip	clip_soft	scale	sum_stat	strand_pair	description
0	H3K9AC_S0	src/tests/data/H3K9AC_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K9ac S0
1	H3K27AC_S0	src/tests/data/H3K27AC_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K27ac S0

```

## Dataset construction

To ensure consistency with the original pre-training dataset, you'll need to use the same parameters that were used during its construction. Since the dataset creation script is deterministic, running the same command with your new data will generate a split that matches the original logic. For reference, you can find the exact parameters and command used in [`train.md`](train.md).

```bash
hound_data -l 16384 --local --stride 4109 -t 0.05 -v 0.05 -o data_ac src/tests/data/sc3.fa.gz -w 32 src/tests/data/targets_sc3_ac.txt
```

## Model Training

Next, we define a model architecture, for which the trunk matches the pretrained model, but the head differs. To specify that we want to load the pretrained model, specify its path in the `train` section of the params JSON.

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
    "num_workers": 4,
    "pretrained_model": "train_out/model_best.pth"
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

Then run the following command to train a model.

```bash
hound_train -o train_out params.json data_ac
```

Observe progress, including validation metrics in `train_out/log.txt`.
