# Training and evaluating models

## Introduction

This guide demonstrates how to train a Borzoi-like regulatory sequence machine learning model from scratch using our command-line interface.

## Necessary files

The simplest dataset construction requires a genome FASTA file and some genomic tracks in BigWig (or our bespoke HDF5-based format.) Make a table describing the tracks, which includes the following columns.

```
	identifier	file	clip	clip_soft	scale	sum_stat	strand_pair	description
0	H3K4ME3_S0	src/tests/data/H3K4ME3_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K4me3 S0
1	H3K36ME3_S0	src/tests/data/H3K36ME3_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K36me3 S0
```

See [targets.md](targets.md) for full column documentation.

## Dataset construction

Since we're working with yeast and small genes, smaller sequence lengths are appropriate. We'll stride the sequences by ~1/4 their length plus 15 bp to shift the bins and max pool windows.

```bash
hound_data -l 16384 --local --stride 4109 -t 0.05 -v 0.05 -o data_me src/tests/data/sc3.fa.gz -w 32 src/tests/data/targets_sc3_me.txt
```

## Blacklist/mappability

We've noticed that some tracks have problematic regions where high coverage likely doesn't reflect the biological reality. E.g. see this [UCSC page for hg38](https://genome.ucsc.edu/cgi-bin/hgTrackUi?g=problematicSuper). If you have it, provide to `hound_data` using the `-b` option, and it'll clip the values in overlapping bins to baseline balues.

Similarly, you can provide a BED file of regions you consider to be unmappable due to repetitive sequence with the `-u` option, and control how aggressively to clip values in overlapping bins with the `--umap_clip` options. Finally, `--umap_t` option allows you to remove sequences above the specified proportion of bins overlapping these unmappable regions.

## Model Training

Next, we define a model architecture and training options in JSON format. For the training options, check out `trainer.py` to see how they work. For the different neural network blocks, see `blocks.py` and `layers.py`.

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

Run the following command to train a model.

```bash
hound_train -o train_out params.json data_me
```

Observe progress, including validation metrics in `train_out/log.txt`.

### Per-track loss weights

Every track contributes equally to the loss by default. Two knobs change that: the targets table's `weight` column, which travels with the data (see [targets docs](targets.md)), and `train.loss_weight`, a list of rules that multiply it for one run.

Each rule selects tracks by equality on any targets column and supplies a `weight`. The first matching rule wins, so list specific rules before general ones. A rule naming a column the table lacks is an error, as is a weight that is negative, non-finite, or zeroes out every track.

```json
"loss_weight": [
  { "group": "H3K9me3", "weight": 0.1 },
  { "assay": "chip", "weight": 0.2 }
]
```

H3K9me3 tracks get 0.1 here and every other ChIP track 0.2, each multiplying whatever the `weight` column already says.

`loss_weight_gene` does the same for the gene head against `targets_gene.txt`, which has its own columns. The two lists are independent, so a coverage rule can never reach the gene head by accident.

The loss is a weighted mean over tracks, so only the ratios matter: scaling every weight by a constant leaves training unchanged, and `global_clipnorm` keeps its meaning across weightings. The resolved weights are printed at startup, with a warning for any rule whose selector matched nothing. The `mlm` loss scores nucleotides rather than tracks and rejects both keys.

### Validation metrics and early stopping

Each epoch, `log.txt` gets a summary line per dataset (`loss`, `r`, `r2`, `spec`), and `metrics.tsv` gets every metric as a long table with columns `epoch, split, dataset, group, metric, value`. Dataset-level rows have `group` = `all`.

Tracks are grouped by the targets `group` column (else the `description` prefix). Each group with at least `train.spec_group_min` tracks (default 20, after merging strand pairs) gets:

- `r/<group>`, `r2/<group>`: mean per-track Pearson r and R² over the group's tracks.
- `spec/<group>`: specificity, the mean per-track Pearson r of what remains after removing the group's shared signal. Each track's values (strand pairs summed) are quantile-normalized to the group with a fixed per-track lookup table; predictions use the same table as their targets. The group mean at each position is then regressed out of each track, with the track's own slope and intercept. It rewards predicting how a track differs from its group, not the shared signal or the track's depth and signal-to-noise.

`spec` is the unweighted mean of `spec/<group>` over groups. `hound_eval` reports the same metric per track (`spec` column of `acc.txt`), in its single pass. The tables are built from the exact whole-genome histograms of each track's stored fp16 values, which `hound_data` writes to each `examples/*.zarr` as `target_hist`. For an older dataset, add them with `python -c "from baskerville import dataset; dataset.write_target_hist('data_dir')"`.

`train.stop_stat` chooses the statistic that selects `model_best.pth` and drives `patience`. It is a `{metric: weight}` dict over the keys above plus `loss`, `r_gene`, and `r2_gene`, and the weighted sum is maximized. The presets `"weighted_r_r2"` (default, `{"r": 1, "r2": 0.25}`), `"r"`, `"r2"`, and `"loss"` (`{"loss": -1}`) remain available as strings; these presets score gene-only datasets by −loss. A dict applies only its own keys, e.g. `{"r_gene": 1}` or `{"loss": -1}` for gene-only datasets.

```json
"stop_stat": { "spec": 1, "spec/gtex": 0.5, "r": 0.5, "r2": 0.125 }
```

This selects on specificity across all groups, with extra weight on the `gtex` group and half the default accuracy statistic as a tiebreaker against models that stop predicting the shared signal. The sum runs over the first `early_stop_datasets` validation datasets (default all). A key missing from one dataset contributes 0 there, e.g. `spec/gtex` for a mouse dataset without that group, but a key missing from every dataset is an error, raised before training if it involves `spec`. Use a negative weight for `loss`.

## Model Evaluation

Finally, evaluate on the test set using the following command, now turning on reverse complement ensembling.

```bash
hound_eval -o eval_out --rc params.json train_out/model_best.pth data_me
```

The file `eval_out/acc.txt` will be a table of various accuracy metrics computed for eack track.
