# Targets Table

The targets table is a tab-separated file describing the genomic tracks that the model predicts. It is used throughout training, evaluation, and variant scoring.

## Format

Tab-separated with a 0-based integer index as the first (unnamed) column. Loaded via `pd.read_csv(path, sep="\t", index_col=0)`.

## Columns

### Required

| Column        | Type   | Description                                          |
| ------------- | ------ | ---------------------------------------------------- |
| `identifier`  | string | Unique track name (e.g. `H3K4ME3_S0`, `DNASE:K562+`) |
| `file`        | string | Coverage track: BigWig, .w5 HDF5, or .hw (hdwig)     |
| `description` | string | Human-readable label (e.g. `CHIP:H3K4me3 S0`)        |

### Data transformation

These columns control how raw coverage values are transformed during dataset construction and how predictions are untransformed back to original scale.

| Column      | Type   | Description                                                                                                        |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| `clip`      | float  | Hard clipping threshold; values are clipped to [-clip, clip]                                                       |
| `clip_soft` | float  | Soft clipping threshold; values above this are smoothly compressed via `clip_soft - 1 + sqrt(val - clip_soft + 1)` |
| `scale`     | float  | Scaling factor applied during data generation; predictions are divided by this during untransform                  |
| `sum_stat`  | string | Summary statistic transform applied per bin: `sum`, `sum_sqrt`, or `sum_exp75`                                     |

### Loss weighting

| Column   | Type  | Description                                                                             |
| -------- | ----- | --------------------------------------------------------------------------------------- |
| `weight` | float | Per-track multiplier on the training loss. Default 1. Never applied to the data itself. |

Both `targets.txt` and `targets_gene.txt` honor it, each weighting its own head's loss. Weights must be finite and nonnegative, and at least one track must be positive.

`scale` and `weight` are easy to confuse:

|                                     | `scale`                                  | `weight`                 |
| ----------------------------------- | ---------------------------------------- | ------------------------ |
| applied                             | at dataset construction, before clipping | at training, to the loss |
| stored in the dataset               | yes                                      | never                    |
| moves the `clip` / `clip_soft` knee | yes                                      | no                       |
| changing it                         | requires regenerating the dataset        | free                     |

Because `poisson_mn` is linear in the targets, a track's pull on the trunk is proportional to its stored values — so `scale` doubles as an implicit loss weight. `weight` separates that role out, leaving `scale` to set where the squash and clip thresholds land. A run can override `weight` for selected tracks without touching the data; see [training docs](train.md).

### Model untransform mode

Model params can choose which inverse transform logic to use at prediction time.

Set `params["model"]["untransform"]` to one of:

- `"borzoi"`: Use Borzoi inverse order, matching Borzoi-style preprocessing.
- Missing or any other value: Use the default inverse logic.

For `"borzoi"`, inverse operations are applied in this order:

1. Divide by `scale`.
2. Undo `clip_soft` with `clip_soft + (x - clip_soft)^2` for values above `clip_soft`.
3. Undo `sum_stat` power for `_sqrt` tracks with `x^(4/3)`.

### Track relationships

| Column        | Type   | Description                                                                                                                                                                                       |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `strand_pair` | int    | Index of the paired strand target. When `strand_pair == own index`, the target is strandless. Otherwise, the two targets form a pair that must be swapped during reverse complement augmentation. |
| `group`       | string | User-defined grouping label for related tracks (used in evaluation dashboards). If missing, it is inferred on the fly (typically from the identifier prefix).                                     |

At runtime, a `strand` column (`+`, `-`, or `.`) is derived from `strand_pair` and the identifier suffix — it is not stored in the file.

### SNP scoring

| Column   | Type   | Description                                                                                                                                                                   |
| -------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `window` | string | `local` = compute variant scores over a local window centered on the variant (size set by `--local_window`, default 2048 bp). Empty or absent = score over the full sequence. |

See [SNP scoring docs](snps.md) for details on windowed scoring.

### Derived columns

This column is created at runtime and does not appear in the file:

- **`strand`**: Derived from `strand_pair` and identifier suffix (`+`, `-`, or `.` for strandless).

## Examples

### Coverage targets

```
	identifier	file	clip	clip_soft	scale	sum_stat	strand_pair	description
0	H3K4ME3_S0	data/H3K4ME3_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K4me3 S0
1	H3K36ME3_S0	data/H3K36ME3_S0_coverage.w5	512	512	1.0	sum_sqrt	0	CHIP:H3K36me3 S0
```

### Gene expression targets

Gene targets are simpler — typically just `identifier`, `file`, `scale`, and `description`:

```
	identifier	file	scale	description
0	RNA_S0	data/gene_expr_S0.tsv	1.0	RNA-seq sample 0
1	RNA_S1	data/gene_expr_S1.tsv	1.0	RNA-seq sample 1
```
