# Gradients & Attribution

This page covers two complementary pieces:

1. The low-level `SeqNN.gradients()` API for computing per-nucleotide gradients of model predictions w.r.t. input sequence.
2. High-level command‑line scripts (`hound_grad` and `hound_grad_folds`) that batch this computation over genes (from a GTF) and optionally across model folds / replicates, writing HDF5 and BigWig outputs for downstream visualization.

---

## 1. Python API: `SeqNN.gradients()`

Compute gradients of model predictions with respect to input sequences for sequence interpretation and attribution analysis.

### Basic Usage

```python
import torch, pandas as pd
from baskerville.seqnn import SeqNN

# Load parameters & instantiate (example)
model = SeqNN(params_dict)

# Input one-hot or differentiable representation (channels=4, seq_length)
x = torch.randn(4, model.params['seq_length'], requires_grad=True)

# Compute gradients over all spatial bins & tasks (head 0)
grads = model.gradients(x, hi=0)

# Focus on specific spatial region (bins 100–200)
roi_grads = model.gradients(x, hi=0, spatial_slice=slice(100, 200))

# Focus on first 3 tasks
task_grads = model.gradients(x, hi=0, task_slice=[0, 1, 2])

# Apply inverse (un‑transform) target scaling before gradient computation
targets_df = pd.read_csv('targets.txt', sep='\t', index_col=0)
untrans_grads = model.gradients(x, hi=0, untransform_targets_df=targets_df)
```

### Custom Aggregation Functions

For more complex aggregation patterns, you can provide a custom aggregation function via the `agg_fn` parameter. This is useful for specialized analyses like circadian biology, differential analysis, or custom weighted aggregations.

```python
# Example 1: Circadian mesor analysis
def circadian_agg_fn(yh, gene_spatial_slice, circadian_targets):
    """
    Compute gradient w.r.t. deviation from circadian mesor for first timepoint.

    Args:
        yh: [num_targets, seq_len_bins] predictions (already untransformed)
        gene_spatial_slice: spatial indices for gene CDS
        circadian_targets: list of 8 target indices for circadian cycle
    """
    # Get gene predictions for 8 circadian targets
    gene_preds = yh[circadian_targets, :][:, gene_spatial_slice]  # [8, gene_bins]
    gene_values = gene_preds.sum(dim=1)  # [8] - sum over gene spatial bins

    # Compute mesor (average over 8 time points)
    mesor = gene_values.mean()

    # For first time point, subtract mesor (deviation from baseline)
    result = gene_values[0] - mesor

    return result

# Usage for circadian analysis
circadian_grads = model.gradients(
    x,
    hi=0,
    untransform_targets_df=targets_df,
    agg_fn=circadian_agg_fn,
    agg_fn_kwargs={
        'gene_spatial_slice': gene_cds_slice,
        'circadian_targets': [0, 1, 2, 3, 4, 5, 6, 7]
    }
)

# Example 2: Differential expression analysis
def differential_agg_fn(yh, gene_slice, treatment_targets, control_targets):
    """Compute gradient w.r.t. log fold change between conditions."""
    # Get gene expression for treatment vs control
    treatment_expr = yh[treatment_targets, :][:, gene_slice].sum()
    control_expr = yh[control_targets, :][:, gene_slice].sum()

    # Log fold change (with pseudocount)
    log_fc = torch.log((treatment_expr + 1e-6) / (control_expr + 1e-6))

    return log_fc

# Usage for differential analysis
diff_grads = model.gradients(
    x,
    agg_fn=differential_agg_fn,
    agg_fn_kwargs={
        'gene_slice': gene_spatial_slice,
        'treatment_targets': [0, 2, 4],
        'control_targets': [1, 3, 5]
    }
)

# Example 3: Weighted tissue-specific analysis
def tissue_weighted_agg_fn(yh, gene_slice, tissue_targets, tissue_weights):
    """Compute gradient w.r.t. weighted tissue expression."""
    gene_expr = yh[tissue_targets, :][:, gene_slice].sum(dim=1)  # [num_tissues]
    weighted_expr = (gene_expr * tissue_weights).sum()
    return weighted_expr

# Usage for tissue-weighted analysis
tissue_grads = model.gradients(
    x,
    agg_fn=tissue_weighted_agg_fn,
    agg_fn_kwargs={
        'gene_slice': gene_spatial_slice,
        'tissue_targets': [0, 1, 2, 3],
        'tissue_weights': torch.tensor([0.4, 0.3, 0.2, 0.1])  # Brain, liver, heart, muscle
    }
)
```

### Arguments

| Argument                 | Description                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x`                      | Input sequence tensor shaped `(4, seq_length)`. (A batch dimension is optional; tests exercise the no-batch case.)                                                        |
| `hi`                     | Model head index (multi‑head models). Default `0`.                                                                                                                        |
| `spatial_slice`          | Restrict gradient attribution to selected output bins. Accepts `slice`, list/array of indices, or boolean mask (length = output_length). Ignored if `agg_fn` is provided. |
| `task_slice`             | Restrict to a subset of tasks (indices or boolean mask). Ignored if `agg_fn` is provided.                                                                                 |
| `untransform_targets_df` | DataFrame used to inverse-transform targets prior to backprop (columns like `scale`, `sum_stat`, `clip_soft`). Must align with (possibly sliced) tasks.                   |
| `log_transform`          | Apply optional log transform to summed coverage before attribution. Ignored if `agg_fn` is provided.                                                                      |
| `agg_fn`                 | Custom aggregation function that takes `yh` tensor (after untransform) and returns a scalar. If provided, `spatial_slice`, `task_slice`, and `log_transform` are ignored. |
| `agg_fn_kwargs`          | Dictionary of keyword arguments to pass to `agg_fn`. Default `None`.                                                                                                      |

Returns a tensor the same shape as the input `x`.

**Note**: When `agg_fn` is provided, the default slice-based aggregation (`spatial_slice`, `task_slice`, `log_transform`) is bypassed in favor of the custom function. This allows for arbitrary aggregation patterns while maintaining backward compatibility.

### Common Attribution Patterns

Reference nucleotide importance:

```python
ref_attr = (grads * x).sum(dim=0)  # Sum over nucleotide channels
```

Unsigned (variance) importance:

```python
unsigned = grads.var(dim=0)
```

Regional smoothing:

```python
from scipy.ndimage import gaussian_filter1d
smooth = gaussian_filter1d(unsigned.detach().cpu().numpy(), sigma=3.0)
```

Variant effect (simple ref/alt difference):

```python
ref_grads = model.gradients(ref_x, hi=0)
alt_grads = model.gradients(alt_x, hi=0)
delta = alt_grads - ref_grads
```

Ensembling (reverse complement / shifts) is controlled by model attributes `ensemble_rc` and `ensemble_shifts` (see tests for examples) and automatically averaged inside `gradients()`.

---

## 2. CLI Scripts

### `hound_grad`

Computes per‑nucleotide gradients for each gene defined in a GTF. For every gene, a centered sequence window of length `seq_length` (from the model params) is extracted, gradients are computed with respect to strand‑appropriate tasks, and results are aggregated. Optionally reverse complement augmentation (`--rc`) averages forward & RC orientations.

**Note**: This script currently uses the default slice-based aggregation approach (`spatial_slice`, `task_slice`, `log_transform`). Custom aggregation functions are available through the Python API but not yet supported in the command-line interface.

Outputs:

- `scores.h5` containing:
  - `seqs`: `(num_genes, L, 4)` one‑hot sequences
  - `grads`: `(num_genes, L, 4)` mean-centered gradients (after augmentation & normalization)
  - `gene`, `chr`, `start`, `end`, `strand`: gene metadata
- Optional BigWigs if `--bigwig`:
  - `<gene>_ref.bw`: reference nucleotide gradient score per base ( (seq \* grad).sum(axis=1) )
  - `<gene>_var.bw`: variance of gradients across nucleotides per base

#### Command

```
hound_grad \
	-f GENOME.fa \
	-t targets.txt \
	[--bigwig] [--log] [--rc] [--head 0] \
	-o grad_out \
	params.json model_best.pth genes.gtf
```

#### Key Options

| Flag                  | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| `-f / --genome_fasta` | Genome FASTA (indexed by `samtools faidx`).                                    |
| `-t / --targets_file` | Targets table (tab‑separated) used to set output slice & untransform metadata. |
| `--head`              | Model head index (default 0).                                                  |
| `--log`               | Apply log transform to summed coverage (passed to gradients).                  |
| `--rc`                | Add reverse complement augmentation (averages FWD + RC).                       |
| `--bigwig`            | Emit per‑gene BigWig files for quick visualization.                            |
| `-o / --out_dir`      | Output directory (created if absent).                                          |

#### Interpreting BigWigs

`*_ref.bw` approximates signed importance of the reference base; `*_var.bw` highlights uncertainty or multi‑base sensitivity (useful for motif discovery). Both can be loaded into IGV, UCSC, or pyBigWig-based tooling.

---

### `hound_grad_folds`

Automates running `hound_grad` over multiple cross‑validation folds (and optional replicate crosses) discovered in a `models_dir`. It generates per‑fold subdirectories `grad_out/f{fold}c{cross}` each with its own `scores.h5` (and BigWigs if requested).

#### Command

```
hound_grad_folds \
	-f GENOME.fa \
	-t targets.txt \
	--folds N  --crosses C \
	[--bigwig] [--log] [--rc] [--head 0] \
	-o grad_out \
	[--backend local] [--queue gpuq] [--conda_env myenv] [--parallel_jobs K] \
	params.json models_dir genes.gtf
```

#### Additional Options

| Flag                   | Description                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `--crosses`            | Number of replicate crosses (default 1).                                                          |
| `--folds`              | Explicit number of folds; if absent auto-detect.                                                  |
| `--name`               | SLURM job name prefix (default `grad`).                                                           |
| `--queue`              | SLURM queue (influences CPU/GPU/time defaults).                                                   |
| `--conda_env`          | Conda environment to activate inside jobs.                                                        |
| `--backend`            | `local` (in-process) or `slurm`; defaults to `slurm` if `slurmrunner` is installed, else `local`. |
| `-p / --parallel_jobs` | Max parallel local jobs or SLURM submissions in flight.                                           |

---

## 3. Typical Workflow

1. Train models / folds (see `docs/train.md` or `hound_train_folds`).
2. Ensure you have `targets.txt`, `params.json`, `genes.gtf`, and `GENOME.fa` (indexed) available.
3. Run single model gradients:
   ```
   hound_grad -f GENOME.fa -t targets.txt params.json model_best.pth genes.gtf -o grad_out --bigwig
   ```
4. (Optional) Run across folds:
   ```
   hound_grad_folds -f GENOME.fa -t targets.txt -o grad_out params.json models_dir genes.gtf --bigwig --rc
   ```
5. Inspect `grad_out/f0c0/*.bw` in a genome browser; compute motif enrichments using the variance track, etc.
6. Downstream analysis: combine fold HDF5s, aggregate per‑gene statistics, integrate with SNP or ISM analyses.

---

## 4. Programmatic Access to HDF5

```python
import h5py
with h5py.File('grad_out/scores.h5') as h5:
		seqs = h5['seqs'][:]      # (num_genes, L, 4)
		grads = h5['grads'][:]    # (num_genes, L, 4)
		genes = [g.decode() for g in h5['gene']]
		# Reference attribution
		ref_attr = (seqs * grads).sum(axis=2)
```

Mean-centering performed by the script implies `grads[gi, pos].mean(axis=-1)=0`; signed importance for the reference base therefore stands out relative to other nucleotides.
