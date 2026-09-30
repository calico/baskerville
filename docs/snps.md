# SNP Scoring

This guide covers how to score Single Nucleotide Polymorphisms (SNPs) and other variants using sequence models to predict their functional effects on regulatory activity.

The scripts described deliver gene-agnostic scores, where all output bins are considered equally.

## Core Scripts

### hound_snp

The primary script for scoring SNPs using a single trained model.

#### Basic Usage

```bash
python -m baskerville.scripts.hound_snp \
    params.json \
    model.pth \
    variants.vcf \
    -t targets.txt \
    -o snp_output \
    -f genome.fa
```

#### Required Arguments

- `params_file`: JSON file containing model parameters
- `model_file`: Trained model file (.pth format)
- `vcf_file`: Input VCF file containing variants to score
- `-t, --targets_file`: File specifying target indexes and labels in table format
- `-f, --genome_fasta`: Reference genome FASTA file

#### Key Options

**Output and Performance**

- `-o, --out_dir`: Output directory for results (default: `snp_out`)
- `-m, --mix_dtype`: Mixed precision dtype (`float16`, `bfloat16`, `float32`)
- `--head`: Model head to use for predictions (default: 0)

**Variant Processing**

- `-c, --cluster_pct`: Cluster SNPs within a percentage of sequence length for computational efficiency (default: 0). See below for more detail.
- `-g, --genes_gtf`: Enable gene-specific scoring mode (provide GTF). See Gene-Specific Scoring.
- `-n, --norm`: HDF5 file to use for normalization quantiles instead of current variant set
- `--index_start/--index_end`: Process only a subset of variants (0-based indexing); intended to enable parallel processing.

**Prediction Options**

- `--rc`: Average forward and reverse complement predictions
- `--shifts`: Ensemble prediction shifts (comma-separated, default: "0")
- `--indel_stitch`: Apply shift augmentation stiching to improve indel scoring. Works best when variants are centered, i.e. -c 0.
- `--span`: (Gene mode only) Aggregate across the entire gene span (min exon start to max exon end) instead of exon-only bins.

**Windowed Scoring**

- `--local_window`: Local window size in bp (default: 2048). Applies to targets whose `window` column is set to `local` in the [targets table](targets.md).

When a target has `window=local`, its score is computed over a window of `--local_window` bp centered on the variant position, rather than the full output sequence. This improves variant effect scores for epigenomic tracks like ATAC-seq and DNase-seq, where the relevant signal is concentrated near the variant. Stats that benefit most: `SUM`, `D2`, `logSUM`, `logD2`.

The window is converted to output bins and centered on the variant's bin position. Near sequence boundaries, the window truncates asymmetrically. Targets without `window=local` are scored over the full sequence as usual.

Gene-specific scoring (`covgene/`, `gene/`) is unaffected — gene scores always use gene-specific exon bins regardless of the `window` column.

**Statistics**

- `--stats`: Comma-separated list of statistics to compute (default: "logSUM")
  - `SUM`: Sum difference between alt and ref predictions
  - `logSUM`: Log-transformed sum difference
  - `D1`: L1 norm of difference vector
  - `D2`: L2 norm of difference vector
  - `logD1/logD2`: Log-transformed norms
  - `REF/ALT`: Raw reference and alternative predictions. (Unreasonably large for any more than a few variants!)

### Score Normalization

The normalization feature allows for better calibration of variant scores, especially useful for small VCF files where the variant set itself provides insufficient context for statistical comparison.

#### Using Normalization

**For hound_snp:**

```bash
python -m baskerville.scripts.hound_snp \
    params.json model.pth variants.vcf \
    -t targets.txt -f genome.fa \
    -n large_population_scores.h5 \
    -o snp_output
```

**For hound_snp_folds:**

```bash
python -m baskerville.scripts.hound_snp_folds \
    params_file model_dir vcf_file \
    -f genome.fa \
    -t targets.txt \
    -n normalization_subdir \
    --crosses 4 --folds 4
```

#### How Normalization Works

- **Without normalization**: Quantiles are computed from the current variant set being scored
- **With normalization**: Quantiles are computed from a larger, previously scored dataset
- **Fold-specific normalization**: In cross-fold ensembles, each fold uses its corresponding normalization file

#### Requirements

- Normalization files must contain the same statistics being computed (e.g., `logSUM`, `D2`)
- Files must be in HDF5 format with the same structure as scoring output

### SNP Clustering

The `-c, --cluster_pct` parameter enables computational optimization by grouping nearby variants to share reference sequence predictions.

#### How Clustering Works

When `-c, --cluster_pct > 0`:

1. **Distance Calculation**: The valid clustering distance is computed as:
   ```
   valid_distance = sequence_length × cluster_pct
   ```
2. **Grouping Algorithm**:
   - SNPs are processed sequentially (requires sorted VCF)
   - A new cluster starts when either:
     - The chromosome changes
     - The distance from the current cluster's first SNP exceeds `valid_distance`
   - All SNPs within the valid distance are grouped into the same cluster

3. **Shared Reference**: All SNPs in a cluster use the same reference sequence prediction, centered on the cluster's midpoint

#### Benefits and Trade-offs

**Computational Benefits:**

- Significantly reduces model inference calls for dense variant regions

**Accuracy Considerations:**

- Reference predictions are shared within clusters
- Alternative predictions remain specific to each variant, but the variant will not be centered.

#### Recommended Values

- **Conservative**: `0.01-0.05` (1-5%) - Minimal clustering, maximum accuracy
- **Aggressive**: `0.05-0.25` (5-25%) - Maximum efficiency for very dense regions

Beyond 25%, you may begin to lose accuracy due to lost context on the sequence flanks.

**Note**: Clustering requires position-sorted VCF files and disables reference allele flipping (`flip_ref=False`), so the first allele in the VCF must match the reference.

### Gene-Specific Scoring (-g / --genes_gtf)

Passing `-g genes.gtf` to `hound_snp` switches from SNP (variant-centric) mode to SNP-by-gene mode. Each variant is scored separately for every overlapping gene, restricting predictions to bins that fall within the gene's exonic span (subject to model output cropping) and to targets on the gene's strand.

#### Basic Usage

```bash
python -m baskerville.scripts.hound_snp \
    params.json model.pth variants.vcf \
    -t targets.txt -f genome.fa \
    -g genes.gtf \
    -o snp_gene_out
```

#### What Happens Internally

1. Genes are read from the GTF (per-gene aggregation; transcripts are not distinguished once exons are merged per gene object in the loader).
2. Genes are optionally clustered with `-c/--cluster_pct` exactly like SNP clustering, but using gene midpoints. A single reference forward pass is reused for all genes and SNPs in the cluster to reduce compute.
3. Each cluster defines a genomic window sized to the model `seq_length`; SNPs inside are one-hot encoded for reference and alternate alleles. (Indels still honored; `--indel_stitch` works the same.)
4. For every SNP and every gene in the cluster that contains that SNP, the model predictions are sliced to bins overlapping the gene's exonic output coordinates. Bins off the edges after output cropping are ignored (a warning prints if everything is outside).
5. Target tensor is strand-filtered: if gene is `+`, only targets whose `strand != '-'` are retained; if gene is `-`, only targets whose `strand != '+'` are retained (mirrors plus/minus masks in code).
6. Requested statistics (`--stats`) are computed on the sliced reference vs alternate predictions.

#### Output Differences

In gene mode `scores.h5` contains additional datasets:

- `gene_ids`: All gene IDs (sorted, unique) that overlapped at least one scored SNP.
- `snp_idx`: For each SNP–gene pair row, the index into the global `snp` list.
- `gene_idx`: Matching index into `gene_ids`.

Each SNP statistic dataset has shape `(num_snp_gene_pairs, num_targets_retained)`; each row corresponds to one SNP–gene pair. Unlike SNP mode, raw prediction tensors (`REF`, `ALT`) are NOT supported in gene mode (the current implementation would cause a shape mismatch). Avoid specifying `REF` or `ALT` in `--stats` when using `-g`.

Normalization (if `-n` provided) is performed over SNP–gene pair score values instead of per-SNP values.

#### Gene Span vs Exons

By default only exon-overlapping bins are used (union of all exons). Supplying `--span` switches to aggregating bins covering the entire gene span (from the earliest exon start to the latest exon end), including intronic regions within that span. This can smooth sparse signals but may dilute exon-specific effects for very long introns.

#### Clustering Genes (`-c` with `-g`)

`-c` in gene mode clusters by gene midpoint (not SNP position). A larger value increases reference prediction reuse but may move some SNPs far from center, modestly impacting accuracy for long genes or large clusters.

#### When to Use Gene Mode

Use `-g` when you need:

- Gene-centric association (multiple SNPs in one gene collapsed later by external routines).
- Strand-specific evaluation restricted to gene orientation.
- Reduction of noise from flanking non-genic bins.

If you only need variant-level statistics irrespective of gene context (e.g., genome-wide prioritization), omit `-g`.

#### Downstream Aggregation

Because output is per SNP–gene pair, typical downstream steps involve aggregating all rows with the same `gene_idx` (e.g., max |logSUM|, sum of positive effects, etc.). This is intentionally left flexible.

#### Limitations / Notes

- `REF` / `ALT` raw outputs unsupported (see above).
- All overlapping genes are considered; no distance-based pruning beyond cluster window.
- Multi-allelic variants should be split before running; only first ALT allele per line is used by upstream VCF parsing logic.

Example to list top absolute logSUM per gene (pseudocode):

```python
import h5py, numpy as np
with h5py.File('snp_gene_out/scores.h5') as h:
    genes = [g.decode() for g in h['gene_ids'][...]]
    snp_idx = h['snp_idx'][...]
    gene_idx = h['gene_idx'][...]
    logsum = h['logSUM'][...]  # shape: pairs x targets
    per_gene = {}
    for row,(gi) in enumerate(gene_idx):
        val = np.max(np.abs(logsum[row]))
        per_gene.setdefault(gi, []).append(val)
    top = sorted(((genes[gi], max(v)) for gi,v in per_gene.items()), key=lambda x: -x[1])[:20]
    for g,v in top: print(g, v)
```

### Output Format

Results are saved as HDF5 files in the output directory:

**scores.h5** contains:

- `snp`: SNP identifiers
- `chr`: Chromosome names
- `pos`: Genomic positions
- `ref_allele/alt_allele`: Reference and alternative alleles
- `targets`: Group containing complete target information (target_ids, target_labels, and all other target metadata)
- Statistics datasets (e.g., `logSUM`, `D2`) with shape `(num_snps, num_targets)`
- `quantiles`: Array of quantile values used for normalization
- `*_quantiles`: Quantile distributions for each statistic, computed from either the current variant set or a specified normalization file

## Cross-Fold Ensemble Scoring

### hound_snp_folds

For robust predictions using ensemble models from cross-validation folds.

#### Basic Usage

```bash
python -m baskerville.scripts.hound_snp_folds \
    params_file model_dir vcf_file \
    -f genome.fa \
    -t targets.txt \
    --crosses 4 \
    --folds 4 \
    --backend local \
    -o snps_out
```

#### Key Features

**Ensemble Options**

- `--crosses`: Number of cross-validation rounds
- `--folds`: Number of folds per cross
- `--f_list`: Subset of folds to use (comma-separated)
- `-n, --norm`: Model directory subdirectory containing normalization HDF5 files for each fold

**Execution Control**

- `--backend`: `local` (run in-process), `slurm` (submit SLURM jobs; the default when `slurmrunner` is installed, else `local`), or `gcp` (Batch)
- `-p, --parallel_jobs`: Number of parallel jobs
- `--embed`: Embed output in the models directory, instead of a separate output directory; typically used for benchmarking.

**SLURM Integration**

- Automatically submits jobs to SLURM clusters
- `--name`: Job name prefix
- `-e, --conda_env`: Conda environment to activate in each job (default: the current environment)

The script automatically:

1. Submits individual SNP scoring jobs for each fold
2. Uses fold-specific normalization files when `--norm` is specified
3. Collects and averages results across folds
4. Produces final ensemble predictions

#### Fold-Specific Normalization

When using the `--norm` option with `hound_snp_folds`, the script expects a subdirectory structure:

```
models_dir/
├── f0c0/
│   ├── train/model_best.pth
│   └── normalization_subdir/scores.h5  # Normalization for fold 0, cross 0
├── f1c0/
│   ├── train/model_best.pth
│   └── normalization_subdir/scores.h5  # Normalization for fold 1, cross 0
└── ...
```

Each fold automatically uses its corresponding normalization file, ensuring proper cross-validation discipline and preventing data leakage.

#### Output Structure

```
snps_out/
├── f0c0/
    └── scores.h5  # replicate results
├── f1c0/
    └── scores.h5  # replicate results
├── ...
└── ensemble/
    └── scores.h5  # Final averaged results
```
