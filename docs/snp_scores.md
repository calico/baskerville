# SNP Score Naming System

SNP scores quantify the predicted effect of a variant on regulatory activity. Scores are specified via `--stats` as a comma-separated list, e.g. `--stats logFC,covgene/logD2`.

## Prefixes

Prefixes control **what** is scored and **how** results are indexed in the output HDF5.

| Prefix           | Indexed by    | Data source                        | Example         |
| ---------------- | ------------- | ---------------------------------- | --------------- |
| `cov/` (or none) | SNP           | Full-sequence coverage predictions | `cov/logFC`     |
| `covgene/`       | SNP-gene pair | Gene-sliced coverage predictions   | `covgene/logD2` |
| `gene/`          | SNP-gene pair | Gene head aggregated predictions   | `gene/logFC`    |

- **`cov/`** stats produce one score per SNP per coverage target, comparing full-sequence ref vs alt predictions. Unprefixed stats (e.g. `logFC`) are treated as `cov/`. Strand-paired targets are combined via a strand transform matrix.
- **`covgene/`** stats produce one score per SNP-gene pair per coverage target. Predictions are sliced to the gene's exon bins (respecting strand) and to gene tracks (targets `gene` column, or `assay` in RNA/RNA3/CAGE) before computing the score. Requires `-g/--genes_gtf`. When no `cov/` stats are requested, only gene tracks are predicted and `targets_cov.txt` lists just those.
- **`gene/`** stats produce one score per SNP-gene pair per gene head target. Requires a model with a gene head. Currently only `gene/logFC` is supported (log fold change of the gene head's aggregated predictions).

## Score functions

All scores are averaged over shifts and stored as float16. Predictions `R` (ref) and `A` (alt) have shape `(shifts, targets, length)`.

| Stat     | Formula                                               | Notes                                                                         |
| -------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| `logFC`  | `log2(mean_s[sum_l(A)]+1) - log2(mean_s[sum_l(R)]+1)` | Log fold change of summed predictions. Signed; most common score.             |
| `logSUM` | `mean_s[ sum_l(log2(A+1)) - sum_l(log2(R+1)) ]`       | Sum of per-bin log differences. Similar to logFC but sums in log space first. |
| `logSED` | Synonym for `logFC`                                   | Kept for backward compatibility.                                              |
| `SUM`    | `mean_s[ sum_l(A) - sum_l(R) ]`                       | Linear sum difference.                                                        |
| `logD2`  | `mean_s[ \|\| log2(A+1) - log2(R+1) \|\|_2 ]`         | L2 norm of per-bin log differences. Unsigned; measures magnitude of change.   |
| `logD1`  | `mean_s[ \|\| log2(A+1) - log2(R+1) \|\|_1 ]`         | L1 norm of per-bin log differences.                                           |
| `D2`     | `mean_s[ \|\| A - R \|\|_2 ]`                         | L2 norm in linear space.                                                      |
| `D1`     | `mean_s[ \|\| A - R \|\|_1 ]`                         | L1 norm in linear space.                                                      |
| `REF`    | Raw ref predictions                                   | Full prediction tensor (shifts × length × targets).                           |
| `ALT`    | Raw alt predictions                                   | Full prediction tensor.                                                       |

## HDF5 output structure

```
scores.h5
├── snp              # SNP rsids (S,)
├── chr              # chromosomes (S,)
├── pos              # positions (S,)
├── ref_allele       # reference alleles (S,)
├── alt_allele       # alternate alleles (S,)
├── quantiles        # quantile grid breakpoints (Q,)
│
├── cov/
│   ├── logFC            # SNP-indexed score (S, T_cov)
│   └── logFC_quantiles  # per-target thresholds (T_cov, Q)
│
├── gene_ids         # unique gene IDs (G,)
├── snp_idx          # SNP index per pair (P,)
├── gene_idx         # gene index per pair (P,)
│
├── covgene/
│   ├── logD2            # pair-indexed score (P, T_cov)
│   └── logD2_quantiles  # per-target thresholds (T_cov, Q)
│
└── gene/
    ├── logFC            # pair-indexed gene head logFC (P, T_gene)
    └── logFC_quantiles  # per-target thresholds (T_gene, Q)
```

Where `S` = number of SNPs, `P` = number of SNP-gene pairs, `T_cov` = coverage targets, `T_gene` = gene expression targets, `G` = unique genes.

## Examples

```bash
# Coverage scores only (no GTF)
hound_snp -f genome.fa --stats logFC,logD2 params.json model.pth snps.vcf

# Coverage + gene-sliced coverage + gene head
hound_snp -f genome.fa --stats logFC,covgene/logFC,covgene/logD2,gene/logFC -g genes.gtf params.json model.pth snps.vcf

# Gene-centered mode (only covgene/ and gene/ stats computed)
hound_snp -f genome.fa --stats covgene/logFC,gene/logFC -g genes.gtf --center_gene params.json model.pth snps.vcf
```

When `--center_gene` (or `--pregrouped_seqs`) is used, sequences are centered on genes rather than variants. Only `covgene/` and `gene/` prefixed stats are computed; `cov/` stats are ignored.
