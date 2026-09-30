# In Silico Saturated Mutagenesis (ISM)

This guide covers how to perform in silico saturated mutagenesis (ISM) to assess the functional impact of all possible nucleotide changes in defined genomic regions.

## Overview

ISM systematically mutates every position in a genomic region to understand how sequence changes affect predicted regulatory activity. The process involves:

1. **Loading genomic regions** from BED files or variants from VCF files
2. **Extracting sequences** for each region of interest
3. **Systematically mutating** every position in the defined region
4. **Computing scores** for each possible nucleotide change using sequence-based models
5. **Outputting results** in HDF5 format for downstream analysis

## ISM for BED Regions

### hound_ism_bed

The primary script for performing ISM on genomic regions defined in a BED file.

#### Basic Usage

```bash
python -m baskerville.scripts.hound_ism_bed \
    params.json \
    model.pth \
    regions.bed \
    -t targets.txt \
    -o ism_output \
    -f genome.fa \
    -l 200
```

#### Required Arguments

- `params_file`: JSON file containing model parameters
- `model_file`: Trained model file (.pth format)
- `bed_file`: BED file containing genomic regions to analyze
- `-t, --targets_file`: File specifying target indexes and labels in table format
- `-f, --genome_fasta`: Reference genome FASTA file

#### Key Options

**Output and Performance**

- `-o, --out_dir`: Output directory for results (default: `ism_bed_out`)
- `-m, --mix_dtype`: Mixed precision dtype (`float16`, `bfloat16`, `float32`)
- `--head`: Model head to use for predictions (default: 0)

**Mutation Region Definition**

- `-l, --mut_len`: Length of center sequence to mutate (default: 0)
- `-u, --mut_up`: Nucleotides upstream of center to mutate (default: 0)
- `-d, --mut_down`: Nucleotides downstream of center to mutate (default: 0)

When using `-u` and `-d`, the total mutation length becomes `mut_up + mut_down`. Otherwise, `-l` defines a centered region.

**Prediction Options**

- `--rc`: Average forward and reverse complement predictions
- `--shifts`: Ensemble prediction shifts (comma-separated, default: "0")

**Statistics**

- `--stats`: Comma-separated list of statistics to compute (default: "logSUM")
  - `SUM`: Sum difference between mutated and reference predictions
  - `logSUM`: Log-transformed sum difference
  - `D1`: L1 norm of difference vector
  - `D2`: L2 norm of difference vector
  - `logD1/logD2`: Log-transformed norms

`covgene/`-prefixed stats (e.g. `covgene/logD2`) slice predictions to the gene's exon bins (respecting strand) and to gene tracks (targets `gene` column, or `assay` in RNA/RNA3/CAGE) before computing the score; requires `-g/--genes_gtf`.

#### Examples

**Basic ISM with 100bp region:**

```bash
python -m baskerville.scripts.hound_ism_bed \
    params.json model.pth regions.bed \
    -t targets.txt -f genome.fa \
    -l 100 --stats logSUM,logD2
```

**Asymmetric region (50bp upstream, 150bp downstream):**

```bash
python -m baskerville.scripts.hound_ism_bed \
    params.json model.pth regions.bed \
    -t targets.txt -f genome.fa \
    -u 50 -d 150 --stats logSUM
```

## Output Format

Results are saved as HDF5 files in the output directory:

**scores.h5** contains:

- `label`: Region identifiers from BED file
- `seqs`: Boolean array of reference sequences in mutation region (shape: `num_regions × mut_len × 4`)
- `<stat>`: ISM score arrays for each statistic (shape: `num_regions × mut_len × 4 × num_targets`)

### Output Structure

```
ism_output/
└── scores.h5
```

### Data Interpretation

For each region and each position in the mutation region:

- **Reference nucleotides**: Scores are set to zero (no mutation performed)
- **Alternative nucleotides**: Contain actual ISM scores representing predicted functional impact

The `seqs` dataset indicates which nucleotide is the reference at each position (True = reference nucleotide).

## ISM for Variants (VCF)

For analyzing variants from VCF files, use `hound_ism_snp` which extends the ISM approach to center the analysis around specific variant positions. This script accepts similar parameters but focuses the mutagenesis region around each variant in the VCF file, scoring each haplotype defined by the two alleles.

```bash
python -m baskerville.scripts.hound_ism_snp \
    params.json model.pth variants.vcf \
    -t targets.txt -f genome.fa \
    -l 200 --stats logSUM
```

### SNP Output Format

The SNP version produces a different HDF5 structure with separate groups for reference and alternative alleles:

**scores.h5** contains:

- `/ref/label`: Variant identifiers from VCF file
- `/ref/seqs`: Boolean array of reference sequences (shape: `num_variants × 4 × mut_len`)
- `/ref/<stat>`: ISM scores for reference allele context (shape: `num_variants × mut_len × 4 × num_targets`)
- `/alt/label`: Same variant identifiers
- `/alt/seqs`: Boolean array of alternative sequences (shape: `num_variants × 4 × mut_len`)
- `/alt/<stat>`: ISM scores for alternative allele context (shape: `num_variants × mut_len × 4 × num_targets`)

This structure allows direct comparison between ISM patterns in reference versus alternative allele contexts, providing insight into how the variant itself affects the mutagenesis landscape around the position.
