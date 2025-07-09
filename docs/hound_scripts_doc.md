# Baskerville Hound Scripts Documentation

This document describes the main functionality and differences between five key prediction scripts in the Baskerville suite.

## Overview

The Baskerville package includes several "hound" scripts for making genomic predictions using trained deep learning models. Each script serves a specific purpose in the genomic prediction pipeline, from basic sequence prediction to variant effect analysis.

| Script | Input Type | Analysis Focus | Output Type | Parallelization |
|--------|------------|----------------|-------------|-----------------|
| `hound_predbed.py` | BED regions | Sequence prediction | Per-region predictions | Single process |
| `hound_predvcf.py` | VCF variants | Full window prediction | Ref/alt predictions | Single process |
| `hound_snp_slurm.py` | VCF variants | Variant effects | Effect statistics | SLURM cluster |
| `hound_snp.py` | VCF variants | Variant effects | Effect statistics | Single/multi-worker |
| `hound_snpgene.py` | VCF + GTF | Gene-context effects | Gene-level statistics | Single/multi-worker |


## Script Descriptions

### 1. `hound_predbed.py` - BED File Sequence Prediction

**Purpose**: Predicts sequences from genomic regions defined in a BED file.

**Key Features**:
- Takes genomic coordinates from a BED file as input
- Generates model predictions for specified genomic regions
- Supports both full predictions and summed predictions across sequence length
- Can generate BigWig files for genome browser visualization
- Supports embedding layer extraction for sequence representations

**Input**:
- BED file with genomic coordinates
- Model parameters and weights
- Genome FASTA file
- Optional targets file for specific prediction tracks

**Output**:
- HDF5 file with predictions for each BED region
- Optional BigWig files for visualization
- Predictions can be per-position or summed across regions

---

### 2. `hound_predvcf.py` - VCF Variant Window Prediction

**Purpose**: Predicts full genomic windows for variants from a VCF file, generating predictions for both reference and alternate alleles.

**Key Features**:
- Processes SNPs from VCF files (INDELs use reference sequence for alternate allele)
- Generates full-window predictions at 32bp-bin resolution
- Provides separate predictions for reference and alternate alleles
- Focuses on complete genomic context around variants

**Input**:
- VCF file with genetic variants
- Model parameters and weights
- Genome FASTA file
- Targets file (required for stranded datasets)

**Output**:
- HDF5 file with separate reference and alternate predictions
- Full prediction windows for each variant
- Variant metadata (chromosome, position, ref/alt alleles)

---

### 3. `hound_snp_slurm.py` - Cluster-Parallelized SNP Analysis

**Purpose**: Orchestrates parallel computation of variant effect predictions across a SLURM cluster.

**Key Features**:
- Distributes SNP analysis across multiple compute nodes
- Manages job submission and monitoring via SLURM
- Handles result aggregation from parallel workers
- Optimized for large-scale variant effect studies

**Input**:
- Same inputs as `hound_snp.py`
- Additional SLURM configuration parameters
- Number of parallel processes to spawn

**Output**:
- Aggregated results from all parallel workers
- Job management and monitoring files
- Same final output format as `hound_snp.py`

---

### 4. `hound_snp.py` - SNP Variant Effect Prediction

**Purpose**: Computes variant effect predictions for SNPs in a VCF file using various statistical measures.

**Key Features**:
- Calculates multiple variant effect statistics (logSUM, SAD, D1, D2, JS, etc.)
- Supports SNP clustering to reduce redundant predictions
- Handles both single-worker and multi-worker execution modes
- Comprehensive suite of distance and information-theory based metrics

**Available Statistics**:
- **Basic Sum-based**: SUM, logSUM, logSED, sqrtSUM
- **Maximum Effect**: SAX (finds position of maximum difference)
- **Distance-based**: D1/D2 (L1/L2 distances), logD1/logD2, sqrtD1/sqrtD2
- **Information Theory**: JS (Jensen-Shannon divergence), logJS

**Input**:
- VCF file with SNPs
- Model parameters and weights
- Genome FASTA file
- Targets file

**Output**:
- HDF5 file with variant effect scores
- Statistical summaries for each SNP across all prediction targets
- Configurable output statistics

---

### 5. `hound_snpgene.py` - Gene-Context SNP Analysis

**Purpose**: Computes variant effect predictions for SNPs with respect to gene exons defined in a GTF file.

**Key Features**:
- Focuses variant analysis on gene regions (exons)
- Integrates GTF gene annotations
- Supports gene span aggregation
- Gene clustering to optimize predictions
- Same statistical framework as `hound_snp.py`

**Input**:
- VCF file with SNPs
- GTF file with gene annotations
- Model parameters and weights
- Genome FASTA file
- Targets file

## Common Parameters

All scripts share several common parameters:
- `--rc`: Average forward and reverse complement predictions
- `--shifts`: Ensemble prediction shifts
- `--targets_file`: Specify target indexes and labels
- `--untransform_old`: Apply legacy untransformation
- `--float16`: Use mixed precision for memory efficiency
- `--gcs`: Support for Google Cloud Storage input/output

## Choosing the Right Script

- **Use `hound_predbed.py`** when you want predictions for specific genomic regions
- **Use `hound_predvcf.py`** when you need full prediction windows around variants
- **Use `hound_snp.py`** for variant effect analysis with statistical summaries
- **Use `hound_snp_slurm.py`** for large-scale variant studies requiring cluster computing
- **Use `hound_snpgene.py`** when focusing on variant effects within gene contexts

## Technical Notes

- All scripts support GPU acceleration and mixed precision
- VCF-based scripts currently focus on SNPs (limited INDEL support)
- BigWig output is only available in `hound_predbed.py`
- Gene-context analysis requires properly formatted GTF files
- Cluster parallelization requires SLURM job scheduler setup 