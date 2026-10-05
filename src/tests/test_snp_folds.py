#!/usr/bin/env python
"""
Test hound_snp_folds workflow with both gene-agnostic and gene-specific modes.
"""

import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import pysam
import pytest
import shutil
import tempfile

from baskerville.scripts.hound_snp_folds import snp_folds
from baskerville.multi import collect_scores


class MockArgs:
    """Mock arguments object for hound_snp_folds."""

    def __init__(self, **kwargs):
        # Default values
        self.cluster_pct = 0
        self.genome_fasta = None
        self.genes_gtf = None
        self.head = 0
        self.indel_stitch = False
        self.out_dir = "snp_out"
        self.rc = False
        self.shifts = "0"
        self.span = False
        self.snp_stats = "logSUM"
        self.targets_file = None
        self.crosses = 1
        self.conda_env = None
        self.embed = False
        self.num_folds = None
        self.fold_subset_list = None
        self.backend = "local"  # Always run jobs in-process for tests
        self.name = "snp"
        self.parallel_jobs = 1
        self.job_size = 2  # Small job size for testing
        self.queue = "geforce"
        self.norm_subdir = None  # optional normalization subdirectory

        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_real_model_dir(base_dir, num_folds=2, num_crosses=1):
    """Create a real models directory structure with the pre-trained model."""
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Path to the real pre-trained model
    source_model = os.path.join(pathlib.Path(__file__).parent, "data", "sc3_model.pth")

    for fi in range(num_folds):
        for ci in range(num_crosses):
            fold_dir = os.path.join(models_dir, f"f{fi}c{ci}", "train")
            os.makedirs(fold_dir, exist_ok=True)

            # Copy the real model file
            model_file = os.path.join(fold_dir, "model_best.pth")
            shutil.copy2(source_model, model_file)

    return models_dir


def create_mock_vcf(vcf_file, num_snps=4, chrom="chrI"):
    """Create a simple mock VCF file with correct reference alleles."""
    # Get the correct reference file path
    test_dir = pathlib.Path(__file__).parent
    fasta_file = f"{test_dir}/data/sc3.fa.gz"

    with open(vcf_file, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Get the actual reference bases
        fasta = pysam.FastaFile(fasta_file)
        for i in range(num_snps):
            pos = 1000 + i * 1000  # Use smaller positions that exist in the test genome
            snp_id = f"rs{i + 1}"
            ref = fasta.fetch(chrom, pos - 1, pos)  # 0-based indexing for pysam
            alt = "G" if ref != "G" else "C"  # Choose a different alt allele
            f.write(f"{chrom}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\t.\t.\n")
        fasta.close()


def create_mock_job_output(job_dir, snp_indices, num_targets=2, gene_mode=False):
    """Create mock job output HDF5 file."""
    os.makedirs(job_dir, exist_ok=True)
    scores_file = os.path.join(job_dir, "scores.h5")

    # Create the targets_cov.txt file that hound_snp creates
    targets_file = os.path.join(job_dir, "targets_cov.txt")
    index_names = [f"target_{i}" for i in range(num_targets)]
    targets_data = {
        "identifier": index_names,
        "description": [f"Target {i} description" for i in range(num_targets)],
        "strand": ["+"] * num_targets,
    }
    targets_df = pd.DataFrame(targets_data, index=index_names)
    targets_df.to_csv(targets_file, sep="\t")

    num_snps = len(snp_indices)

    with h5py.File(scores_file, "w") as h5f:
        # Basic SNP info
        snp_ids = [f"chrI_{1000 + i * 1000}_A_T" for i in snp_indices]
        h5f.create_dataset("snp", data=np.array(snp_ids, dtype="S"))
        h5f.create_dataset("chr", data=np.array([b"chrI"] * num_snps, dtype="S"))
        h5f.create_dataset(
            "pos", data=np.array([1000 + i * 1000 for i in snp_indices], dtype="uint32")
        )
        h5f.create_dataset("ref_allele", data=np.array([b"A"] * num_snps, dtype="S"))
        h5f.create_dataset("alt_allele", data=np.array([b"T"] * num_snps, dtype="S"))

        # No targets group - targets file is copied separately

        if gene_mode:
            # Gene-specific mode
            gene_ids = [f"gene_{i}" for i in range(2)]  # 2 genes for testing
            h5f.create_dataset("gene_ids", data=np.array(gene_ids, dtype="S"))

            # Create SNP-gene pairs (each SNP associated with each gene)
            num_pairs = num_snps * len(gene_ids)
            snp_idx = np.repeat(range(num_snps), len(gene_ids))
            gene_idx = np.tile(range(len(gene_ids)), num_snps)

            h5f.create_dataset("snp_idx", data=snp_idx.astype(np.int32))
            h5f.create_dataset("gene_idx", data=gene_idx.astype(np.int32))

            # Score data (one score per SNP-gene pair)
            scores = np.random.randn(num_pairs, num_targets).astype(np.float32)
            h5f.create_dataset("covgene/logSUM", data=scores)

        else:
            # Gene-agnostic mode (direct SNP scoring)
            scores = np.random.randn(num_snps, num_targets).astype(np.float32)
            h5f.create_dataset("cov/logSUM", data=scores)


def test_snp_folds_genome():
    """Test hound_snp_folds workflow in gene-agnostic mode with real model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        vcf_file = os.path.join(temp_dir, "test.vcf")
        create_mock_vcf(vcf_file, num_snps=4)

        # Create models directory structure with real model
        models_dir = create_real_model_dir(temp_dir, num_folds=2, num_crosses=1)

        # Create output directory
        out_dir = os.path.join(temp_dir, "snp_out")

        # Real args for actual execution
        args = MockArgs(
            params_file=f"{pathlib.Path(__file__).parent}/data/params_sc3.json",
            models_dir=models_dir,
            vcf_file=vcf_file,
            out_dir=out_dir,
            job_size=2,  # 2 SNPs per job, so 2 jobs total
            num_folds=2,
            crosses=1,
            targets_file=f"{pathlib.Path(__file__).parent}/data/targets_sc3_ac.txt",
            genome_fasta=f"{pathlib.Path(__file__).parent}/data/sc3.fa.gz",
        )

        # Run the real workflow
        snp_folds(args)

        # Verify outputs were created
        for fi in range(2):  # 2 folds
            fold_out_dir = os.path.join(out_dir, f"f{fi}c0")
            final_scores_file = os.path.join(fold_out_dir, "scores.h5")

            assert os.path.exists(final_scores_file), (
                f"Missing final scores file for fold {fi}"
            )

            # Verify the final scores file has correct structure
            with h5py.File(final_scores_file, "r") as h5f:
                assert "snp" in h5f
                assert "cov/logSUM" in h5f
                assert "gene_ids" not in h5f  # Should not be in gene-agnostic mode

                # Should have all 4 SNPs
                assert len(h5f["snp"]) == 4
                assert h5f["cov/logSUM"].shape[0] == 4

                # Verify we have actual predicted scores (not just zeros)
                scores = h5f["cov/logSUM"][:]
                assert np.any(scores != 0), (
                    "All scores are zero - model may not be working"
                )

                # Check quantile datasets
                for stat in ["logSUM"]:
                    quantiles_key = f"cov/{stat}_quantiles"
                    assert quantiles_key in h5f, f"Missing '{quantiles_key}' dataset"

                # Check that quantiles array exists
                assert "quantiles" in h5f, "Missing 'quantiles' dataset"

        # Verify ensemble output
        ensemble_dir = os.path.join(out_dir, "ensemble")
        ensemble_scores_file = os.path.join(ensemble_dir, "scores.h5")
        assert os.path.exists(ensemble_scores_file), "Missing ensemble scores file"


def test_snp_folds_gene():
    """Test hound_snp_folds workflow in gene-specific mode using real model execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use the real test VCF file we created
        test_dir = pathlib.Path(__file__).parent
        vcf_file = f"{test_dir}/data/sc3_snps.vcf"
        gtf_file = f"{test_dir}/data/sc3_genes.gtf"

        # Create models directory structure with real model
        models_dir = create_real_model_dir(temp_dir, num_folds=2, num_crosses=1)

        # Create output directory
        out_dir = os.path.join(temp_dir, "snp_out")

        # Args for gene-specific mode (real execution)
        args = MockArgs(
            params_file=f"{pathlib.Path(__file__).parent}/data/params_sc3.json",
            models_dir=models_dir,
            vcf_file=vcf_file,
            out_dir=out_dir,
            job_size=4,  # Process all 4 SNPs in one job to avoid zero-score issues
            num_folds=2,
            crosses=1,
            genes_gtf=gtf_file,  # This triggers gene-specific mode
            targets_file=f"{pathlib.Path(__file__).parent}/data/targets_sc3_ac.txt",
            genome_fasta=f"{pathlib.Path(__file__).parent}/data/sc3.fa.gz",
            processes=1,  # Use single process for test stability
        )
        args.snp_stats = "covgene/logSUM"

        # Run the workflow with real execution
        snp_folds(args)

        # Verify outputs were created
        for fi in range(2):  # 2 folds
            fold_out_dir = os.path.join(out_dir, f"f{fi}c0")
            final_scores_file = os.path.join(fold_out_dir, "scores.h5")

            assert os.path.exists(final_scores_file), (
                f"Missing final scores file for fold {fi}"
            )

            # Verify the final scores file has correct structure for gene-specific mode
            with h5py.File(final_scores_file, "r") as h5f:
                assert "snp" in h5f
                assert "gene_ids" in h5f, "Should have gene_ids in gene-specific mode"
                assert "snp_idx" in h5f
                assert "gene_idx" in h5f
                assert "covgene/logSUM" in h5f

                # Check quantile datasets
                assert "quantiles" in h5f, "Missing 'quantiles' dataset"
                assert "covgene/logSUM_quantiles" in h5f, (
                    "Missing covgene/logSUM_quantiles"
                )

                print(
                    f"Fold {fi} - SNPs: {len(h5f['snp'])}, Genes: {len(h5f['gene_ids'])}"
                )
                print(f"Fold {fi} - SNP-gene pairs: {len(h5f['snp_idx'])}")

                # Should have all 4 SNPs and some genes (we found 7 genes in our test)
                assert len(h5f["snp"]) == 4
                assert len(h5f["gene_ids"]) > 0, "Should have found some genes"

                # Should have SNP-gene pairs
                assert len(h5f["snp_idx"]) > 0, "Should have found SNP-gene overlaps"
                assert len(h5f["gene_idx"]) > 0, "Should have found SNP-gene overlaps"
                assert h5f["covgene/logSUM"].shape[0] > 0, (
                    "Should have scores for SNP-gene pairs"
                )

                # Verify index ranges are correct
                assert np.max(h5f["snp_idx"][:]) < 4, "SNP indices should be < 4"
                assert np.max(h5f["gene_idx"][:]) < len(h5f["gene_ids"]), (
                    "Gene indices should be < number of genes"
                )

        # Verify ensemble output
        ensemble_dir = os.path.join(out_dir, "ensemble")
        ensemble_scores_file = os.path.join(ensemble_dir, "scores.h5")
        assert os.path.exists(ensemble_scores_file), "Missing ensemble scores file"

        # Verify ensemble file structure
        with h5py.File(ensemble_scores_file, "r") as h5f:
            assert "snp" in h5f
            assert "gene_ids" in h5f, (
                "Ensemble should have gene_ids in gene-specific mode"
            )
            assert "snp_idx" in h5f
            assert "gene_idx" in h5f
            assert "covgene/logSUM" in h5f

            print(f"Ensemble - SNPs: {len(h5f['snp'])}, Genes: {len(h5f['gene_ids'])}")
            print(f"Ensemble - SNP-gene pairs: {len(h5f['snp_idx'])}")

            # Should have all 4 SNPs and some genes
            assert len(h5f["snp"]) == 4
            assert len(h5f["gene_ids"]) > 0, "Ensemble should have found some genes"


def test_collect_scores_mode_detection():
    """Test that collect_scores correctly detects gene-agnostic vs gene-specific modes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test gene-agnostic mode detection
        job_dir = os.path.join(temp_dir, "gene_agnostic", "job0")
        create_mock_job_output(job_dir, [0, 1], gene_mode=False)

        out_dir = os.path.join(temp_dir, "gene_agnostic")
        collect_scores(out_dir, 1)

        final_file = os.path.join(out_dir, "scores.h5")
        with h5py.File(final_file, "r") as h5f:
            assert "gene_ids" not in h5f
            assert "snp_idx" not in h5f
            assert "gene_idx" not in h5f

        # Test gene-specific mode detection
        job_dir = os.path.join(temp_dir, "gene_specific", "job0")
        create_mock_job_output(job_dir, [0, 1], gene_mode=True)

        out_dir = os.path.join(temp_dir, "gene_specific")
        collect_scores(out_dir, 1)

        final_file = os.path.join(out_dir, "scores.h5")
        with h5py.File(final_file, "r") as h5f:
            assert "gene_ids" in h5f
            assert "snp_idx" in h5f
            assert "gene_idx" in h5f


def test_collect_scores_gene_mode_index_remapping():
    """Test that collect_scores correctly remaps indices in gene-specific mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = temp_dir

        # Create two jobs with overlapping gene sets but different SNPs
        # Job 0: SNPs 0-1, Genes 0-1 (gene_0, gene_1)
        job0_dir = os.path.join(out_dir, "job0")
        create_mock_job_output(job0_dir, [0, 1], gene_mode=True)

        # Job 1: SNPs 2-3, Genes 0-1 (same genes, different SNPs)
        job1_dir = os.path.join(out_dir, "job1")
        create_mock_job_output(job1_dir, [2, 3], gene_mode=True)

        # Collect scores
        collect_scores(out_dir, 2)

        # Verify the final output
        final_file = os.path.join(out_dir, "scores.h5")
        with h5py.File(final_file, "r") as h5f:
            # Should have 4 SNPs total
            assert len(h5f["snp"]) == 4

            # Should have 2 unique genes
            assert len(h5f["gene_ids"]) == 2

            # Should have 4 SNPs × 2 genes = 8 pairs
            assert len(h5f["snp_idx"]) == 8
            assert len(h5f["gene_idx"]) == 8

            # Verify index ranges
            snp_idx = h5f["snp_idx"][:]
            gene_idx = h5f["gene_idx"][:]

            assert np.min(snp_idx) >= 0
            assert np.max(snp_idx) < 4  # SNP indices should be 0-3
            assert np.min(gene_idx) >= 0
            assert np.max(gene_idx) < 2  # Gene indices should be 0-1

            # Verify all SNPs are represented
            unique_snp_indices = np.unique(snp_idx)
            assert len(unique_snp_indices) == 4
            assert set(unique_snp_indices) == {0, 1, 2, 3}

            # Verify all genes are represented
            unique_gene_indices = np.unique(gene_idx)
            assert len(unique_gene_indices) == 2
            assert set(unique_gene_indices) == {0, 1}


if __name__ == "__main__":
    pytest.main([__file__])
