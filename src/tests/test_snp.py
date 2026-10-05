import json
import os
import pathlib
import tempfile
import pytest
import subprocess

import argparse

import h5py
import numpy as np
import pandas as pd
import torch

from baskerville.scripts.hound_snp_folds import build_snp_cmd
from baskerville.snps import parse_mix_dtype, partition_snp_stats


@pytest.fixture
def test_vcf_file():
    """Use the fixed test VCF file with known SNPs and correct reference alleles."""
    test_dir = str(pathlib.Path(__file__).parent)
    vcf_file = f"{test_dir}/data/sc3_snps.vcf"

    # Verify the file exists
    assert os.path.exists(vcf_file), f"Test VCF file not found: {vcf_file}"

    return vcf_file


def test_snp_basic(static_model_dir, test_vcf_file):
    """Test basic hound_snp functionality and output format."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "--stats",
            "logSUM,logD2",
            "--rc",
            "--shifts",
            "0,1",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check if command succeeded
        assert result.returncode == 0, f"hound_snp failed: {result.stderr}"

        # Check output files exist
        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file), "scores.h5 file not created"

        # Verify HDF5 file structure and output format
        with h5py.File(scores_file, "r") as h5f:
            # Check required datasets
            assert "snp" in h5f, "Missing 'snp' dataset"
            assert "chr" in h5f, "Missing 'chr' dataset"
            assert "pos" in h5f, "Missing 'pos' dataset"
            assert "ref_allele" in h5f, "Missing 'ref_allele' dataset"
            assert "alt_allele" in h5f, "Missing 'alt_allele' dataset"

            # Check statistics datasets (stored under cov/ prefix)
            for stat in ["logSUM", "logD2"]:
                assert f"cov/{stat}" in h5f, f"Missing 'cov/{stat}' dataset"

            # Check quantile datasets
            for stat in ["logSUM", "logD2"]:
                quantiles_key = f"cov/{stat}_quantiles"
                assert quantiles_key in h5f, f"Missing '{quantiles_key}' dataset"

            # Check that quantiles array exists
            assert "quantiles" in h5f, "Missing 'quantiles' dataset"

            # Check target information - should be copied as separate file
            targets_file_path = os.path.join(temp_dir, "targets_cov.txt")
            assert os.path.exists(targets_file_path), (
                "Missing targets_cov.txt file in output directory"
            )

            targets_df = pd.read_csv(targets_file_path, sep="\t", index_col=0)
            assert len(targets_df) == 2, "Expected 2 targets (from targets_sc3_me.txt)"

            # Check data consistency
            num_snps = len(h5f["snp"])
            assert num_snps == 4, f"Expected 4 SNPs (from sc3_snps.vcf), got {num_snps}"

            # Check shapes
            assert h5f["cov/logSUM"].shape[0] == num_snps, "cov/logSUM shape mismatch"
            assert h5f["cov/logSUM"].shape[1] == 2, (
                "Expected 2 targets (from targets_sc3_me.txt)"
            )

            # Verify data types and ranges
            for stat in ["logSUM", "logD2"]:
                scores = h5f[f"cov/{stat}"][:]
                assert scores.dtype == np.float16, f"cov/{stat} should be float16"
                assert not np.any(np.isnan(scores)), f"cov/{stat} contains NaN values"
                assert np.all(np.isfinite(scores)), (
                    f"cov/{stat} contains infinite values"
                )
                assert scores.var() > 0, f"cov/{stat} should have variance > 0"
                if stat == "logD2":
                    assert np.all(scores >= 0), f"cov/{stat} should be non-negative"


def test_snp_index_range(static_model_dir, test_vcf_file):
    """Test hound_snp with SNP index range."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-o",
            temp_dir,
            "--index_start",
            "0",
            "--index_end",
            "3",
            "-t",
            targets_file,
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"hound_snp failed: {result.stderr}"

        # Check output
        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file)

        with h5py.File(scores_file, "r") as h5f:
            num_snps = len(h5f["snp"])
            assert num_snps == 3, (
                f"Expected 3 SNPs (from index range 0-3), got {num_snps}"
            )


def test_snp_cluster_snps(static_model_dir, test_vcf_file):
    """Test hound_snp with SNP clustering."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-o",
            temp_dir,
            "-c",
            "0.25",  # Cluster SNPs within 25% of seq length
            "-t",
            targets_file,
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"hound_snp failed: {result.stderr}"

        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file)


@pytest.mark.parametrize("center_gene", [False, True])
def test_snp_gene_scoring(static_model_dir, test_vcf_file, center_gene):
    """Test hound_snp with gene-specific scoring."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",  # Option for gene annotations
            gtf_file,
            "--stats",
            "logSUM,covgene/logSUM,covgene/logD2",
            "--rc",
            params_file,
            model_file,
            test_vcf_file,
        ]

        if center_gene:
            cmd.insert(3, "--center_gene")
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"hound_snp with gene scoring failed: {result.stderr}"
        )

        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file)

        with h5py.File(scores_file, "r") as h5f:
            # Check for common SNP datasets
            assert "snp" in h5f
            assert "chr" in h5f
            assert "pos" in h5f
            num_snps_in_vcf = len(h5f["snp"])

            if center_gene:
                assert "cov/logSUM" not in h5f
            else:
                assert h5f["cov/logSUM"].shape == (num_snps_in_vcf, 2)

            # Check for gene-specific datasets
            assert "gene_ids" in h5f, "Missing 'gene_ids' dataset"
            assert "snp_idx" in h5f, "Missing 'snp_idx' (score mapping)"
            assert "gene_idx" in h5f, "Missing 'gene_idx' (score mapping)"

            num_genes = len(h5f["gene_ids"])
            num_snp_gene_pairs = len(h5f["snp_idx"])
            assert len(h5f["gene_idx"]) == num_snp_gene_pairs

            assert num_genes > 0, (
                f"Should find genes with sc3_snps.vcf, got {num_genes}"
            )
            assert num_snp_gene_pairs > 0, (
                f"Should find SNP-gene overlaps, got {num_snp_gene_pairs}"
            )

            # Verify indices are valid
            assert np.all(h5f["snp_idx"][:] < num_snps_in_vcf), (
                "Invalid snp_idx values (out of bounds)"
            )
            assert np.all(h5f["snp_idx"][:] >= 0), "Invalid snp_idx values (negative)"
            assert np.all(h5f["gene_idx"][:] < num_genes), (
                "Invalid gene_idx values (out of bounds)"
            )
            assert np.all(h5f["gene_idx"][:] >= 0), "Invalid gene_idx values (negative)"

            # Check gene-sliced coverage score datasets (covgene/ prefix)
            for stat in ["covgene/logSUM", "covgene/logD2"]:
                assert stat in h5f, f"Missing '{stat}' dataset"
                scores = h5f[stat][:]
                assert scores.shape[1] == 2, (
                    f"Expected 2 targets for {stat}, got {scores.shape[1]}"
                )
                assert scores.dtype == np.float16, f"{stat} should be float16"

                if num_snp_gene_pairs > 0:
                    assert scores.shape[0] == num_snp_gene_pairs, (
                        f"{stat} shape mismatch with snp_idx/gene_idx length"
                    )
                    assert not np.any(np.isnan(scores)), f"{stat} contains NaN values"
                    assert np.all(np.isfinite(scores)), (
                        f"{stat} contains infinite values"
                    )
                    if num_snp_gene_pairs > 1 and scores.shape[1] > 0:
                        assert scores.var() > 0, (
                            f"{stat} should have variance > 0 (pairs: {num_snp_gene_pairs})"
                        )

            # Check target information - should be copied as separate file
            targets_file_path = os.path.join(temp_dir, "targets_cov.txt")
            assert os.path.exists(targets_file_path), (
                "Missing targets_cov.txt file in output directory"
            )

            # Read targets file to verify content
            import pandas as pd

            targets_df = pd.read_csv(targets_file_path, sep="\t", index_col=0)
            assert len(targets_df) == 2, "Expected 2 targets (from targets_sc3_me.txt)"


def test_snp_gene_scoring_span(static_model_dir, test_vcf_file):
    """Test hound_snp with gene-specific scoring using --span."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",
            gtf_file,
            "--span",
            "--stats",
            "covgene/logSUM",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"hound_snp with gene scoring span failed: {result.stderr}"
        )

        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file)

        with h5py.File(scores_file, "r") as h5f:
            assert "gene_ids" in h5f
            assert "snp_idx" in h5f
            assert "gene_idx" in h5f
            assert "covgene/logSUM" in h5f
            # Shape sanity: rows == snp_gene pairs, columns == targets
            logsum = h5f["covgene/logSUM"][:]
            assert logsum.ndim == 2
            assert logsum.shape[1] == 2  # 2 targets


@pytest.mark.parametrize("center_gene", [False, True])
def test_snp_covgene_only_gene_tracks(static_model_dir, test_vcf_file, center_gene):
    """covgene/-only runs predict and report gene tracks only."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"

    with tempfile.TemporaryDirectory() as temp_dir:
        # mark the second track as non-gene
        targets_df = pd.read_csv(
            f"{test_dir}/data/targets_sc3_me.txt", sep="\t", index_col=0
        )
        targets_df.loc[1, "strand_pair"] = 1
        targets_df.loc[1, "gene"] = 0
        targets_file = f"{temp_dir}/targets.txt"
        targets_df.to_csv(targets_file, sep="\t")

        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",
            gtf_file,
            "--stats",
            "covgene/logSUM",
            params_file,
            model_file,
            test_vcf_file,
        ]
        if center_gene:
            cmd.append("--center_gene")

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Failed: {result.stderr}"

        with h5py.File(f"{temp_dir}/scores.h5", "r") as h5f:
            assert h5f["covgene/logSUM"].shape[1] == 1
        for name in ["targets_cov.txt", "targets_covgene.txt"]:
            out_df = pd.read_csv(f"{temp_dir}/{name}", sep="\t", index_col=0)
            assert list(out_df.identifier) == ["H3K4ME3_S0"]


def test_snp_cov_model_gene_scoring(static_model_dir, test_vcf_file):
    """Test coverage-only model with GTF: SNP-indexed logFC + gene-sliced logD2.

    Coverage models have no gene head, so gene/ datasets should NOT be created.
    """
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",
            gtf_file,
            "--stats",
            "logFC,covgene/logD2",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Failed: {result.stderr}"

        with h5py.File(f"{temp_dir}/scores.h5", "r") as h5f:
            num_snps = len(h5f["snp"])

            # SNP-indexed coverage logFC (full-sequence)
            assert "cov/logFC" in h5f, "Missing 'cov/logFC'"
            logfc = h5f["cov/logFC"][:]
            assert logfc.shape == (num_snps, 2)
            assert logfc.dtype == np.float16
            assert not np.any(np.isnan(logfc))

            # gene-sliced logD2 (pair-indexed)
            assert "covgene/logD2" in h5f, "Missing 'covgene/logD2'"
            assert "gene_ids" in h5f
            assert "snp_idx" in h5f
            assert "gene_idx" in h5f
            num_pairs = len(h5f["snp_idx"])
            assert num_pairs > 0
            logd2 = h5f["covgene/logD2"][:]
            assert logd2.shape == (num_pairs, 2)
            assert logd2.dtype == np.float16
            assert not np.any(np.isnan(logd2))

            # no gene head → no gene/ datasets
            all_keys = []
            h5f.visititems(lambda name, obj: all_keys.append(name))
            assert not any(k.startswith("gene/") for k in all_keys)


def test_snp_gene_head_model_scoring(model_gene_dir, test_vcf_file):
    """Test gene head model with GTF: cov/logFC + covgene/logD2 + gene/logFC.

    The gene head model has both coverage and gene heads. gene/logFC stores
    the gene head logFC (explicitly requested, not auto-added).
    """
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3_gene.json"
    model_file = f"{model_gene_dir}/model_best.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",
            gtf_file,
            "--stats",
            "logFC,covgene/logD2,gene/logFC",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Failed: {result.stderr}"

        with h5py.File(f"{temp_dir}/scores.h5", "r") as h5f:
            num_snps = len(h5f["snp"])
            num_pairs = len(h5f["snp_idx"])
            assert num_pairs > 0

            # SNP-indexed coverage logFC (full-sequence)
            assert "cov/logFC" in h5f
            logfc_cov = h5f["cov/logFC"][:]
            assert logfc_cov.shape == (num_snps, 2)
            assert logfc_cov.dtype == np.float16

            # gene-sliced coverage logD2
            assert "covgene/logD2" in h5f
            logd2 = h5f["covgene/logD2"][:]
            assert logd2.shape == (num_pairs, 2)

            # gene head logFC (explicitly requested)
            assert "gene/logFC" in h5f
            logfc_gene_head = h5f["gene/logFC"][:]
            assert logfc_gene_head.shape == (num_pairs, 2)  # 2 gene targets
            assert logfc_gene_head.dtype == np.float16
            assert not np.any(np.isnan(logfc_gene_head))


# Test removed - no longer needed since we use fixed sc3_snps.vcf file


def test_snp_normalization(static_model_dir, test_vcf_file):
    """Test hound_snp with normalization file."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # First, create a normalization file by running without normalization
        norm_dir = f"{temp_dir}/norm"
        os.makedirs(norm_dir)

        cmd_norm = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            norm_dir,
            "--stats",
            "logSUM,logD2",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd_norm, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Creating normalization file failed: {result.stderr}"
        )

        norm_file = f"{norm_dir}/scores.h5"
        assert os.path.exists(norm_file), "Normalization file not created"

        # Now run with normalization
        test_dir_norm = f"{temp_dir}/with_norm"
        os.makedirs(test_dir_norm)

        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            test_dir_norm,
            "-n",
            norm_file,
            "--stats",
            "logSUM,logD2",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"hound_snp with normalization failed: {result.stderr}"
        )

        # Check output files exist
        scores_file = f"{test_dir_norm}/scores.h5"
        assert os.path.exists(scores_file), "scores.h5 file not created"

        # Verify HDF5 file structure
        with h5py.File(scores_file, "r") as h5f:
            # Check required datasets
            assert "snp" in h5f, "Missing 'snp' dataset"
            assert "quantiles" in h5f, "Missing 'quantiles' dataset"

            # Check quantile datasets exist
            for stat in ["logSUM", "logD2"]:
                quantiles_key = f"cov/{stat}_quantiles"
                assert quantiles_key in h5f, f"Missing '{quantiles_key}' dataset"

            # Verify that quantiles were computed (should have same structure as norm file)
            with h5py.File(norm_file, "r") as norm_h5:
                for stat in ["logSUM", "logD2"]:
                    quantiles_key = f"cov/{stat}_quantiles"
                    if quantiles_key in norm_h5:
                        # Quantiles should be the same shape (targets dimension)
                        assert (
                            h5f[quantiles_key].shape[1]
                            == norm_h5[quantiles_key].shape[1]
                        ), f"Quantiles shape mismatch for {stat}"


def test_snp_normalization_missing_file(static_model_dir, test_vcf_file):
    """Test hound_snp with missing normalization file (should fail)."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a non-existent normalization file
        missing_norm_file = f"{temp_dir}/missing_norm.h5"

        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-n",
            missing_norm_file,
            "--stats",
            "logSUM",
            params_file,
            model_file,
            test_vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        # Should fail with FileNotFoundError
        assert result.returncode != 0, (
            "Expected failure with missing normalization file"
        )
        assert "not found" in result.stderr.lower(), (
            f"Expected 'not found' error, got: {result.stderr}"
        )


def test_vcf_fixture_validity(test_vcf_file):
    """Test that the test VCF file is valid and contains expected SNPs."""
    # VCF file should exist
    assert os.path.exists(test_vcf_file)

    # Verify it's a valid VCF format
    with open(test_vcf_file, "r") as f:
        content = f.read()
        assert "##fileformat=VCFv4.2" in content
        assert "#CHROM" in content

    # Check that we have the expected number of variants
    lines = content.strip().split("\n")
    variant_lines = [line for line in lines if not line.startswith("#")]
    assert len(variant_lines) == 4, (
        f"Expected 4 variants in sc3_snps.vcf, got {len(variant_lines)}"
    )

    # Verify specific SNPs are present
    expected_positions = [1000, 2000, 3000, 4000]
    positions_found = []
    for line in variant_lines:
        fields = line.split("\t")
        assert fields[0] == "chrI", f"Expected chromosome chrI, got {fields[0]}"
        positions_found.append(int(fields[1]))

    assert positions_found == expected_positions, (
        f"Expected positions {expected_positions}, got {positions_found}"
    )


@pytest.mark.parametrize(
    "stats,expected",
    [
        ([], ([], [], [])),
        (
            ["logFC", "cov/logD2", "gene/logFC", "covgene/nD1", "logFC"],
            (["cov/logFC", "cov/logD2", "cov/logFC"], ["covgene/nD1"], ["gene/logFC"]),
        ),
        (
            ["other/name", "covgene/logSUM", "gene/logFC", "covgene/nD1"],
            (["cov/other/name"], ["covgene/logSUM", "covgene/nD1"], ["gene/logFC"]),
        ),
    ],
)
def test_partition_snp_stats(stats, expected):
    original = stats.copy()
    assert partition_snp_stats(stats) == expected
    assert stats == original


def test_parse_mix_dtype_warns_below_float32(capsys):
    assert parse_mix_dtype("float32") is torch.float32
    assert "WARNING" not in capsys.readouterr().err
    assert parse_mix_dtype("bfloat16") is torch.bfloat16
    assert "WARNING" in capsys.readouterr().err


def test_snp_folds_passes_mix_dtype():
    args = argparse.Namespace(
        mix_dtype="bfloat16", params_file="p.json", vcf_file="v.vcf"
    )
    assert "-m bfloat16" in build_snp_cmd(args, "m.pth", 0, 1)
    args.mix_dtype = "float32"
    assert "-m" not in build_snp_cmd(args, "m.pth", 0, 1).split()


def test_snp_folds_passes_compile():
    args = argparse.Namespace(compile=True, params_file="p.json", vcf_file="v.vcf")
    assert "--compile" in build_snp_cmd(args, "m.pth", 0, 1).split()


def test_snp_mix_dtype_takes_effect(static_model_dir, test_vcf_file):
    """-m bfloat16 must change the model's precision (it was once a silent no-op)."""
    test_dir = str(pathlib.Path(__file__).parent)
    scores = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for mix_dtype in ["float32", "bfloat16"]:
            out_dir = f"{temp_dir}/{mix_dtype}"
            cmd = [
                "python",
                "-m",
                "baskerville.scripts.hound_snp",
                "-f",
                f"{test_dir}/data/sc3.fa.gz",
                "-t",
                f"{test_dir}/data/targets_sc3_me.txt",
                "-o",
                out_dir,
                "--stats",
                "logSUM",
                "-m",
                mix_dtype,
                f"{test_dir}/data/params_sc3.json",
                f"{static_model_dir}/sc3_model.pth",
                test_vcf_file,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"hound_snp failed: {result.stderr}"
            assert ("WARNING: --mix_dtype" in result.stderr) == (mix_dtype != "float32")
            with h5py.File(f"{out_dir}/scores.h5", "r") as h5:
                scores[mix_dtype] = h5["cov/logSUM"][:]
    assert not np.array_equal(scores["float32"], scores["bfloat16"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile targets GPU")
def test_snp_compile_matches_eager(static_model_dir, test_vcf_file):
    """--compile runs and reproduces eager scores up to fusion reassociation."""
    test_dir = str(pathlib.Path(__file__).parent)
    scores = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for compile_flag in [[], ["--compile"]]:
            out_dir = f"{temp_dir}/{len(compile_flag)}"
            cmd = [
                "python",
                "-m",
                "baskerville.scripts.hound_snp",
                "-f",
                f"{test_dir}/data/sc3.fa.gz",
                "-t",
                f"{test_dir}/data/targets_sc3_me.txt",
                "-o",
                out_dir,
                "--stats",
                "logSUM",
                "--rc",
                *compile_flag,
                f"{test_dir}/data/params_sc3.json",
                f"{static_model_dir}/sc3_model.pth",
                test_vcf_file,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"hound_snp failed: {result.stderr}"
            with h5py.File(f"{out_dir}/scores.h5", "r") as h5:
                scores[len(compile_flag)] = h5["cov/logSUM"][:].astype("float32")
    np.testing.assert_allclose(scores[1], scores[0], rtol=1e-2, atol=1e-2)
