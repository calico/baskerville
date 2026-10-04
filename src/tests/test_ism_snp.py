import os
import pathlib
import tempfile
import pytest
import subprocess

import h5py
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def ism_output(static_model_dir):
    """Fixture that runs ISM once and provides the output file for multiple tests."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    vcf_file = f"{test_dir}/data/sc3_snps.vcf"

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_ism_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-l",
            "20",  # Short mutation length for faster testing
            "--stats",
            "logSUM,logD2",
            "--rc",
            "--shifts",
            "0",
            params_file,
            model_file,
            vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"ISM fixture failed: {result.stderr}")

        scores_file = f"{temp_dir}/scores.h5"

        yield scores_file


def test_ism_snp_basic(ism_output):
    """Test basic hound_ism_snp functionality using shared ISM output."""
    scores_file = ism_output

    # Check output file exists
    assert os.path.exists(scores_file)

    # Verify output structure
    with h5py.File(scores_file, "r") as h5f:
        # Check basic datasets
        assert "label" in h5f
        assert "ref" in h5f
        assert "alt" in h5f

        # Check ref and alt groups have the expected datasets
        ref_group = h5f["ref"]
        alt_group = h5f["alt"]

        assert "seqs" in ref_group
        assert "cov/logSUM" in ref_group
        assert "cov/logD2" in ref_group

        assert "seqs" in alt_group
        assert "cov/logSUM" in alt_group
        assert "cov/logD2" in alt_group

        # Check dimensions
        num_snps = len(h5f["label"])  # One label per SNP
        mut_len = 20
        num_nucleotides = 4
        num_targets = 2  # from targets_sc3_me.txt

        assert ref_group["seqs"].shape == (num_snps, num_nucleotides, mut_len)
        assert ref_group["cov/logSUM"].shape == (
            num_snps,
            mut_len,
            num_nucleotides,
            num_targets,
        )
        assert ref_group["cov/logD2"].shape == (
            num_snps,
            mut_len,
            num_nucleotides,
            num_targets,
        )

        # Check alt group has same shapes
        assert alt_group["seqs"].shape == ref_group["seqs"].shape
        assert alt_group["cov/logSUM"].shape == ref_group["cov/logSUM"].shape
        assert alt_group["cov/logD2"].shape == ref_group["cov/logD2"].shape

        # Check data types
        assert ref_group["cov/logSUM"].dtype == np.float16
        assert ref_group["cov/logD2"].dtype == np.float16
        assert ref_group["seqs"].dtype == bool
        assert alt_group["cov/logSUM"].dtype == np.float16
        assert alt_group["cov/logD2"].dtype == np.float16
        assert alt_group["seqs"].dtype == bool

        # Verify scores exist and are finite for both ref and alt
        ref_logsum_scores = ref_group["cov/logSUM"][:]
        ref_logd2_scores = ref_group["cov/logD2"][:]
        alt_logsum_scores = alt_group["cov/logSUM"][:]
        alt_logd2_scores = alt_group["cov/logD2"][:]

        assert not np.all(ref_logsum_scores == 0), "All ref logSUM scores are zero"
        assert not np.all(ref_logd2_scores == 0), "All ref logD2 scores are zero"
        assert not np.all(alt_logsum_scores == 0), "All alt logSUM scores are zero"
        assert not np.all(alt_logd2_scores == 0), "All alt logD2 scores are zero"

        assert np.all(np.isfinite(ref_logsum_scores)), (
            "ref logSUM contains non-finite values"
        )
        assert np.all(np.isfinite(ref_logd2_scores)), (
            "ref logD2 contains non-finite values"
        )
        assert np.all(np.isfinite(alt_logsum_scores)), (
            "alt logSUM contains non-finite values"
        )
        assert np.all(np.isfinite(alt_logd2_scores)), (
            "alt logD2 contains non-finite values"
        )


def test_ism_snp_sequence_encoding(ism_output):
    """Test that sequence encoding works correctly using shared ISM output."""
    scores_file = ism_output

    with h5py.File(scores_file, "r") as h5f:
        ref_seqs = h5f["ref"]["seqs"][:]
        alt_seqs = h5f["alt"]["seqs"][:]

        # Each position should have exactly one nucleotide set to True
        for seqs, seq_type in [(ref_seqs, "ref"), (alt_seqs, "alt")]:
            for seq_idx in range(seqs.shape[0]):
                for pos_idx in range(seqs.shape[2]):  # shape is now [seq, channel, pos]
                    pos_sum = np.sum(seqs[seq_idx, :, pos_idx])  # sum across channels
                    assert pos_sum == 1, (
                        f"Position {pos_idx} in {seq_type} sequence {seq_idx} has {pos_sum} nucleotides, expected 1"
                    )


def test_ism_snp_score_patterns(ism_output):
    """Test that ISM scores show expected patterns using shared ISM output."""
    scores_file = ism_output

    with h5py.File(scores_file, "r") as h5f:
        ref_seqs = h5f["ref"]["seqs"][:]
        alt_seqs = h5f["alt"]["seqs"][:]
        ref_scores = h5f["ref"]["cov/logSUM"][:]
        alt_scores = h5f["alt"]["cov/logSUM"][:]

        # Check that reference positions (where the sequence has the nucleotide)
        # don't have scores (should be zero because we skip non-reference)
        for seqs, scores, seq_type in [
            (ref_seqs, ref_scores, "ref"),
            (alt_seqs, alt_scores, "alt"),
        ]:
            for seq_idx in range(seqs.shape[0]):
                for pos_idx in range(seqs.shape[2]):  # shape is now [seq, channel, pos]
                    for nuc_idx in range(4):
                        if seqs[seq_idx, nuc_idx, pos_idx]:  # Reference nucleotide
                            # All scores for this position should be zero (not computed)
                            ref_pos_scores = scores[seq_idx, pos_idx, nuc_idx, :]
                            assert np.all(ref_pos_scores == 0), (
                                f"{seq_type} reference position has non-zero scores: {ref_pos_scores}"
                            )
                        else:  # Alternative nucleotide
                            # Should have actual scores (non-zero for at least some)
                            alt_pos_scores = scores[seq_idx, pos_idx, nuc_idx, :]
                            # Note: scores could legitimately be zero, so we just check they're finite
                            assert np.all(np.isfinite(alt_pos_scores)), (
                                f"Non-finite {seq_type} alternative scores: {alt_pos_scores}"
                            )


def test_ism_snp_mutation_regions(static_model_dir):
    """Test ISM SNP with custom mutation regions."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    targets_file = f"{test_dir}/data/targets_sc3_me.txt"
    vcf_file = f"{test_dir}/data/sc3_snps.vcf"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_ism_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-u",
            "5",  # 5 upstream
            "-d",
            "10",  # 10 downstream
            "--stats",
            "logSUM",
            params_file,
            model_file,
            vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"hound_ism_snp with custom regions failed: {result.stderr}"
        )

        # Check output file exists
        scores_file = f"{temp_dir}/scores.h5"
        assert os.path.exists(scores_file)

        # Verify mutation length is correctly set
        with h5py.File(scores_file, "r") as h5f:
            expected_mut_len = 15  # 5 + 10
            assert (
                h5f["ref"]["seqs"].shape[2] == expected_mut_len
            )  # shape is [seq, channel, pos]
            assert h5f["ref"]["cov/logSUM"].shape[1] == expected_mut_len
            assert (
                h5f["alt"]["seqs"].shape[2] == expected_mut_len
            )  # shape is [seq, channel, pos]
            assert h5f["alt"]["cov/logSUM"].shape[1] == expected_mut_len


@pytest.mark.parametrize("stat", ["cov/logSUM", "covgene/logSUM"])
def test_ism_snp_gene_track_selection(static_model_dir, stat):
    """Only covgene/ scores require gene-track metadata and select gene tracks."""
    test_dir = str(pathlib.Path(__file__).parent)
    params_file = f"{test_dir}/data/params_sc3.json"
    model_file = f"{static_model_dir}/sc3_model.pth"
    fasta_file = f"{test_dir}/data/sc3.fa.gz"
    gtf_file = f"{test_dir}/data/sc3_genes.gtf"
    vcf_file = f"{test_dir}/data/sc3_snps.vcf"

    with tempfile.TemporaryDirectory() as temp_dir:
        # mark the second track as non-gene
        targets_df = pd.read_csv(
            f"{test_dir}/data/targets_sc3_me.txt", sep="\t", index_col=0
        )
        targets_df.loc[1, "strand_pair"] = 1
        targets_df.loc[1, "gene"] = 0
        if stat.startswith("covgene/"):
            expected_targets = ["H3K4ME3_S0"]
        else:
            targets_df = targets_df.drop(columns="gene")
            expected_targets = list(targets_df.identifier)
        targets_file = f"{temp_dir}/targets.txt"
        targets_df.to_csv(targets_file, sep="\t")

        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_ism_snp",
            "-f",
            fasta_file,
            "-t",
            targets_file,
            "-o",
            temp_dir,
            "-g",
            gtf_file,
            "-l",
            "10",
            "--stats",
            stat,
            params_file,
            model_file,
            vcf_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Failed: {result.stderr}"

        with h5py.File(f"{temp_dir}/scores.h5", "r") as h5f:
            assert h5f[f"ref/{stat}"].shape[-1] == len(expected_targets)
            assert h5f[f"alt/{stat}"].shape[-1] == len(expected_targets)
        for name in ["targets_cov.txt", "targets_covgene.txt"]:
            if name == "targets_covgene.txt" and not stat.startswith("covgene/"):
                assert not os.path.exists(f"{temp_dir}/{name}")
                continue
            out_df = pd.read_csv(f"{temp_dir}/{name}", sep="\t", index_col=0)
            assert list(out_df.identifier) == expected_targets


def test_ism_snp_bfloat16(static_model_dir):
    """-m bfloat16 runs (predictions cast to float32 for scoring) and warns."""
    test_dir = str(pathlib.Path(__file__).parent)
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python",
            "-m",
            "baskerville.scripts.hound_ism_snp",
            "-f",
            f"{test_dir}/data/sc3.fa.gz",
            "-t",
            f"{test_dir}/data/targets_sc3_me.txt",
            "-o",
            temp_dir,
            "-l",
            "4",
            "--stats",
            "logSUM",
            "-m",
            "bfloat16",
            f"{test_dir}/data/params_sc3.json",
            f"{static_model_dir}/sc3_model.pth",
            f"{test_dir}/data/sc3_snps.vcf",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"hound_ism_snp failed: {result.stderr}"
        assert "WARNING: --mix_dtype bfloat16" in result.stderr
