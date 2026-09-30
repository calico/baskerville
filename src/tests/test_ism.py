#!/usr/bin/env python
# Copyright 2023 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import json
import os
import pathlib
import pytest
import torch
import numpy as np
import pandas as pd

from baskerville import ism
from baskerville import seqnn
from baskerville import dataset


@pytest.fixture
def params():
    """Load parameters from params_sc3.json file."""
    params_file = os.path.join(os.path.dirname(__file__), "data", "params_sc3.json")
    with open(params_file) as f:
        params = json.load(f)
    return params["model"]


@pytest.fixture
def targets_df():
    """Load targets DataFrame from test data."""
    targets_file = os.path.join(os.path.dirname(__file__), "data", "targets_sc3_me.txt")
    return pd.read_csv(targets_file, sep="\t", index_col=0)


@pytest.fixture
def seqnn_model(params, static_model_dir):
    """Create a SeqNN model with real trained weights."""
    # Create model with sc3 parameters
    model = seqnn.SeqNN(params)

    # Restore trained weights using the model's restore method
    model_file = os.path.join(static_model_dir, "sc3_model.pth")
    model.restore(model_file)

    return model


@pytest.fixture
def sample_sequence(params):
    """Create a sample one-hot encoded sequence."""
    seq_length = params["seq_length"]
    seq_1hot = np.zeros((4, seq_length))

    # Create a realistic sequence pattern
    np.random.seed(42)  # For reproducibility
    for i in range(seq_length):
        nucleotide = np.random.randint(0, 4)
        seq_1hot[nucleotide, i] = 1

    return seq_1hot


class TestISMClass:
    """Test suite for the ISM class."""

    def test_ism_class_basic(self, seqnn_model, targets_df, sample_sequence):
        """Test basic functionality of ISM class."""
        # Test parameters - use small region for faster testing
        mut_start = 8000  # Middle of sequence
        mut_end = 8010  # 10 positions
        snp_stats = ["logSUM"]
        strand_transform = None

        # Create ISM analyzer
        ism_analyzer = ism.ISM(
            seqnn_model=seqnn_model,
            targets_df=targets_df,
            cov_stats=snp_stats,
            strand_transform=strand_transform,
            head=0,
        )

        # Run analysis
        result = ism_analyzer.compute(sample_sequence, mut_start, mut_end)

        # Assertions
        assert isinstance(result, ism.ISMResult)
        assert "logSUM" in result.cov
        expected_shape = (mut_end - mut_start, 4, len(targets_df))
        assert result.cov["logSUM"].shape == expected_shape

    def test_ism_class_multiple_stats(self, seqnn_model, targets_df, sample_sequence):
        """Test ISM class with multiple statistics."""
        # Test parameters
        mut_start = 8000
        mut_end = 8005  # Even smaller for multiple stats
        snp_stats = ["logSUM", "logD2"]
        strand_transform = None

        # Create ISM analyzer
        ism_analyzer = ism.ISM(
            seqnn_model=seqnn_model,
            targets_df=targets_df,
            cov_stats=snp_stats,
            strand_transform=strand_transform,
            head=0,
        )

        # Run analysis
        result = ism_analyzer.compute(sample_sequence, mut_start, mut_end)

        # Assertions
        assert len(result.cov) == 2
        for stat in snp_stats:
            assert stat in result.cov
            expected_shape = (mut_end - mut_start, 4, len(targets_df))
            assert result.cov[stat].shape == expected_shape

    def test_ism_class_reuse(self, seqnn_model, targets_df, sample_sequence):
        """Test that ISM class can be reused for multiple sequences."""
        snp_stats = ["logSUM"]
        strand_transform = None

        # Create ISM analyzer once
        ism_analyzer = ism.ISM(
            seqnn_model=seqnn_model,
            targets_df=targets_df,
            cov_stats=snp_stats,
            strand_transform=strand_transform,
            head=0,
        )

        # Test on different regions with the same analyzer
        result1 = ism_analyzer.compute(sample_sequence, 8000, 8003)
        result2 = ism_analyzer.compute(sample_sequence, 9000, 9003)

        # Both should work and have correct shapes
        expected_shape = (3, 4, len(targets_df))
        assert result1.cov["logSUM"].shape == expected_shape
        assert result2.cov["logSUM"].shape == expected_shape

        # Results should be different (different regions)
        assert not np.allclose(result1.cov["logSUM"], result2.cov["logSUM"]), (
            "Different regions should produce different ISM scores"
        )

    def test_ism_class_configuration_stored(
        self, seqnn_model, targets_df, sample_sequence
    ):
        """Test that ISM class properly stores its configuration."""
        snp_stats = ["logSUM", "logD2"]
        strand_transform = None
        head = 0

        # Create ISM analyzer
        ism_analyzer = ism.ISM(
            seqnn_model=seqnn_model,
            targets_df=targets_df,
            cov_stats=snp_stats,
            strand_transform=strand_transform,
            head=head,
        )

        # Check that configuration is stored correctly
        assert ism_analyzer.seqnn_model is seqnn_model
        assert ism_analyzer.targets_df is targets_df
        assert ism_analyzer.cov_stats == snp_stats
        assert ism_analyzer.strand_transform is strand_transform
        assert ism_analyzer.head == head

    def test_ism_class_covgene(self, seqnn_model, targets_df, sample_sequence):
        """Test ISM class covgene/logFC scoring with a synthetic gene mask."""
        mut_start = 8000
        mut_end = 8003
        mut_len = mut_end - mut_start
        num_targets = len(targets_df)

        # Create ISM analyzer with both cov and covgene stats
        ism_analyzer = ism.ISM(
            seqnn_model=seqnn_model,
            targets_df=targets_df,
            cov_stats=["logSUM"],
            covgene_stats=["logFC"],
            head=0,
        )

        # Construct synthetic gene_out_mask: 2 genes, each covering a
        # contiguous block of bins in the prediction region.
        output_length = seqnn_model.output_length()
        num_genes = 2
        device = seqnn_model.device
        gene_out_mask = torch.zeros(
            (1, num_genes, output_length), dtype=torch.bool, device=device
        )
        gene_out_mask[0, 0, 100:120] = True
        gene_out_mask[0, 1, 300:340] = True
        gene_presence = torch.ones((1, num_genes), dtype=torch.bool, device=device)

        # gene-track mask selecting a single track (unstranded)
        gene_mask = torch.zeros(num_targets, dtype=torch.bool, device=device)
        gene_mask[0] = True

        result = ism_analyzer.compute(
            sample_sequence,
            mut_start,
            mut_end,
            gene_out_mask=gene_out_mask,
            gene_presence=gene_presence,
            plus_mask=gene_mask,
            minus_mask=gene_mask,
            gene_strands=["+", "+"],
        )

        # cov stat present with expected shape
        assert isinstance(result, ism.ISMResult)
        assert result.cov["logSUM"].shape == (mut_len, 4, num_targets)

        # covgene results: one dict per gene, each with logFC sliced to the
        # gene-track mask width (1 of the 2 targets)
        assert len(result.covgene) == num_genes
        for gi in range(num_genes):
            assert "logFC" in result.covgene[gi]
            assert result.covgene[gi]["logFC"].shape == (mut_len, 4, 1)

        # no gene_stats requested — gene results should be empty per gene
        assert len(result.gene) == num_genes
        for gi in range(num_genes):
            assert result.gene[gi] == {}

        # covgene scores shouldn't be uniformly zero (mutations should affect
        # at least one position/nucleotide/target)
        assert np.any(result.covgene[0]["logFC"] != 0)
        assert np.any(result.covgene[1]["logFC"] != 0)
