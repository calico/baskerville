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

"""
In Silico Mutagenesis (ISM) functionality.

Supports three score categories:
- Coverage stats (e.g., logSUM, logD2): full-sequence coverage scores
- Covgene stats (e.g., covgene/logD2): coverage sliced to gene exon bins
- Gene stats (e.g., gene/logFC): gene head predictions
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from baskerville import dataset
from baskerville import snps


@dataclass
class ISMResult:
    """Results from ISM analysis with optional gene scores.

    Attributes:
        cov: Coverage stat scores {stat: array(mut_len, 4, cov_targets)}.
        covgene: Per-gene coverage scores [{stat: array(mut_len, 4, gene_targets)}, ...].
        gene: Per-gene head scores [{stat: array(mut_len, 4, gene_depth)}, ...].
    """

    cov: dict = field(default_factory=dict)
    covgene: list = field(default_factory=list)
    gene: list = field(default_factory=list)


class ISM:
    """In Silico Mutagenesis (ISM) analyzer.

    Args:
        seqnn_model: Neural network model for sequence prediction.
        targets_df: DataFrame with target information.
        cov_stats: Coverage stats to compute (e.g., ['logSUM']).
        covgene_stats: Gene-sliced coverage stats (e.g., ['logD2']).
        gene_stats: Gene head stats (e.g., ['logFC']).
        strand_transform: Strand transformation matrix (optional).
        head: Model head to use for prediction (default: 0).
    """

    def __init__(
        self,
        seqnn_model,
        targets_df,
        cov_stats=None,
        covgene_stats=None,
        gene_stats=None,
        strand_transform=None,
        head=0,
    ):
        self.seqnn_model = seqnn_model
        self.targets_df = targets_df
        self.strand_transform = strand_transform
        self.head = head
        self.cov_stats = cov_stats or []
        self.covgene_stats = covgene_stats or []
        self.gene_stats = gene_stats or []

    @property
    def has_genes(self):
        return bool(self.covgene_stats or self.gene_stats)

    def compute(
        self,
        seq_1hot,
        mut_start,
        mut_end,
        gene_out_mask=None,
        gene_presence=None,
        plus_mask=None,
        minus_mask=None,
        gene_strands=None,
    ):
        """Compute ISM scores for a sequence.

        Args:
            seq_1hot: One-hot encoded sequence [channels, length].
            mut_start: Start position for mutagenesis.
            mut_end: End position for mutagenesis.
            gene_out_mask: Gene bin mask (1, num_genes, output_length) for gene head.
            gene_presence: Gene presence mask (1, num_genes).
            plus_mask: Boolean mask for plus-strand targets.
            minus_mask: Boolean mask for minus-strand targets.
            gene_strands: List of gene strand characters ('+' or '-').

        Returns:
            ISMResult with cov, covgene, and gene scores.
        """
        mut_len = mut_end - mut_start
        score_genes = gene_out_mask is not None and self.has_genes
        num_genes = gene_out_mask.shape[1] if score_genes else 0

        # determine output dimensions
        if self.strand_transform is not None:
            cov_targets = self.strand_transform.shape[1]
        else:
            cov_targets = len(self.targets_df)

        # initialize cov results
        cov_results = {}
        for stat in self.cov_stats:
            cov_results[stat] = np.zeros((mut_len, 4, cov_targets), dtype=np.float16)

        # gene-track width for covgene/ results
        covgene_targets = int(plus_mask.sum()) if plus_mask is not None else cov_targets

        # initialize gene results
        covgene_results = []
        gene_results = []
        if score_genes:
            for gi in range(num_genes):
                covgene_results.append(
                    {
                        stat: np.zeros((mut_len, 4, covgene_targets), dtype=np.float16)
                        for stat in self.covgene_stats
                    }
                )
                gene_results.append(
                    {
                        stat: np.zeros(
                            (mut_len, 4, self._gene_depth(gene_out_mask)),
                            dtype=np.float16,
                        )
                        for stat in self.gene_stats
                    }
                )

        # 1-hot encode reference and move to GPU
        ref_1hot = torch.tensor(
            np.expand_dims(seq_1hot, axis=0),
            device=self.seqnn_model.device,
            dtype=torch.float32,
        )

        # predict reference
        model_gom = gene_out_mask if score_genes else None
        model_gp = gene_presence if score_genes else None
        ref_output = self.seqnn_model(
            ref_1hot, self.head, gene_out_mask=model_gom, gene_presence=model_gp
        )
        ref_cov = self.seqnn_model.untransform_fn(
            ref_output.coverage.squeeze(0), self.targets_df
        )
        ref_gene = (
            ref_output.gene.squeeze(0) if score_genes and ref_output.has_gene else None
        )

        # for mutation positions
        for mi in tqdm(range(mut_start, mut_end), desc="ISM"):
            for ni in range(4):
                if ref_1hot[0, ni, mi] == 0:
                    # clone and mutate
                    alt_1hot = ref_1hot.clone()
                    alt_1hot[0, :, mi] = 0
                    alt_1hot[0, ni, mi] = 1

                    # predict alternate
                    alt_output = self.seqnn_model(
                        alt_1hot,
                        self.head,
                        gene_out_mask=model_gom,
                        gene_presence=model_gp,
                    )
                    alt_cov = self.seqnn_model.untransform_fn(
                        alt_output.coverage.squeeze(0), self.targets_df
                    )

                    # coverage stats
                    if self.cov_stats:
                        scores = snps.compute_scores_cov(
                            ref_cov.unsqueeze(0),
                            alt_cov.unsqueeze(0),
                            self.cov_stats,
                            self.strand_transform,
                        )
                        for stat in self.cov_stats:
                            cov_results[stat][mi - mut_start, ni] = scores[stat]

                    # gene-level scores
                    if score_genes:
                        alt_gene = (
                            alt_output.gene.squeeze(0) if alt_output.has_gene else None
                        )
                        self._score_genes(
                            ref_cov,
                            alt_cov,
                            ref_gene,
                            alt_gene,
                            gene_out_mask,
                            plus_mask,
                            minus_mask,
                            gene_strands,
                            covgene_results,
                            gene_results,
                            mi - mut_start,
                            ni,
                        )

        return ISMResult(cov=cov_results, covgene=covgene_results, gene=gene_results)

    def _score_genes(
        self,
        ref_cov,
        alt_cov,
        ref_gene,
        alt_gene,
        gene_out_mask,
        plus_mask,
        minus_mask,
        gene_strands,
        covgene_results,
        gene_results,
        pos_idx,
        nuc_idx,
    ):
        """Compute covgene and gene scores for all genes at one mutation."""
        num_genes = gene_out_mask.shape[1]

        for gi in range(num_genes):
            gene_mask = gene_out_mask[0, gi]
            if not gene_mask.any():
                continue

            # covgene: slice coverage to gene bins and filter by strand
            if self.covgene_stats:
                rp = ref_cov.unsqueeze(0)[:, :, gene_mask]
                ap = alt_cov.unsqueeze(0)[:, :, gene_mask]

                if gene_strands is not None and plus_mask is not None:
                    if gene_strands[gi] == "+":
                        rp = rp[:, plus_mask, :]
                        ap = ap[:, plus_mask, :]
                    else:
                        rp = rp[:, minus_mask, :]
                        ap = ap[:, minus_mask, :]

                scores = snps.compute_scores_cov(rp, ap, self.covgene_stats)
                for stat in self.covgene_stats:
                    covgene_results[gi][stat][pos_idx, nuc_idx] = scores[stat]

            # gene head scores
            if self.gene_stats and ref_gene is not None and alt_gene is not None:
                scores = snps.compute_scores_gene(
                    ref_gene[:, gi].unsqueeze(0),
                    alt_gene[:, gi].unsqueeze(0),
                    self.gene_stats,
                )
                for stat in self.gene_stats:
                    gene_results[gi][stat][pos_idx, nuc_idx] = scores[stat]

    def _gene_depth(self, gene_out_mask):
        """Infer gene head depth from the model."""
        return self.seqnn_model.output_depth(self.head, head_type="gene")
