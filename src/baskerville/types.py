"""Structured data types for coverage and gene prediction."""

from dataclasses import dataclass, fields, replace
from typing import Optional

import torch


@dataclass
class ModelOutput:
    """Structured output from SeqNN model forward pass.

    Attributes:
        coverage: Coverage predictions with shape (B, num_targets, target_length).
            None if model has no coverage head.
        gene: Gene expression predictions with shape (B, num_gene_targets, max_genes).
            None if model has no gene head.
    """

    coverage: Optional[torch.Tensor] = None
    gene: Optional[torch.Tensor] = None

    @property
    def has_coverage(self) -> bool:
        """Check if coverage predictions are present."""
        return self.coverage is not None

    @property
    def has_gene(self) -> bool:
        """Check if gene predictions are present."""
        return self.gene is not None


@dataclass
class BatchData:
    """Structured batch data from dataset.

    Attributes:
        sequence: One-hot encoded sequence with shape (4, seq_length) for single
            examples or (B, 4, seq_length) for batched data.
        coverage_targets: Coverage targets with shape (num_targets, target_length)
            or (B, num_targets, target_length). None if no coverage data.
        gene_targets: Gene expression targets with shape (num_gene_targets, max_genes)
            or (B, num_gene_targets, max_genes). None if no gene data.
        gene_presence: Boolean mask indicating valid genes with shape (max_genes,)
            or (B, max_genes). None if no gene data.
        gene_out_mask: Boolean mask of bins overlapping gene exons with shape
            (max_genes, target_bins) or (B, max_genes, target_bins). None if no gene data.
        species_label: MLM-only. One-hot species id with shape (1, num_species) or
            (B, 1, num_species). Selects the trunk normalization index (di). None
            outside MLM.
        exon_mask: MLM-only. Per-position exon mask with shape (seq_length,) or
            (B, seq_length). None if not loaded.
        repeat_mask: MLM-only. Per-position repeat mask, same shape as exon_mask.
            None if not loaded.
    """

    sequence: torch.Tensor
    coverage_targets: Optional[torch.Tensor] = None
    gene_targets: Optional[torch.Tensor] = None
    gene_presence: Optional[torch.Tensor] = None
    gene_out_mask: Optional[torch.Tensor] = None
    species_label: Optional[torch.Tensor] = None
    exon_mask: Optional[torch.Tensor] = None
    repeat_mask: Optional[torch.Tensor] = None

    @property
    def has_coverage(self) -> bool:
        """Check if coverage targets are present."""
        return self.coverage_targets is not None

    @property
    def has_gene(self) -> bool:
        """Check if gene data is present."""
        return self.gene_targets is not None

    def pin_memory(self) -> "BatchData":
        """Page-lock every tensor; DataLoader(pin_memory=True) calls this per batch."""
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).pin_memory()
                for f in fields(self)
                if getattr(self, f.name) is not None
            },
        )

    @staticmethod
    def collate(batch: list["BatchData"]) -> "BatchData":
        """Collate a list of BatchData objects into a batched BatchData.

        Args:
            batch: List of BatchData objects from dataset __getitem__.

        Returns:
            BatchData with all tensors stacked along the batch dimension.
        """
        return BatchData(
            sequence=torch.stack([b.sequence for b in batch]),
            coverage_targets=(
                torch.stack([b.coverage_targets for b in batch])
                if batch[0].has_coverage
                else None
            ),
            gene_targets=(
                torch.stack([b.gene_targets for b in batch])
                if batch[0].has_gene
                else None
            ),
            gene_presence=(
                torch.stack([b.gene_presence for b in batch])
                if batch[0].has_gene
                else None
            ),
            gene_out_mask=(
                torch.stack([b.gene_out_mask for b in batch])
                if batch[0].has_gene
                else None
            ),
            species_label=(
                torch.stack([b.species_label for b in batch])
                if batch[0].species_label is not None
                else None
            ),
            exon_mask=(
                torch.stack([b.exon_mask for b in batch])
                if batch[0].exon_mask is not None
                else None
            ),
            repeat_mask=(
                torch.stack([b.repeat_mask for b in batch])
                if batch[0].repeat_mask is not None
                else None
            ),
        )
