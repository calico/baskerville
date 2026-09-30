"""
Test hound_data with --gtf and --targets_gene options.

This tests the gene expression data generation pipeline.
"""

import json
import os
import numpy as np
import pytest
import zarr


def test_data_gene(data_gene_dir):
    """Test that hound_data correctly generates gene expression data."""

    # Verify output structure
    assert os.path.exists(data_gene_dir), f"Data directory not created: {data_gene_dir}"

    # Check statistics.json has gene metadata
    stats_file = f"{data_gene_dir}/statistics.json"
    assert os.path.exists(stats_file), "statistics.json not found"

    with open(stats_file) as f:
        stats = json.load(f)

    assert "num_targets_gene" in stats, "num_targets_gene missing from statistics"
    assert "max_genes_per_seq" in stats, "max_genes_per_seq missing from statistics"
    assert stats["num_targets_gene"] == 2, (
        f"Expected 2 gene targets, got {stats['num_targets_gene']}"
    )

    print(f"Gene targets: {stats['num_targets_gene']}")
    print(f"Max genes per seq: {stats['max_genes_per_seq']}")

    # Check targets_gene.txt was copied
    targets_gene_file = f"{data_gene_dir}/targets_gene.txt"
    assert os.path.exists(targets_gene_file), "targets_gene.txt not found"

    # Check genes.txt was created
    genes_file = f"{data_gene_dir}/genes.txt"
    assert os.path.exists(genes_file), "genes.txt not found"

    # Check zarr files contain gene arrays
    examples_dir = f"{data_gene_dir}/examples"
    assert os.path.exists(examples_dir), "examples directory not found"

    # Check one of the train zarr files
    train_zarrs = [f for f in os.listdir(examples_dir) if f.startswith("train")]
    assert len(train_zarrs) > 0, "No train zarr files found"

    train_zarr_path = f"{examples_dir}/{train_zarrs[0]}"
    zarr_root = zarr.open(train_zarr_path, mode="r")

    # Verify gene arrays exist
    assert "gene_target" in zarr_root, "gene_target array missing from zarr"
    assert "gene_presence" in zarr_root, "gene_presence array missing from zarr"
    assert "gene_out_mask" in zarr_root, "gene_out_mask array missing from zarr"

    gene_target = zarr_root["gene_target"]
    gene_presence = zarr_root["gene_presence"]
    gene_out_mask = zarr_root["gene_out_mask"]

    print(f"gene_target shape: {gene_target.shape}")
    print(f"gene_presence shape: {gene_presence.shape}")
    print(f"gene_out_mask shape: {gene_out_mask.shape}")

    # Verify shapes
    num_seqs = gene_target.shape[0]
    num_gene_targets = gene_target.shape[1]
    max_genes = gene_target.shape[2]

    assert num_gene_targets == stats["num_targets_gene"], (
        "Gene target dimension mismatch"
    )
    assert max_genes == stats["max_genes_per_seq"], "Max genes dimension mismatch"
    assert gene_presence.shape == (
        num_seqs,
        max_genes,
    ), f"gene_presence shape mismatch: {gene_presence.shape}"
    assert gene_out_mask.shape[0] == num_seqs, (
        f"gene_out_mask num_seqs mismatch: {gene_out_mask.shape}"
    )
    assert gene_out_mask.shape[1] == max_genes, (
        f"gene_out_mask max_genes mismatch: {gene_out_mask.shape}"
    )

    # Check that some genes have valid data
    mask_sample = np.array(gene_presence[0])
    assert mask_sample.sum() > 0, "No valid genes in first sequence"

    print(f"First sequence has {mask_sample.sum()} valid genes")

    # Check gene_target has reasonable values (log-normal distributed)
    target_sample = np.array(gene_target[0])
    valid_values = target_sample[:, mask_sample]
    assert valid_values.max() > 0, "Gene expression values should be positive"

    print(f"Gene expression range: {valid_values.min():.2f} - {valid_values.max():.2f}")

    # Check gene_out_mask has valid entries for valid genes
    bin_mask_sample = np.array(gene_out_mask[0])
    for gi in range(max_genes):
        if mask_sample[gi]:
            assert bin_mask_sample[gi].any(), f"Gene {gi} has empty bin mask"

    print("All gene data checks passed!")


def test_gene_mapping_zarr(data_gene_dir):
    """Test that gene mapping zarr store was created correctly."""
    gene_mapping_path = f"{data_gene_dir}/seqs_gene/gene_mapping.zarr"
    assert os.path.exists(gene_mapping_path), "gene_mapping.zarr not found"

    gene_mapping = zarr.open(gene_mapping_path, mode="r")
    assert "gene_ids" in gene_mapping, "gene_ids array missing from gene_mapping.zarr"
    assert "gene_presence" in gene_mapping, (
        "gene_presence array missing from gene_mapping.zarr"
    )
    assert "gene_out_mask" in gene_mapping, (
        "gene_out_mask array missing from gene_mapping.zarr"
    )

    # Verify shapes match
    num_seqs = gene_mapping["gene_ids"].shape[0]
    max_genes = gene_mapping["gene_ids"].shape[1]

    assert gene_mapping["gene_presence"].shape == (
        num_seqs,
        max_genes,
    ), f"gene_presence shape mismatch: {gene_mapping['gene_presence'].shape}"
    assert gene_mapping["gene_out_mask"].shape[0] == num_seqs, (
        f"gene_out_mask num_seqs mismatch"
    )
    assert gene_mapping["gene_out_mask"].shape[1] == max_genes, (
        f"gene_out_mask max_genes mismatch"
    )

    # Verify gene_ids are strings
    sample_ids = gene_mapping["gene_ids"][0]
    assert sample_ids.dtype.kind == "U", (
        f"gene_ids should be Unicode strings, got {sample_ids.dtype}"
    )

    print(f"gene_mapping.zarr verified: {num_seqs} sequences, {max_genes} max genes")


def test_gene_roundtrip(data_gene_folds_dir):
    """Test that gene expression values can be reconstructed from zarr output.

    This verifies that regardless of internal gene ordering, the gene_id to
    expression mapping is preserved correctly.
    """
    import pandas as pd

    # Read original expression file
    targets_gene_file = f"{data_gene_folds_dir}/targets_gene.txt"
    targets_gene_df = pd.read_csv(targets_gene_file, index_col=0, sep="\t")

    # Read first target's expression file
    expr_file = targets_gene_df.iloc[0].file
    original_expr = pd.read_csv(expr_file, sep="\t", index_col=0)
    if original_expr.shape[1] >= 1:
        original_expr = original_expr.iloc[:, 0]

    # Read gene mapping (contains all sequences)
    gene_mapping_path = f"{data_gene_folds_dir}/seqs_gene/gene_mapping.zarr"
    gene_mapping = zarr.open(gene_mapping_path, mode="r")
    gene_ids_arr = gene_mapping["gene_ids"]
    gene_presence_arr = gene_mapping["gene_presence"]

    # Read fold sizes from statistics.json
    with open(f"{data_gene_folds_dir}/statistics.json") as f:
        stats = json.load(f)

    # Get fold sizes in order
    fold_sizes = []
    fold_idx = 0
    while f"fold{fold_idx}_seqs" in stats:
        fold_sizes.append(stats[f"fold{fold_idx}_seqs"])
        fold_idx += 1

    # Compute cumulative offsets for each fold
    fold_offsets = [0]
    for size in fold_sizes[:-1]:
        fold_offsets.append(fold_offsets[-1] + size)

    # Test each fold
    examples_dir = f"{data_gene_folds_dir}/examples"
    mismatches = 0
    total_genes = 0

    for fold_num in range(len(fold_sizes)):
        fold_zarr_path = f"{examples_dir}/fold{fold_num}.zarr"
        if not os.path.exists(fold_zarr_path):
            continue

        fold_zarr = zarr.open(fold_zarr_path, mode="r")
        if "gene_target" not in fold_zarr:
            continue

        gene_target = fold_zarr["gene_target"]
        num_fold_seqs = gene_target.shape[0]
        global_offset = fold_offsets[fold_num]

        for local_idx in range(num_fold_seqs):
            global_idx = global_offset + local_idx

            gene_ids = gene_ids_arr[global_idx]
            gene_presence = gene_presence_arr[global_idx]
            # gene_target shape: (num_seqs, num_targets, max_genes)
            seq_expr = np.array(gene_target[local_idx, 0, :])

            for gi in range(len(gene_presence)):
                if not gene_presence[gi]:
                    continue

                gene_id = str(gene_ids[gi])
                if gene_id == "" or gene_id not in original_expr.index:
                    continue

                total_genes += 1
                original_val = original_expr.loc[gene_id]
                reconstructed_val = seq_expr[gi]

                # Allow small floating point differences (float16 storage)
                if not np.isclose(
                    original_val, reconstructed_val, rtol=1e-2, atol=1e-2
                ):
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            f"Mismatch in fold{fold_num}: gene={gene_id}, "
                            f"original={original_val:.4f}, reconstructed={reconstructed_val:.4f}"
                        )

    match_rate = (total_genes - mismatches) / total_genes if total_genes > 0 else 0
    print(
        f"Gene expression roundtrip: {total_genes} genes checked, {mismatches} mismatches"
    )
    print(f"Match rate: {match_rate:.2%}")

    assert total_genes > 0, "No genes were checked - test setup issue"
    assert mismatches == 0, (
        f"Found {mismatches} expression value mismatches out of {total_genes} genes"
    )
