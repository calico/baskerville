#!/usr/bin/env python
"""
Quality control analysis for baskerville datasets.

Computes per-target statistics and validates gene data to identify issues
that can cause training anomalies (NaN loss, zero-variance targets, etc.).

Usage:
    hound_data_qc /path/to/data                    # Basic QC
    hound_data_qc /path/to/data --json -v          # Full report with JSON output
    hound_data_qc /path/to/data --split train      # Check specific split only
"""

import argparse
import gc
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

from natsort import natsorted
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr


def main():
    parser = argparse.ArgumentParser(
        description="Quality control analysis for dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hound_data_qc /path/to/data                # Basic QC, outputs qc.tsv
  hound_data_qc /path/to/data --json -v      # Full report with JSON
  hound_data_qc /path/to/data --split train  # Check train split only
        """,
    )
    parser.add_argument("data_dir", help="Dataset directory containing examples/*.zarr")
    parser.add_argument(
        "-b",
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for processing [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="*",
        help="Split to process (train, valid, test, or * for all) [Default: %(default)s]",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output qc_report.json for dashboard",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed diagnostics",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=None,
        help="Output directory [Default: data_dir]",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir

    # Read data parameters
    data_stats_file = f"{args.data_dir}/statistics.json"
    with open(data_stats_file) as f:
        data_stats = json.load(f)

    # Read targets table for group information
    targets_file = Path(args.data_dir) / "targets.txt"
    targets_df = None
    if targets_file.exists():
        targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    # Find zarr files
    if args.split == "*":
        fold_zarr_files = natsorted(glob.glob(f"{args.data_dir}/examples/*.zarr"))
    else:
        fold_zarr_files = natsorted(
            glob.glob(f"{args.data_dir}/examples/{args.split}*.zarr")
        )

    if not fold_zarr_files:
        print(f"ERROR: No zarr files found matching split '{args.split}'")
        sys.exit(1)

    # Check for gene data
    sample_zarr = zarr.open(fold_zarr_files[0], mode="r")
    has_genes = "gene_presence" in sample_zarr
    has_targets = "target" in sample_zarr
    has_gene_targets = "gene_target" in sample_zarr

    # Validate shapes
    shape_issues = validate_shapes(sample_zarr, data_stats)

    # Compute fold stats (separate lists for coverage and gene targets)
    coverage_dicts = []
    gene_target_dicts = []
    gene_issues = {"no_valid_genes": [], "empty_bin_masks": []}
    total_seqs = 0

    for fi, fold_zarr_file in enumerate(fold_zarr_files):
        fold_name = Path(fold_zarr_file).stem
        print(f"Processing {fold_name}...")

        zarr_open = zarr.open(fold_zarr_file, mode="r")

        # Coverage targets
        if has_targets:
            fold_stats = zarr_stats(zarr_open, batch_size=args.batch_size)
            num_targets = fold_stats["mean"].shape[0]

            for ti in range(num_targets):
                fold_task_dict = {
                    "fold": fold_name,
                    "target": ti,
                    "mean": fold_stats["mean"][ti],
                    "variance": fold_stats["variance"][ti],
                    "min": fold_stats["min"][ti],
                    "max": fold_stats["max"][ti],
                    "pct_zeros": fold_stats["pct_zeros"][ti] * 100,
                    "pct_nan": fold_stats["pct_nan"][ti] * 100,
                    "pct_inf": fold_stats["pct_inf"][ti] * 100,
                }
                # Add group if available
                if targets_df is not None and "group" in targets_df.columns:
                    fold_task_dict["group"] = targets_df.iloc[ti]["group"]
                coverage_dicts.append(fold_task_dict)

        # Gene expression targets
        if has_gene_targets:
            fold_stats = zarr_gene_stats(zarr_open, batch_size=args.batch_size)
            num_gene_targets = fold_stats["mean"].shape[0]

            for ti in range(num_gene_targets):
                fold_task_dict = {
                    "fold": fold_name,
                    "target": ti,
                    "mean": fold_stats["mean"][ti],
                    "variance": fold_stats["variance"][ti],
                    "min": fold_stats["min"][ti],
                    "max": fold_stats["max"][ti],
                    "pct_zeros": fold_stats["pct_zeros"][ti] * 100,
                    "pct_nan": fold_stats["pct_nan"][ti] * 100,
                    "pct_inf": fold_stats["pct_inf"][ti] * 100,
                }
                gene_target_dicts.append(fold_task_dict)

        # Gene mask/slice validation
        if has_genes:
            fold_gene_issues = validate_genes(
                zarr_open, data_stats.get("target_length", 0), seq_offset=total_seqs
            )
            gene_issues["no_valid_genes"].extend(fold_gene_issues["no_valid_genes"])
            gene_issues["empty_bin_masks"].extend(fold_gene_issues["empty_bin_masks"])

        # Count sequences
        if has_targets:
            total_seqs += zarr_open["target"].shape[0]
        elif has_genes:
            total_seqs += zarr_open["gene_presence"].shape[0]

        gc.collect()

    # Build results DataFrames
    coverage_df = None
    gene_target_df = None

    if coverage_dicts:
        coverage_df = pd.DataFrame(coverage_dicts)
        qc_file = f"{out_dir}/qc.tsv"
        coverage_df.to_csv(qc_file, sep="\t", index=False, float_format="%.6f")
        print(f"Wrote {qc_file}")

    if gene_target_dicts:
        gene_target_df = pd.DataFrame(gene_target_dicts)
        qc_gene_file = f"{out_dir}/qc_gene.tsv"
        gene_target_df.to_csv(qc_gene_file, sep="\t", index=False, float_format="%.6f")
        print(f"Wrote {qc_gene_file}")

    # Identify problematic targets (check both coverage and gene targets)
    problems = identify_problems(coverage_df, gene_issues, shape_issues)
    gene_target_problems = identify_problems(
        gene_target_df, {"no_valid_genes": [], "empty_bin_masks": []}, []
    )

    # Merge gene target problems into main problems dict with prefix
    problems["gene_zero_variance_targets"] = gene_target_problems[
        "zero_variance_targets"
    ]
    problems["gene_high_zero_targets"] = gene_target_problems["high_zero_targets"]
    problems["gene_nan_targets"] = gene_target_problems["nan_targets"]
    problems["gene_inf_targets"] = gene_target_problems["inf_targets"]

    # Print warnings
    exit_code = print_warnings(
        problems, verbose=args.verbose, has_gene_targets=has_gene_targets
    )

    # Write JSON report if requested
    if args.json:
        report = build_json_report(
            args.data_dir,
            args.split,
            data_stats,
            coverage_df,
            gene_target_df,
            gene_issues,
            shape_issues,
            problems,
            has_genes,
            has_gene_targets,
        )
        json_file = f"{out_dir}/qc_report.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {json_file}")

    sys.exit(exit_code)


def zarr_stats(zarr_open, batch_size=64):
    """
    Compute summary statistics for each target in a zarr dataset.

    Uses online algorithms to process the data in batches, making it memory-efficient
    for extremely large tensors.

    Parameters
    ----------
    zarr_open : zarr.Group
        Opened zarr group containing 'target' dataset.
    batch_size : int
        Number of sequences to process at once.

    Returns
    -------
    dict
        Statistics per target: mean, variance, min, max, pct_zeros, pct_nan, pct_inf
    """
    targets = zarr_open["target"]
    num_seqs, num_targets, target_length = targets.shape

    # Initialize variables for online algorithm
    count = 0
    task_means = np.zeros(num_targets, dtype=np.float64)
    task_m2 = np.zeros(num_targets, dtype=np.float64)
    task_min = np.full(num_targets, np.inf, dtype=np.float64)
    task_max = np.full(num_targets, -np.inf, dtype=np.float64)
    task_zeros = np.zeros(num_targets, dtype=np.int64)
    task_nans = np.zeros(num_targets, dtype=np.int64)
    task_infs = np.zeros(num_targets, dtype=np.int64)

    # Process in batches
    for start_idx in tqdm(range(0, num_seqs, batch_size), desc="Computing stats"):
        end_idx = min(start_idx + batch_size, num_seqs)

        # Load batch and convert to float64 for numerical stability
        current_batch = targets[start_idx:end_idx].astype(np.float64)

        # Reshape to (batch_size*L, T)
        current_batch = np.transpose(current_batch, (0, 2, 1))
        batch_flat = current_batch.reshape(-1, num_targets)
        batch_count = batch_flat.shape[0]

        # Count NaN and inf before replacing
        batch_nans = np.isnan(batch_flat).sum(axis=0)
        batch_infs = np.isinf(batch_flat).sum(axis=0)
        task_nans += batch_nans
        task_infs += batch_infs

        # Replace NaN/inf for statistics computation
        batch_clean = np.nan_to_num(batch_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Update min/max
        batch_min = np.min(batch_clean, axis=0)
        batch_max = np.max(batch_clean, axis=0)
        task_min = np.minimum(task_min, batch_min)
        task_max = np.maximum(task_max, batch_max)

        # Update zero count
        batch_zeros = np.sum(batch_clean == 0, axis=0)
        task_zeros += batch_zeros

        # Online mean/variance (Welford's algorithm)
        batch_mean = np.mean(batch_clean, axis=0)
        batch_M2 = np.sum((batch_clean - batch_mean[np.newaxis, :]) ** 2, axis=0)

        delta = batch_mean - task_means
        new_count = count + batch_count
        task_means = (count * task_means + batch_count * batch_mean) / new_count
        task_m2 = task_m2 + batch_M2 + delta**2 * count * batch_count / new_count
        count = new_count

    # Normalize
    total_values = num_seqs * target_length
    task_vars = task_m2 / count

    return {
        "mean": task_means.astype(np.float32),
        "variance": task_vars.astype(np.float32),
        "min": task_min.astype(np.float32),
        "max": task_max.astype(np.float32),
        "pct_zeros": (task_zeros / total_values).astype(np.float32),
        "pct_nan": (task_nans / total_values).astype(np.float32),
        "pct_inf": (task_infs / total_values).astype(np.float32),
    }


def zarr_gene_stats(zarr_open, batch_size=64):
    """
    Compute summary statistics for gene expression targets.

    For gene_target with shape (num_seqs, num_gene_targets, max_genes),
    computes statistics per gene target across all sequences and genes.

    Parameters
    ----------
    zarr_open : zarr.Group
        Opened zarr group containing 'gene_target' and 'gene_presence' datasets.
    batch_size : int
        Number of sequences to process at once.

    Returns
    -------
    dict
        Statistics per gene target: mean, variance, min, max, pct_zeros, pct_nan, pct_inf
    """
    gene_target = zarr_open["gene_target"]
    gene_presence = zarr_open["gene_presence"]
    num_seqs, num_targets, max_genes = gene_target.shape

    # Initialize variables for online algorithm
    count = np.zeros(num_targets, dtype=np.int64)
    task_means = np.zeros(num_targets, dtype=np.float64)
    task_m2 = np.zeros(num_targets, dtype=np.float64)
    task_min = np.full(num_targets, np.inf, dtype=np.float64)
    task_max = np.full(num_targets, -np.inf, dtype=np.float64)
    task_zeros = np.zeros(num_targets, dtype=np.int64)
    task_nans = np.zeros(num_targets, dtype=np.int64)
    task_infs = np.zeros(num_targets, dtype=np.int64)

    # Process in batches
    for start_idx in tqdm(range(0, num_seqs, batch_size), desc="Computing gene stats"):
        end_idx = min(start_idx + batch_size, num_seqs)

        # Load batch: (batch, num_targets, max_genes)
        batch_targets = gene_target[start_idx:end_idx].astype(np.float64)
        batch_mask = gene_presence[start_idx:end_idx]  # (batch, max_genes)

        # Process each target separately (masked by gene_presence)
        for ti in range(num_targets):
            # Get values for this target: (batch, max_genes)
            target_vals = batch_targets[:, ti, :]

            # Apply mask and flatten
            valid_vals = target_vals[batch_mask]

            if len(valid_vals) == 0:
                continue

            batch_count = len(valid_vals)

            # Count NaN and inf before replacing
            batch_nans = np.isnan(valid_vals).sum()
            batch_infs = np.isinf(valid_vals).sum()
            task_nans[ti] += batch_nans
            task_infs[ti] += batch_infs

            # Replace NaN/inf for statistics computation
            valid_clean = np.nan_to_num(valid_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # Update min/max
            task_min[ti] = min(task_min[ti], valid_clean.min())
            task_max[ti] = max(task_max[ti], valid_clean.max())

            # Update zero count
            task_zeros[ti] += (valid_clean == 0).sum()

            # Online mean/variance (Welford's algorithm)
            batch_mean = valid_clean.mean()
            batch_M2 = ((valid_clean - batch_mean) ** 2).sum()

            delta = batch_mean - task_means[ti]
            new_count = count[ti] + batch_count
            task_means[ti] = (
                count[ti] * task_means[ti] + batch_count * batch_mean
            ) / new_count
            task_m2[ti] = (
                task_m2[ti] + batch_M2 + delta**2 * count[ti] * batch_count / new_count
            )
            count[ti] = new_count

    # Normalize (handle zero counts)
    task_vars = np.where(count > 0, task_m2 / count, 0.0)
    pct_zeros = np.where(count > 0, task_zeros / count, 0.0)
    pct_nan = np.where(count > 0, task_nans / count, 0.0)
    pct_inf = np.where(count > 0, task_infs / count, 0.0)

    return {
        "mean": task_means.astype(np.float32),
        "variance": task_vars.astype(np.float32),
        "min": task_min.astype(np.float32),
        "max": task_max.astype(np.float32),
        "pct_zeros": pct_zeros.astype(np.float32),
        "pct_nan": pct_nan.astype(np.float32),
        "pct_inf": pct_inf.astype(np.float32),
    }


def validate_genes(zarr_open, target_length, seq_offset=0):
    """
    Validate gene masks and bin masks.

    Parameters
    ----------
    zarr_open : zarr.Group
        Opened zarr group containing gene_presence and gene_out_mask.
    target_length : int
        Expected target length for bounds checking.
    seq_offset : int
        Offset to add to sequence indices for multi-file datasets.

    Returns
    -------
    dict
        Issues found: no_valid_genes (list of seq indices),
        empty_bin_masks (list of (seq_idx, gene_idx))
    """
    gene_presence = zarr_open["gene_presence"]
    gene_out_mask = zarr_open["gene_out_mask"]
    num_seqs = gene_presence.shape[0]

    no_valid_genes = []
    empty_bin_masks = []

    for si in tqdm(range(num_seqs), desc="Validating genes"):
        mask = gene_presence[si]
        bin_mask = gene_out_mask[si]

        # Check for no valid genes
        if mask.sum() == 0:
            no_valid_genes.append(seq_offset + si)
            continue

        # Validate bin masks for valid genes
        for gi in range(len(mask)):
            if mask[gi]:
                if not bin_mask[gi].any():
                    empty_bin_masks.append((seq_offset + si, gi))

    return {
        "no_valid_genes": no_valid_genes,
        "empty_bin_masks": empty_bin_masks,
    }


def validate_shapes(zarr_open, data_stats):
    """
    Validate zarr shapes against statistics.json.

    Returns
    -------
    list
        List of shape mismatch descriptions.
    """
    issues = []

    if "target" in zarr_open:
        actual_targets = zarr_open["target"].shape[1]
        declared_targets = data_stats.get("num_targets", actual_targets)
        if actual_targets != declared_targets:
            issues.append(
                f"num_targets: declared {declared_targets}, actual {actual_targets}"
            )

        actual_length = zarr_open["target"].shape[2]
        declared_length = data_stats.get("target_length", actual_length)
        if actual_length != declared_length:
            issues.append(
                f"target_length: declared {declared_length}, actual {actual_length}"
            )

    return issues


def identify_problems(df, gene_issues, shape_issues):
    """
    Identify problematic targets and sequences.

    Returns
    -------
    dict
        Categorized problems.
    """
    problems = {
        "shape_mismatches": shape_issues,
        "zero_variance_targets": [],
        "high_zero_targets": [],
        "nan_targets": [],
        "inf_targets": [],
        "no_valid_genes": gene_issues["no_valid_genes"],
        "empty_bin_masks": gene_issues["empty_bin_masks"],
    }

    if df is None:
        return problems

    # Zero variance (can cause NaN correlation)
    zero_var = df[df["variance"] < 1e-8]
    if len(zero_var) > 0:
        problems["zero_variance_targets"] = zero_var["target"].unique().tolist()

    # High zeros (>99%)
    high_zeros = df[df["pct_zeros"] > 99]
    if len(high_zeros) > 0:
        problems["high_zero_targets"] = high_zeros["target"].unique().tolist()

    # NaN values
    has_nan = df[df["pct_nan"] > 0]
    if len(has_nan) > 0:
        problems["nan_targets"] = has_nan["target"].unique().tolist()

    # Inf values
    has_inf = df[df["pct_inf"] > 0]
    if len(has_inf) > 0:
        problems["inf_targets"] = has_inf["target"].unique().tolist()

    return problems


def print_warnings(problems, verbose=False, has_gene_targets=False):
    """
    Print warnings for identified problems.

    Returns
    -------
    int
        Exit code (0 if no critical issues, 1 if critical issues found).
    """
    has_critical = False

    # Shape mismatches (critical)
    if problems.get("shape_mismatches"):
        has_critical = True
        for issue in problems["shape_mismatches"]:
            print(f"ERROR: Shape mismatch - {issue}")

    # Coverage targets
    if problems.get("zero_variance_targets"):
        targets = problems["zero_variance_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"WARNING: {len(targets)} coverage targets have zero variance (indices: {preview}{suffix})"
        )
        if verbose:
            print(f"  All zero-variance targets: {targets}")

    if problems.get("high_zero_targets"):
        targets = problems["high_zero_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"WARNING: {len(targets)} coverage targets are >99% zeros (indices: {preview}{suffix})"
        )

    if problems.get("nan_targets"):
        has_critical = True
        targets = problems["nan_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"ERROR: {len(targets)} coverage targets contain NaN values (indices: {preview}{suffix})"
        )

    if problems.get("inf_targets"):
        has_critical = True
        targets = problems["inf_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"ERROR: {len(targets)} coverage targets contain inf values (indices: {preview}{suffix})"
        )

    # Gene expression targets
    if problems.get("gene_zero_variance_targets"):
        targets = problems["gene_zero_variance_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"WARNING: {len(targets)} gene targets have zero variance (indices: {preview}{suffix})"
        )
        if verbose:
            print(f"  All zero-variance gene targets: {targets}")

    if problems.get("gene_high_zero_targets"):
        targets = problems["gene_high_zero_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"WARNING: {len(targets)} gene targets are >99% zeros (indices: {preview}{suffix})"
        )

    if problems.get("gene_nan_targets"):
        has_critical = True
        targets = problems["gene_nan_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"ERROR: {len(targets)} gene targets contain NaN values (indices: {preview}{suffix})"
        )

    if problems.get("gene_inf_targets"):
        has_critical = True
        targets = problems["gene_inf_targets"]
        preview = targets[:5]
        suffix = f", ... ({len(targets)} total)" if len(targets) > 5 else ""
        print(
            f"ERROR: {len(targets)} gene targets contain inf values (indices: {preview}{suffix})"
        )

    # Gene mask/slice validation
    if problems.get("no_valid_genes"):
        seqs = problems["no_valid_genes"]
        print(f"WARNING: {len(seqs)} sequences have no valid genes")
        if verbose and len(seqs) <= 20:
            print(f"  Sequence indices: {seqs}")

    if problems.get("empty_bin_masks"):
        has_critical = True
        entries = problems["empty_bin_masks"]
        print(f"ERROR: {len(entries)} genes with empty bin masks found")
        if verbose:
            for seq_idx, gene_idx in entries[:10]:
                print(f"  seq {seq_idx}, gene {gene_idx}: bin mask is all-zero")

    # Check if any issues found
    has_any_issue = any(
        problems.get(k)
        for k in [
            "shape_mismatches",
            "zero_variance_targets",
            "high_zero_targets",
            "nan_targets",
            "inf_targets",
            "no_valid_genes",
            "empty_bin_masks",
            "gene_zero_variance_targets",
            "gene_high_zero_targets",
            "gene_nan_targets",
            "gene_inf_targets",
        ]
    )
    if not has_any_issue:
        print("OK: No issues found")

    return 1 if has_critical else 0


def build_json_report(
    data_dir,
    split,
    data_stats,
    coverage_df,
    gene_target_df,
    gene_issues,
    shape_issues,
    problems,
    has_genes,
    has_gene_targets,
):
    """
    Build JSON report for dashboard consumption.
    """
    report = {
        "data_dir": str(data_dir),
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "data_stats": data_stats,
        "shape_issues": shape_issues,
        "has_genes": has_genes,
        "has_gene_targets": has_gene_targets,
        "problems": {
            # Coverage target problems
            "zero_variance_targets": problems.get("zero_variance_targets", []),
            "high_zero_targets": problems.get("high_zero_targets", []),
            "nan_targets": problems.get("nan_targets", []),
            "inf_targets": problems.get("inf_targets", []),
            # Gene target problems
            "gene_zero_variance_targets": problems.get(
                "gene_zero_variance_targets", []
            ),
            "gene_high_zero_targets": problems.get("gene_high_zero_targets", []),
            "gene_nan_targets": problems.get("gene_nan_targets", []),
            "gene_inf_targets": problems.get("gene_inf_targets", []),
            # Gene mask/slice problems
            "num_seqs_no_valid_genes": len(problems.get("no_valid_genes", [])),
            "num_empty_bin_masks": len(problems.get("empty_bin_masks", [])),
        },
    }

    if coverage_df is not None:
        report["coverage_summary"] = {
            "num_targets": len(coverage_df["target"].unique()),
            "num_folds": len(coverage_df["fold"].unique()),
            "mean_variance": float(coverage_df["variance"].mean()),
            "median_variance": float(coverage_df["variance"].median()),
            "mean_pct_zeros": float(coverage_df["pct_zeros"].mean()),
        }

    if gene_target_df is not None:
        report["gene_target_summary"] = {
            "num_targets": len(gene_target_df["target"].unique()),
            "num_folds": len(gene_target_df["fold"].unique()),
            "mean_variance": float(gene_target_df["variance"].mean()),
            "median_variance": float(gene_target_df["variance"].median()),
            "mean_pct_zeros": float(gene_target_df["pct_zeros"].mean()),
        }

    return report


if __name__ == "__main__":
    main()
