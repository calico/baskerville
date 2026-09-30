#!/usr/bin/env python3
"""
Dataset Quality Control Dashboard

Interactive Streamlit dashboard for exploring dataset quality issues.
Visualizes statistics from hound_data_qc and allows inspection of individual examples.

Usage:
    streamlit run dash_data_qc.py -- -d /path/to/data
    streamlit run dash_data_qc.py  # Enter path in app

The data directory should contain:
- qc.tsv: Per-target statistics from hound_data_qc
- targets.txt: Target metadata (optional, for group information)
- examples/*.zarr: Dataset files (for example viewer)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import zarr
from natsort import natsorted
import glob


def load_qc_data(data_dir):
    """Load QC data from hound_data_qc output.

    Args:
        data_dir: Path to dataset directory

    Returns:
        tuple: (qc_df, targets_df, qc_report) or (None, None, None) on error
    """
    data_path = Path(data_dir)

    # Load QC TSV
    qc_file = data_path / "qc.tsv"
    if not qc_file.exists():
        return None, None, None

    qc_df = pd.read_csv(qc_file, sep="\t")

    # Load targets table if available
    targets_file = data_path / "targets.txt"
    targets_df = None
    if targets_file.exists():
        targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    # Load JSON report if available
    import json

    qc_report = None
    report_file = data_path / "qc_report.json"
    if report_file.exists():
        with open(report_file) as f:
            qc_report = json.load(f)

    return qc_df, targets_df, qc_report


def load_zarr_example(data_dir, seq_idx):
    """Load a specific sequence from zarr files.

    Args:
        data_dir: Path to dataset directory
        seq_idx: Global sequence index

    Returns:
        tuple: (sequence, targets, gene_data) or None
    """
    data_path = Path(data_dir)
    zarr_files = natsorted(glob.glob(str(data_path / "examples" / "*.zarr")))

    if not zarr_files:
        return None

    # Find which zarr file contains this sequence
    cumulative = 0
    for zarr_file in zarr_files:
        zarr_open = zarr.open(zarr_file, mode="r")

        if "target" in zarr_open:
            num_seqs = zarr_open["target"].shape[0]
        elif "gene_presence" in zarr_open:
            num_seqs = zarr_open["gene_presence"].shape[0]
        else:
            continue

        if seq_idx < cumulative + num_seqs:
            local_idx = seq_idx - cumulative

            # Load data
            targets = None
            if "target" in zarr_open:
                targets = zarr_open["target"][local_idx]

            gene_data = None
            if "gene_presence" in zarr_open:
                gene_data = {
                    "mask": zarr_open["gene_presence"][local_idx],
                    "bin_mask": zarr_open["gene_out_mask"][local_idx],
                }
                if "gene_target" in zarr_open:
                    gene_data["targets"] = zarr_open["gene_target"][local_idx]

            return targets, gene_data

        cumulative += num_seqs

    return None


def show_summary(qc_df, qc_report, targets_df):
    """Display summary health metrics."""
    st.header("Dataset Health Summary")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_targets = qc_df["target"].nunique()
        st.metric("Targets", num_targets)

    with col2:
        num_folds = qc_df["fold"].nunique()
        st.metric("Folds", num_folds)

    with col3:
        # Zero variance targets
        zero_var = qc_df[qc_df["variance"] < 1e-8]["target"].nunique()
        st.metric(
            "Zero Variance",
            zero_var,
            delta=None if zero_var == 0 else f"{zero_var} targets",
            delta_color="inverse",
        )

    with col4:
        # NaN targets
        nan_targets = qc_df[qc_df["pct_nan"] > 0]["target"].nunique()
        st.metric(
            "Contains NaN",
            nan_targets,
            delta=None if nan_targets == 0 else f"{nan_targets} targets",
            delta_color="inverse",
        )

    # Show problems from JSON report if available
    if qc_report and "problems" in qc_report:
        problems = qc_report["problems"]

        st.subheader("Issues Detected")

        issues = []
        if problems.get("zero_variance_targets"):
            issues.append(
                f"**Zero variance:** {len(problems['zero_variance_targets'])} targets"
            )
        if problems.get("high_zero_targets"):
            issues.append(
                f"**>99% zeros:** {len(problems['high_zero_targets'])} targets"
            )
        if problems.get("nan_targets"):
            issues.append(f"**Contains NaN:** {len(problems['nan_targets'])} targets")
        if problems.get("inf_targets"):
            issues.append(f"**Contains Inf:** {len(problems['inf_targets'])} targets")
        if problems.get("num_seqs_no_valid_genes", 0) > 0:
            issues.append(
                f"**No valid genes:** {problems['num_seqs_no_valid_genes']} sequences"
            )
        if problems.get("num_invalid_slices", 0) > 0:
            issues.append(
                f"**Invalid gene slices:** {problems['num_invalid_slices']} slices"
            )

        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("No issues detected")


def show_target_statistics(qc_df, targets_df):
    """Display interactive target statistics with group-wise distributions."""
    st.header("Target Statistics")

    # Determine if we have groups
    has_groups = "group" in qc_df.columns
    if not has_groups and targets_df is not None and "group" in targets_df.columns:
        # Merge group info from targets_df
        qc_df = qc_df.merge(
            targets_df[["group"]].reset_index(),
            left_on="target",
            right_on="index",
            how="left",
        )
        has_groups = "group" in qc_df.columns

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_variance = st.number_input(
            "Min variance",
            min_value=0.0,
            max_value=float(qc_df["variance"].max()),
            value=0.0,
            format="%.2e",
        )

    with col2:
        max_pct_zeros = st.slider("Max % zeros", 0.0, 100.0, 100.0)

    with col3:
        if has_groups:
            groups = ["All"] + sorted(qc_df["group"].dropna().unique().tolist())
            selected_group = st.selectbox("Group", groups)
        else:
            selected_group = "All"

    # Apply filters
    filtered = qc_df[
        (qc_df["variance"] >= min_variance) & (qc_df["pct_zeros"] <= max_pct_zeros)
    ]
    if selected_group != "All":
        filtered = filtered[filtered["group"] == selected_group]

    st.write(f"Showing {len(filtered)} of {len(qc_df)} rows")

    # Data table
    st.subheader("Target Data")
    display_cols = ["fold", "target", "mean", "variance", "min", "max", "pct_zeros"]
    if "pct_nan" in filtered.columns:
        display_cols.append("pct_nan")
    if "pct_inf" in filtered.columns:
        display_cols.append("pct_inf")
    if has_groups:
        display_cols.append("group")

    st.dataframe(
        filtered[display_cols].sort_values("variance"),
        use_container_width=True,
        height=300,
    )

    # Group-wise distribution plots
    st.subheader("Distributions by Group")

    if has_groups:
        # Aggregate by target (average across folds)
        target_stats = (
            qc_df.groupby(["target", "group"])
            .agg({"mean": "mean", "variance": "mean", "pct_zeros": "mean"})
            .reset_index()
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                target_stats,
                x="group",
                y="variance",
                title="Variance Distribution by Group",
                log_y=True,
                points="outliers",
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                target_stats,
                x="group",
                y="mean",
                title="Mean Distribution by Group",
                log_y=True,
                points="outliers",
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # % Zeros distribution
        fig = px.box(
            target_stats,
            x="group",
            y="pct_zeros",
            title="% Zeros Distribution by Group",
            points="outliers",
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # No groups - show overall histograms
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                qc_df, x="variance", nbins=50, title="Variance Distribution", log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                qc_df, x="pct_zeros", nbins=50, title="% Zeros Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_gene_validation(qc_report):
    """Display gene validation results."""
    st.header("Gene Validation")

    if qc_report is None:
        st.info("Run `hound_data_qc --json` to generate gene validation data.")
        return

    if not qc_report.get("has_genes", False):
        st.info("No gene data in this dataset.")
        return

    problems = qc_report.get("problems", {})

    col1, col2 = st.columns(2)

    with col1:
        no_genes = problems.get("num_seqs_no_valid_genes", 0)
        st.metric("Sequences with no valid genes", no_genes)

    with col2:
        invalid = problems.get("num_invalid_slices", 0)
        st.metric("Invalid gene slices", invalid)

    if no_genes > 0 or invalid > 0:
        st.warning(
            "Gene data issues detected. These sequences may cause NaN metrics during training."
        )


def show_example_viewer(data_dir, qc_df, targets_df):
    """Interactive viewer for individual sequences."""
    st.header("Example Viewer")

    # Get available targets
    num_targets = qc_df["target"].nunique()

    col1, col2 = st.columns(2)

    with col1:
        target_idx = st.number_input(
            "Target index", min_value=0, max_value=num_targets - 1, value=0
        )

        # Show target info
        target_stats = qc_df[qc_df["target"] == target_idx].iloc[0]
        st.write(f"**Mean:** {target_stats['mean']:.4f}")
        st.write(f"**Variance:** {target_stats['variance']:.4e}")
        st.write(f"**% Zeros:** {target_stats['pct_zeros']:.1f}%")

    with col2:
        # Estimate max sequence index from first zarr file
        zarr_files = natsorted(glob.glob(str(Path(data_dir) / "examples" / "*.zarr")))
        max_seq = 0
        if zarr_files:
            for zf in zarr_files:
                zo = zarr.open(zf, mode="r")
                if "target" in zo:
                    max_seq += zo["target"].shape[0]

        seq_idx = st.number_input(
            "Sequence index",
            min_value=0,
            max_value=max(0, max_seq - 1),
            value=0,
        )

    if st.button("Load Example"):
        with st.spinner("Loading from zarr..."):
            result = load_zarr_example(data_dir, seq_idx)

        if result is None:
            st.error(f"Could not load sequence {seq_idx}")
            return

        targets, gene_data = result

        if targets is not None:
            # Plot coverage track for selected target
            track = targets[target_idx]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=track,
                    mode="lines",
                    name=f"Target {target_idx}",
                    line=dict(width=1),
                )
            )
            fig.update_layout(
                title=f"Coverage Track - Sequence {seq_idx}, Target {target_idx}",
                xaxis_title="Position (bins)",
                yaxis_title="Coverage",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Track statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{track.mean():.4f}")
            with col2:
                st.metric("Variance", f"{track.var():.4e}")
            with col3:
                st.metric("Min", f"{track.min():.4f}")
            with col4:
                st.metric("Max", f"{track.max():.4f}")

        if gene_data is not None:
            st.subheader("Gene Data")
            mask = gene_data["mask"]
            bin_mask = gene_data["bin_mask"]

            num_valid = mask.sum()
            st.write(f"**Valid genes:** {num_valid} / {len(mask)}")

            if num_valid > 0:
                # Show first few genes
                st.write("**Gene bin masks (first 10):**")
                gene_info = []
                for gi in range(min(10, len(mask))):
                    if mask[gi]:
                        bins = bin_mask[gi].nonzero()[0]
                        gene_info.append(
                            {
                                "gene": gi,
                                "num_bins": len(bins),
                                "first_bin": int(bins[0]) if len(bins) > 0 else -1,
                                "last_bin": int(bins[-1]) if len(bins) > 0 else -1,
                            }
                        )
                st.dataframe(pd.DataFrame(gene_info))


def main(default_data_dir=None):
    st.set_page_config(page_title="Dataset QC Dashboard", layout="wide")

    st.title("Dataset Quality Control Dashboard")
    st.markdown("Explore dataset statistics and identify quality issues")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    default_value = default_data_dir or ""
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=default_value,
        help="Path to dataset directory containing qc.tsv and examples/*.zarr",
    )

    if not data_dir:
        st.info("Enter the path to your dataset directory in the sidebar.")
        st.markdown("""
        ### Expected Directory Structure
        ```
        data_directory/
        ├── qc.tsv              # From hound_data_qc
        ├── qc_report.json      # From hound_data_qc --json (optional)
        ├── targets.txt         # Target metadata (optional)
        ├── statistics.json     # Dataset statistics
        └── examples/
            ├── train0.zarr
            ├── train1.zarr
            └── ...
        ```

        ### Generate QC Data
        ```bash
        hound_data_qc /path/to/data --json
        ```
        """)
        return

    # Load data
    with st.spinner("Loading QC data..."):
        qc_df, targets_df, qc_report = load_qc_data(data_dir)

    if qc_df is None:
        st.error(
            f"Could not load qc.tsv from {data_dir}. "
            "Run `hound_data_qc` first to generate QC data."
        )
        return

    st.success(
        f"Loaded QC data: {qc_df['target'].nunique()} targets, "
        f"{qc_df['fold'].nunique()} folds"
    )

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Target Statistics", "Gene Validation", "Example Viewer"]
    )

    with tab1:
        show_summary(qc_df, qc_report, targets_df)

    with tab2:
        show_target_statistics(qc_df, targets_df)

    with tab3:
        show_gene_validation(qc_report)

    with tab4:
        show_example_viewer(data_dir, qc_df, targets_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Quality Control Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=None,
        help="Default data directory path",
    )
    args = parser.parse_args()

    main(default_data_dir=args.data_dir)
