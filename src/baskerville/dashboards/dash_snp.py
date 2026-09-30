#!/usr/bin/env python3
"""
SNP Scores Visualization Dashboard

Interactive Streamlit dashboard for visualizing SNP scores computed by ensemble models.
Displays SNP-target combinations grouped by target categories with interactive filtering.

Features:
- Raw SNP scores for different statistics (SUM, logSUM, D1, logD1, D2, logD2)
- Quantile scores that map raw scores to percentile positions (0-1 scale)
- Interactive filtering by SNP and target groups
- Sortable table with downloadable results
- Score distribution visualizations

Quantile scores enable better comparison across different targets and score types by
normalizing scores to their position in the distribution. A quantile of 0.9 means
the score is in the top 10% for that target, regardless of the raw score scale.

Usage:
    python dash_snp.py -s /path/to/ensemble/scores/directory
    python dash_snp.py --scores_dir /path/to/ensemble/scores/directory
    python dash_snp.py  # Run without default directory

The scores directory should contain:
- scores.h5: Ensemble-averaged scores file
- targets.txt: Target information file

The scores directory will be pre-populated in the app interface if provided via -s/--scores_dir.
Users can still modify the path in the text box within the app.
"""

import argparse
import streamlit as st
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from baskerville.snps import compute_score_quantiles


def load_scores_data(scores_dir):
    """Load SNP scores from ensemble directory.

    Args:
        scores_dir (str): Directory containing ensemble scores file (scores.h5)

    Returns:
        tuple: (snps, targets_df, scores_dict, available_score_types, quantiles_data)
    """
    # Read targets
    targets_file = Path(scores_dir) / "targets.txt"
    if not targets_file.exists():
        st.error(f"Targets file not found: {targets_file}")
        return None, None, None, None, None

    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    # Read ensemble scores
    scores_file = Path(scores_dir) / "scores.h5"
    if not scores_file.exists():
        st.error(f"Ensemble scores file not found: {scores_file}")
        return None, None, None, None, None

    # Define valid score types
    valid_score_types = ["SUM", "logSUM", "D1", "logD1", "D2", "logD2"]

    try:
        with h5py.File(scores_file, "r") as scores_h5:
            # Read SNPs
            snps = [s.decode("utf-8") for s in scores_h5["snp"][:]]

            # Get available score types
            available_scores = [
                key for key in scores_h5.keys() if key in valid_score_types
            ]

            if not available_scores:
                st.error("No valid score types found in scores file!")
                return None, None, None, None, None

            # Load scores directly (already averaged in ensemble)
            scores_dict = {}
            for score_type in available_scores:
                scores_dict[score_type] = scores_h5[score_type][:].astype(np.float32)

            # Load quantiles data if available
            quantiles_data = {}
            if "quantiles" in scores_h5:
                quantiles_data["quantiles"] = scores_h5["quantiles"][:]

                # Load quantile thresholds for each score type
                for score_type in available_scores:
                    quantiles_key = f"{score_type}_quantiles"
                    if quantiles_key in scores_h5:
                        quantiles_data[quantiles_key] = scores_h5[quantiles_key][:]

    except Exception as e:
        st.error(f"Error loading ensemble scores from {scores_file}: {e}")
        return None, None, None, None, None

    return snps, targets_df, scores_dict, available_scores, quantiles_data


def create_snp_target_table(
    snps, targets_df, scores_dict, target_mask, quantiles_data=None
):
    """Create a long-format DataFrame with SNP-target combinations.

    Args:
        snps: List of SNP identifiers
        targets_df: DataFrame with target information
        scores_dict: Dictionary of ensemble scores for each score type
        target_mask: Boolean mask for selecting targets
        quantiles_data: Dictionary containing quantiles and quantile thresholds (optional)

    Returns:
        DataFrame with columns: snp, target_index, [score_types...], [quantile_columns...], identifier, description
    """
    snp_target_data = []

    # Compute quantile scores if quantiles data is available
    quantile_scores = {}
    if quantiles_data and "quantiles" in quantiles_data:
        quantiles = quantiles_data["quantiles"]
        for score_type, scores in scores_dict.items():
            quantiles_key = f"{score_type}_quantiles"
            if quantiles_key in quantiles_data:
                # Get quantile thresholds for selected targets
                quantile_thresholds = quantiles_data[quantiles_key][
                    targets_df[target_mask].index, :
                ]
                # Compute quantile positions for scores
                quantile_scores[f"{score_type}_quantile"] = compute_score_quantiles(
                    scores[:, targets_df[target_mask].index],
                    quantile_thresholds,
                    quantiles,
                )

    for i, snp in enumerate(snps):
        for j, target_idx in enumerate(targets_df[target_mask].index):
            row_data = {
                "snp": snp,
                "target_index": target_idx,
                "identifier": targets_df.loc[target_idx, "identifier"],
                "description": targets_df.loc[target_idx, "description"],
            }

            # Add all available score types
            for score_type, scores in scores_dict.items():
                row_data[score_type] = scores[i, target_idx]

                # Add quantile if available
                quantile_key = f"{score_type}_quantile"
                if quantile_key in quantile_scores:
                    row_data[quantile_key] = quantile_scores[quantile_key][i, j]

            snp_target_data.append(row_data)

    return pd.DataFrame(snp_target_data)


def main(default_scores_dir=None):
    st.set_page_config(page_title="SNP Scores Dashboard", page_icon="🧬", layout="wide")

    st.title("🧬 SNP Scores Visualization Dashboard")
    st.markdown("Interactive visualization of SNP scores computed by ensemble models")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Directory input - use default if provided
    default_value = default_scores_dir or ""
    scores_dir = st.sidebar.text_input(
        "Scores Directory",
        value=default_value,
        help="Directory containing ensemble scores file (scores.h5) and targets.txt",
    )

    if default_scores_dir:
        st.sidebar.info(f"Default directory from command line: {default_scores_dir}")

    if not scores_dir:
        st.info("Please enter the path to your scores directory in the sidebar.")
        st.markdown("""
        ### Expected Directory Structure:
        ```
        scores_directory/
        ├── targets.txt
        └── scores.h5  # Ensemble scores file
        ```
        """)
        return

    # Load data
    with st.spinner("Loading scores data..."):
        snps, targets_df, scores_dict, available_score_types, quantiles_data = (
            load_scores_data(scores_dir)
        )

    if snps is None:
        return

    # Check if quantiles are available
    has_quantiles = quantiles_data and "quantiles" in quantiles_data
    quantiles_info = ""
    if has_quantiles:
        n_quantiles = len(quantiles_data["quantiles"])
        available_quantile_types = [
            score_type
            for score_type in available_score_types
            if f"{score_type}_quantiles" in quantiles_data
        ]
        if available_quantile_types:
            quantiles_info = f" Quantiles available for: {', '.join(available_quantile_types)} ({n_quantiles} quantile levels)."
        else:
            has_quantiles = False
            quantiles_info = " (Quantile data found but no valid quantile thresholds for available score types.)"

    st.success(
        f"Loaded {len(snps)} SNPs and {len(targets_df)} targets. Available score types: {', '.join(available_score_types)}.{quantiles_info}"
    )

    # Show available target groups
    available_groups = targets_df["group"].unique()
    # Filter out nan/null groups
    available_groups = [
        group
        for group in available_groups
        if pd.notna(group) and str(group).lower() != "nan"
    ]
    group_counts = [
        (group, (targets_df["group"] == group).sum()) for group in available_groups
    ]
    group_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending

    st.sidebar.markdown("### Available Target Groups:")
    for group, count in group_counts:
        st.sidebar.write(f"- {group}: {count} targets")

    # Main content
    st.header("SNP Scores Analysis")

    # Group selection - use sorted groups
    selected_group = st.selectbox(
        "Select Target Group",
        options=[group for group, _ in group_counts],
        help="Choose which target group to visualize",
    )

    # Create data for selected group
    target_mask = targets_df["group"] == selected_group
    group_df = create_snp_target_table(
        snps,
        targets_df,
        scores_dict,
        target_mask,
        quantiles_data,
    )

    # Determine default sort column after creating the data
    # Default to logD2 quantile if available, then logD2 raw score, otherwise first available score type
    if "logD2_quantile" in group_df.columns:
        default_score = "logD2_quantile"
    elif "logD2" in available_score_types:
        default_score = "logD2"
    else:
        default_score = available_score_types[0]

    st.header(f"{selected_group.upper()} Target Group")

    # Add quantiles explanation if available
    if has_quantiles:
        with st.expander("ℹ️ About Quantile Scores", expanded=False):
            st.markdown("""
                **Quantile scores** make it easier to compare SNP effects across different targets and score types 
                by mapping raw scores to their percentile position in the distribution.
                
                - **0.0**: Score is at the minimum (0th percentile)
                - **0.5**: Score is at the median (50th percentile) 
                - **0.9**: Score is in the top 10% (90th percentile)
                - **0.99**: Score is in the top 1% (99th percentile)
                - **1.0**: Score is at the maximum (100th percentile)
                
                Use quantile scores to identify SNPs with consistently high or low effects across targets,
                regardless of the different scales and distributions of raw score types.
                """)

    # SNP exclusion feature
    st.subheader("Exclude SNPs from Analysis")
    excluded_snps_text = st.text_area(
        "Enter SNP IDs to exclude (one per line or comma-separated):",
        value="",
        height=100,
        help="Copy and paste SNP identifiers here to exclude them from the table. You can use multiple lines or separate with commas.",
    )

    # Parse excluded SNPs
    excluded_snps = set()
    if excluded_snps_text.strip():
        # Handle both comma-separated and newline-separated
        excluded_snps_raw = excluded_snps_text.replace(",", "\n").split("\n")
        excluded_snps = {snp.strip() for snp in excluded_snps_raw if snp.strip()}
        if excluded_snps:
            st.info(
                f"Excluding {len(excluded_snps)} SNP(s): {', '.join(list(excluded_snps)[:5])}{'...' if len(excluded_snps) > 5 else ''}"
            )

    # Controls in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # SNP selection for filtering
        selected_snp = st.selectbox(
            "Filter by SNP (optional)",
            options=["All SNPs"] + sorted(group_df["snp"].unique()),
            help="Select a specific SNP to focus on",
        )

    with col2:
        # Score display mode
        if has_quantiles:
            show_quantiles = st.selectbox(
                "Score Display",
                options=["Raw Scores", "Quantile Scores", "Both"],
                index=2,  # Default to "Both"
                help="Choose whether to show raw scores, quantiles, or both",
            )
        else:
            show_quantiles = "Raw Scores"

    with col3:
        # Sort column selection
        sort_options = [
            "snp",
            "target_index",
            "identifier",
            "description",
        ] + available_score_types

        # Add quantile columns to sort options if available
        if has_quantiles:
            quantile_columns = [
                f"{score_type}_quantile"
                for score_type in available_score_types
                if f"{score_type}_quantiles" in quantiles_data
            ]
            sort_options.extend(quantile_columns)

        sort_column = st.selectbox(
            "Sort by Column",
            options=sort_options,
            index=(
                sort_options.index(default_score)
                if default_score in sort_options
                else 0
            ),
            help="Choose which column to sort by",
        )

    with col4:
        # Sort direction
        sort_ascending = st.selectbox(
            "Sort Direction",
            options=[False, True],
            format_func=lambda x: (
                "Descending (High → Low)" if not x else "Ascending (Low → High)"
            ),
            help="Choose sort direction - useful for signed scores like SUM",
        )

    # Filter data
    if selected_snp != "All SNPs":
        display_df = group_df[group_df["snp"] == selected_snp].copy()
    else:
        display_df = group_df.copy()

    # Apply SNP exclusions
    if excluded_snps:
        display_df = display_df[~display_df["snp"].isin(excluded_snps)]

    # Sort data
    display_df = display_df.sort_values(sort_column, ascending=sort_ascending)

    # Display table
    st.subheader("SNP-Target Pairs")
    st.markdown(
        f"*Sorted by {sort_column} ({'ascending' if sort_ascending else 'descending'})*"
    )

    # Determine which columns to show based on user selection
    display_cols = ["snp", "identifier", "description"]

    # Add score types and their quantiles based on display mode
    for score_type in available_score_types:
        if show_quantiles in ["Raw Scores", "Both"]:
            display_cols.append(score_type)
        if show_quantiles in ["Quantile Scores", "Both"]:
            quantile_col = f"{score_type}_quantile"
            if quantile_col in group_df.columns:
                display_cols.append(quantile_col)

    # Use column_config to format score columns nicely
    column_config = {}
    for score_type in available_score_types:
        if show_quantiles in ["Raw Scores", "Both"]:
            column_config[score_type] = st.column_config.NumberColumn(
                score_type, format="%.4f"
            )

        # Format quantile columns
        if show_quantiles in ["Quantile Scores", "Both"]:
            quantile_col = f"{score_type}_quantile"
            if quantile_col in group_df.columns:
                column_config[quantile_col] = st.column_config.NumberColumn(
                    f"{score_type} quantile",
                    format="%.3f",
                    help=f"Quantile position for {score_type} score (0-1 scale)",
                )

    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config=column_config,
    )

    # Visualizations
    st.header("Score Distributions")

    col1, col2 = st.columns(2)

    with col1:
        # Histogram of currently selected sort column (if it's a score type)
        viz_score = (
            sort_column if sort_column in available_score_types else default_score
        )
        fig_hist = px.histogram(
            display_df.head(1000),  # Limit for performance
            x=viz_score,
            nbins=50,
            title=f"{viz_score} Distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Top SNPs by max score
        if selected_snp == "All SNPs":
            viz_score_for_top = viz_score
            snp_max_scores = (
                group_df.groupby("snp")[viz_score_for_top]
                .max()
                .sort_values(ascending=False)
                .head(20)
            )
            fig_bar = px.bar(
                x=snp_max_scores.index,
                y=snp_max_scores.values,
                title=f"Top 20 SNPs by Max {viz_score_for_top}",
                labels={"x": "SNP", "y": f"Max {viz_score_for_top}"},
            )
            fig_bar.update_layout(xaxis={"tickangle": 45})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Scatter plot for specific SNP - use first two score types available
            if len(available_score_types) >= 2:
                score_x = available_score_types[0]
                score_y = available_score_types[1]
                fig_scatter = px.scatter(
                    display_df.head(100),
                    x=score_x,
                    y=score_y,
                    hover_data=["identifier"],
                    title=f"{score_x} vs {score_y} for {selected_snp}",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Need at least 2 score types for scatter plot")

    # Download section
    st.header("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download Filtered Results as CSV"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_group}_{selected_snp}_filtered.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Download All Group Data as CSV"):
            csv = group_df.to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"{selected_group}_all_snp_target_pairs.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNP Scores Visualization Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default scores directory
  python dash_snp.py -s /path/to/scores/directory
  python dash_snp.py --scores_dir /path/to/scores/directory
  
  # Run without default (user will need to enter path in the app)
  python dash_snp.py
        """.strip(),
    )

    parser.add_argument(
        "-s",
        "--scores_dir",
        type=str,
        help="Default scores directory path to populate in the app",
    )

    args = parser.parse_args()

    # Validate the scores directory if provided
    if args.scores_dir:
        scores_path = Path(args.scores_dir)
        if not scores_path.exists():
            print(f"Warning: Scores directory does not exist: {args.scores_dir}")
        elif not scores_path.is_dir():
            print(f"Warning: Scores path is not a directory: {args.scores_dir}")
        else:
            # Check if it looks like a valid scores directory
            targets_file = scores_path / "targets.txt"
            if not targets_file.exists():
                print(f"Warning: targets.txt not found in {args.scores_dir}")

    # Run the Streamlit app with the default directory
    main(default_scores_dir=args.scores_dir)
