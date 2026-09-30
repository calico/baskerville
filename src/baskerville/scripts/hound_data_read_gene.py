#!/usr/bin/env python
# Copyright 2024 Calico Life Sciences LLC
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
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr

"""
hound_data_read_gene

Read gene expression values for one RNA-seq sample across all sequences.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Read gene expression values for one RNA-seq sample."
    )
    parser.add_argument(
        "-d",
        dest="data_dir",
        required=True,
        help="Data directory",
    )
    parser.add_argument(
        "-t",
        dest="target_index",
        type=int,
        required=True,
        help="Gene target index [Default: %(default)s]",
    )
    args = parser.parse_args()

    # Read targets_gene file
    targets_gene_file = f"{args.data_dir}/targets_gene.txt"
    if not os.path.exists(targets_gene_file):
        print(
            f"Error: targets_gene.txt not found at {targets_gene_file}", file=sys.stderr
        )
        exit(1)

    targets_gene_df = pd.read_csv(targets_gene_file, index_col=0, sep="\t")
    gene_target = targets_gene_df.iloc[args.target_index]

    print(
        f"Processing gene target {args.target_index}: {gene_target.get('identifier', 'unknown')}"
    )

    # Read gene expression file for this target (sample)
    if not os.path.exists(gene_target.file):
        print(
            f"Error: Gene expression file not found at {gene_target.file}",
            file=sys.stderr,
        )
        exit(1)

    expr_df = pd.read_csv(gene_target.file, sep="\t", index_col=0)
    # Assume expr_df has columns: gene_id (index), expression
    # If it has multiple columns, use the first numerical column
    if expr_df.shape[1] > 1:
        # Find first numerical column
        for col in expr_df.columns:
            if pd.api.types.is_numeric_dtype(expr_df[col]):
                expr_df = expr_df[[col]]
                expr_df.columns = ["expression"]
                break
    elif expr_df.shape[1] == 1:
        expr_df.columns = ["expression"]
    else:
        print(f"Error: Gene expression file has no data columns", file=sys.stderr)
        exit(1)

    print(f"Loaded expression for {len(expr_df)} genes")

    # Strip version suffixes for robust matching across GENCODE versions
    expr_df.index = expr_df.index.str.split(".").str[0]

    # Read gene mapping from zarr store
    gene_data_dir = f"{args.data_dir}/seqs_gene"
    gene_mapping_path = f"{gene_data_dir}/gene_mapping.zarr"
    if not os.path.exists(gene_mapping_path):
        print(
            f"Error: Gene mapping zarr not found at {gene_mapping_path}",
            file=sys.stderr,
        )
        exit(1)

    gene_mapping = zarr.open(gene_mapping_path, mode="r")
    gene_ids_arr = gene_mapping["gene_ids"]
    gene_presence_arr = gene_mapping["gene_presence"]

    num_seqs = gene_ids_arr.shape[0]
    max_genes = gene_ids_arr.shape[1]
    print(f"Processing {num_seqs} sequences with max {max_genes} genes per sequence")

    # Output file
    seqs_gene_file = f"{gene_data_dir}/{args.target_index}.h5"

    # Collect gene expression values for each sequence
    gene_values = []
    genes_found = 0
    genes_missing = 0

    for si in tqdm(range(num_seqs), desc="Reading gene values"):
        # Load gene mapping for this sequence from zarr
        gene_ids = gene_ids_arr[si]
        gene_presence = gene_presence_arr[si]

        seq_gene_values = np.zeros(max_genes, dtype="float32")

        # Fill in expression values for mapped genes
        for gi in range(max_genes):
            if not gene_presence[gi]:
                continue
            gene_id = gene_ids[gi]
            if gene_id in expr_df.index:
                seq_gene_values[gi] = expr_df.loc[gene_id, "expression"]
                genes_found += 1
                # Apply scaling if specified
                if hasattr(gene_target, "scale"):
                    seq_gene_values[gi] *= gene_target.scale
            else:
                genes_missing += 1

        # Apply transformation based on sum_stat
        if hasattr(gene_target, "sum_stat") and pd.notna(gene_target.sum_stat):
            if gene_target.sum_stat == "sqrt":
                seq_gene_values = np.sqrt(seq_gene_values)
            elif gene_target.sum_stat == "log2":
                seq_gene_values = np.log2(seq_gene_values + 1)
            elif gene_target.sum_stat not in [None, "none", ""]:
                raise ValueError(
                    f"Unrecognized sum_stat for gene target: {gene_target.sum_stat}"
                )

        gene_values.append(seq_gene_values)

    print(f"Genes found: {genes_found}, missing: {genes_missing}")

    # Save to H5
    with h5py.File(seqs_gene_file, "w") as h5f:
        h5f.create_dataset("target", data=np.array(gene_values, dtype="float16"))

    print(f"Saved gene expression data to {seqs_gene_file}")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
