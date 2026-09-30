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
import argparse
from collections import defaultdict
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import torch
import zarr

from baskerville import dataset
from baskerville import seqnn
from baskerville.hardware import check_mixed_precision

"""
hound_eval

Evaluate the accuracy of a trained model on held-out sequences.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--aggregate_genes",
        action="store_true",
        help="Aggregate predictions per unique gene across sequences",
    )
    parser.add_argument(
        "-di",
        "--dataset",
        dest="di",
        default=0,
        type=int,
        help="Dataset index to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--rank",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Step across positions [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "-tg",
        "--targets_gene_file",
        default=None,
        help="Gene targets file [Default: {data_dir}/targets_gene.txt]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # inputs

    # read targets (optional for gene-only datasets)
    if args.targets_file is None:
        args.targets_file = f"{args.data_dir}/targets.txt"
    if os.path.exists(args.targets_file):
        targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    else:
        targets_df = None

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params.get("train", {})

    # set strand pairs (using new indexing)
    if targets_df is not None and "strand_pair" in targets_df.columns:
        params_model["strand_pair"] = dataset.strand_pair_indices(targets_df)

    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        mode="eval",
        targets_file=args.targets_file if targets_df is not None else None,
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file)
    seqnn_model.ensemble_rc = args.rc
    seqnn_model.ensemble_shifts = args.shifts
    mix_dtype = params_train.get("mix_dtype", "float32")
    if mix_dtype != "float32":
        if check_mixed_precision():
            if mix_dtype == "float16":
                seqnn_model.mix_dtype = torch.float16
            elif mix_dtype == "bfloat16":
                seqnn_model.mix_dtype = torch.bfloat16
            else:
                print(f"Warning: Unrecognized mixed precision dtype {mix_dtype}")
        else:
            print("Warning: Mixed precision training not supported on this GPU.")

    #######################################################
    # evaluate

    # determine if we need to save values
    return_values = args.save or args.rank or args.aggregate_genes
    zarr_path = (
        os.path.join(args.out_dir, "preds_targets.zarr") if return_values else None
    )

    # run evaluation
    results = seqnn_model.eval(
        eval_data,
        hi=args.di,
        batch_size=params_train.get("batch_size", 1),
        return_values=return_values,
        step=args.step,
        zarr_store=zarr_path,
    )

    #######################################################
    # coverage metrics

    if "coverage" in results:
        cov_results = results["coverage"]
        eval_r = cov_results["r"]
        eval_r2 = cov_results["r2"]

        print("Coverage PearsonR: %7.5f" % np.mean(eval_r))
        print("Coverage R2:       %7.5f" % np.mean(eval_r2))

        # write target-level statistics
        targets_acc_df = pd.DataFrame(
            {
                "index": targets_df.index,
                "pearsonr": eval_r,
                "r2": eval_r2,
                "identifier": targets_df.identifier,
                "description": targets_df.description,
            }
        )

        # specificity Pearson, at plus-representative tracks of scored groups
        if cov_results["spec"] is not None:
            targets_acc_df["spec"] = cov_results["spec"]
            print("Coverage spec:     %7.5f" % np.nanmean(cov_results["spec"]))

        # Spearman R
        if args.rank:
            eval_preds = cov_results["preds"]
            eval_targets = cov_results["targets"]
            eval_spearmanr = []
            for ti in tqdm(range(eval_preds.shape[1]), desc="Spearman"):
                eval_preds_ti = eval_preds[:, ti, :].flatten()
                eval_targets_ti = eval_targets[:, ti, :].flatten()
                spear_ti = spearmanr(eval_preds_ti, eval_targets_ti)[0]
                eval_spearmanr.append(spear_ti)
            targets_acc_df["spearmanr"] = eval_spearmanr
            print("Coverage SpearmanR: %7.5f" % np.mean(eval_spearmanr))

        targets_acc_df.to_csv(
            f"{args.out_dir}/acc.txt", sep="\t", index=False, float_format="%.5f"
        )

    #######################################################
    # gene metrics

    if "gene" in results:
        gene_results = results["gene"]
        eval_r_gene = gene_results["r"]
        eval_r2_gene = gene_results["r2"]

        print("Gene PearsonR:     %7.5f" % np.mean(eval_r_gene))
        print("Gene R2:           %7.5f" % np.mean(eval_r2_gene))

        # load gene targets metadata
        if args.targets_gene_file is None:
            args.targets_gene_file = f"{args.data_dir}/targets_gene.txt"
        targets_gene_df = pd.read_csv(args.targets_gene_file, index_col=0, sep="\t")

        # write per-target gene metrics
        targets_gene_acc_df = pd.DataFrame(
            {
                "index": targets_gene_df.index,
                "pearsonr": eval_r_gene,
                "r2": eval_r2_gene,
                "identifier": targets_gene_df.identifier,
                "description": targets_gene_df.description,
            }
        )
        targets_gene_acc_df.to_csv(
            f"{args.out_dir}/acc_gene.txt", sep="\t", index=False, float_format="%.5f"
        )

        # gene-level aggregation (auto with --save, or explicit --aggregate_genes)
        if args.aggregate_genes or args.save:
            gene_preds = gene_results["preds"]
            gene_targets_arr = gene_results["targets"]
            gene_presence_arr = gene_results["masks"]

            # load gene IDs from split-specific zarr
            split_zarr_path = f"{args.data_dir}/examples/{args.split}.zarr"
            split_zarr = zarr.open_group(split_zarr_path, mode="r")
            try:
                gene_ids = split_zarr["gene_ids"][:]

                # form gene expression matrix
                gm = form_gene_matrix(
                    gene_preds, gene_targets_arr, gene_presence_arr, gene_ids
                )

                # save gene expression matrix as compressed TSV
                if args.save:
                    gene_columns = list(targets_gene_df.identifier)
                    for name, key in [
                        ("gene_preds", "preds"),
                        ("gene_targets", "targets"),
                    ]:
                        df = pd.DataFrame(
                            gm[key], index=gm["ids"], columns=gene_columns
                        )
                        df.index.name = "gene_id"
                        df.to_csv(
                            f"{args.out_dir}/{name}.tsv.gz",
                            sep="\t",
                            float_format="%.4f",
                        )

                # compute and write per-gene metrics
                gene_agg = aggregate_genes(
                    gene_preds,
                    gene_targets_arr,
                    gene_presence_arr,
                    gene_ids,
                    gene_matrix=gm,
                )
                gene_agg.to_csv(
                    f"{args.out_dir}/acc_gene_agg.txt",
                    sep="\t",
                    index=False,
                    float_format="%.5f",
                )
                print(f"Gene aggregated PearsonR: {gene_agg['pearsonr'].mean():.5f}")
            except Exception:
                print("Warning: gene_ids not found, skipping gene aggregation.")
                print("Re-run hound_data to generate gene_ids in split zarr files.")

    #######################################################
    # cleanup

    if return_values and not args.save:
        if zarr_path is not None and os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)


def form_gene_matrix(gene_preds, gene_targets, gene_presence, gene_ids):
    """Average per-sequence gene predictions into a per-gene expression matrix.

    Genes appearing in multiple sequences have their predictions averaged.
    Targets are taken from the first occurrence (identical across sequences).

    Args:
        gene_preds: (num_seqs, num_gene_targets, max_genes)
        gene_targets: (num_seqs, num_gene_targets, max_genes)
        gene_presence: (num_seqs, max_genes) boolean
        gene_ids: (num_seqs, max_genes) string array

    Returns:
        dict with keys:
            ids: sorted list of unique gene ID strings
            preds: (num_genes, num_gene_targets) float64
            targets: (num_genes, num_gene_targets) float64
            counts: (num_genes,) int array, sequences per gene
    """
    num_seqs, num_targets, max_genes = gene_preds.shape

    pred_sums = defaultdict(lambda: np.zeros(num_targets, dtype=np.float64))
    target_vals = {}
    counts = defaultdict(int)

    for si in range(num_seqs):
        for gi in range(max_genes):
            if not gene_presence[si, gi]:
                continue
            gene_id = gene_ids[si, gi]
            if isinstance(gene_id, bytes):
                gene_id = gene_id.decode("utf-8")
            pred_sums[gene_id] += gene_preds[si, :, gi].astype(np.float64)
            counts[gene_id] += 1
            if gene_id not in target_vals:
                target_vals[gene_id] = gene_targets[si, :, gi].astype(np.float64)

    ids = sorted(pred_sums.keys())
    return {
        "ids": ids,
        "preds": np.array([pred_sums[g] / counts[g] for g in ids]),
        "targets": np.array([target_vals[g] for g in ids]),
        "counts": np.array([counts[g] for g in ids]),
    }


def aggregate_genes(
    gene_preds, gene_targets, gene_presence, gene_ids, gene_matrix=None
):
    """Aggregate predictions per gene and compute per-gene accuracy metrics.

    Args:
        gene_preds: (num_seqs, num_gene_targets, max_genes)
        gene_targets: (num_seqs, num_gene_targets, max_genes)
        gene_presence: (num_seqs, max_genes) boolean
        gene_ids: (num_seqs, max_genes) string array
        gene_matrix: Pre-computed result from form_gene_matrix (optional).

    Returns:
        DataFrame with per-gene metrics
    """
    if gene_matrix is None:
        gene_matrix = form_gene_matrix(
            gene_preds, gene_targets, gene_presence, gene_ids
        )

    results = []
    for i, gene_id in enumerate(gene_matrix["ids"]):
        pred = gene_matrix["preds"][i]
        target = gene_matrix["targets"][i]

        valid = ~(np.isnan(pred) | np.isnan(target))
        if valid.sum() >= 2:
            r, _ = pearsonr(pred[valid], target[valid])
            ss_res = np.sum((pred[valid] - target[valid]) ** 2)
            ss_tot = np.sum((target[valid] - target[valid].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r = np.nan
            r2 = np.nan

        results.append(
            {
                "gene_id": gene_id,
                "pearsonr": r,
                "r2": r2,
                "n_sequences": gene_matrix["counts"][i],
            }
        )

    return pd.DataFrame(results)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
