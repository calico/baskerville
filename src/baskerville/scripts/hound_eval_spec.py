#!/usr/bin/env python
# Copyright 2020 Calico Life Sciences LLC
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
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from baskerville import dataset
from baskerville import seqnn
from baskerville import spec_norm
from baskerville.hardware import check_mixed_precision

"""
hound_eval_spec

Test the accuracy of a trained model on targets/predictions normalized across targets.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--band",
        default=spec_norm.DEFAULT_BAND_SIZE,
        type=int,
        help="Tracks read per band during streaming normalization [Default: %(default)s]",
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
        "-m",
        "--group_min",
        default=20,
        type=int,
        help="Minimum target group size to consider [Default: %(default)s]",
    )
    parser.add_argument(
        "--ncpus",
        default=2,
        type=int,
        help="Threads for column sort/normalization [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="spec_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--ram",
        default=False,
        action="store_true",
        help="Hold preds/targets in RAM instead of streaming to a Zarr store on disk "
        "(default streams to disk; only use --ram on big-memory nodes)",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--scratch_dir",
        default=None,
        help="Directory for the temporary preds/targets Zarr store. Must be a "
        "real disk with room (NOT a tmpfs like many /tmp mounts) "
        "[Default: the output directory]",
    )
    parser.add_argument(
        "--seq_chunk",
        default=None,
        type=int,
        help="Zarr seq-axis chunk for the preds/targets store; wider chunks cut the "
        "file count and per-column read ops (writes are buffered a chunk at a time; "
        "RAM ≈ 2 * seq_chunk * num_targets * bins * 2 bytes for preds+targets). "
        "Default: None (uses batch_size) [Default: %(default)s]",
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
        "--target_groups",
        default=None,
        type=str,
        help="Comma separated string of target groups",
    )
    parser.add_argument(
        "--var_pct",
        default=1.0,
        type=float,
        help="Highly variable site proportion to take [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # targets

    # read table
    if args.targets_file is None:
        args.targets_file = f"{args.data_dir}/targets.txt"
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    target_slice = targets_df.index.values
    num_targets = targets_df.shape[0]

    # set target groups
    targets_df["group"] = dataset.target_groups(targets_df)

    if args.target_groups is None:
        args.target_groups = sorted(set(targets_df.group))
    else:
        args.target_groups = args.target_groups.split(",")

    #######################################################
    # model

    # read parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params.get("train", {})

    # update strand pairs for new indexing (positional partner per target)
    strand_pair_pos = dataset.strand_pair_indices(targets_df)
    params_model["strand_pair"] = strand_pair_pos

    # Collapse stranded pairs to their plus representative: one column per
    # experiment, summed at eval write-time (combine_pairs below) -- store is
    # ~half size for stranded assays.
    rep_pos, pair_pos = dataset.strand_collapse(targets_df)
    targets_strand_df = targets_df.iloc[rep_pos]
    store_columns = [
        (int(r),) if p == r else (int(r), int(p)) for r, p in zip(rep_pos, pair_pos)
    ]
    num_store_targets = len(store_columns)

    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        mode="eval",
        targets_file=args.targets_file,
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model, output_slice=target_slice)
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
    # targets/predictions

    # By default stream preds/targets to a Zarr store on disk so the full
    # (num_seqs, num_targets, bins) arrays never live in RAM (--ram keeps them in
    # memory, only viable on big-memory nodes). Scratch is --scratch_dir, else the
    # output dir -- deliberately NOT $TMPDIR or /tmp: SLURM sets $TMPDIR=/tmp on the
    # compute nodes, a small mount shared across jobs where big stores ENOSPC-crash.
    # The output dir is per-job on real disk (NFS here, ~TB free); point
    # --scratch_dir at node-local scratch only if it is a real disk with room.
    tmp_root = None
    zarr_path = None
    if args.ram and args.seq_chunk:
        print(
            "Warning: --seq_chunk is ignored with --ram (no on-disk store)", flush=True
        )
    if not args.ram:
        scratch_root = args.scratch_dir or args.out_dir
        tmp_root = tempfile.mkdtemp(prefix="hound_spec_", dir=scratch_root)
        zarr_path = os.path.join(tmp_root, "preds_targets.zarr")
        free_gb = shutil.disk_usage(tmp_root).free / 1e9
        # uncompressed upper bound for preds + targets (float16); on-disk is
        # smaller after compression, but fail fast if there is clearly no room
        est_gb = (
            2
            * len(eval_data)
            * num_store_targets
            * (eval_data.target_length // args.step)
            * 2
        ) / 1e9
        print(
            "Streaming preds/targets to %s (%.0f GB free, ~%.0f GB uncompressed)"
            % (zarr_path, free_gb, est_gb),
            flush=True,
        )
        if free_gb < 0.4 * est_gb:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise OSError(
                "Not enough scratch space at %s (%.0f GB free, ~%.0f GB needed). "
                "Set --scratch_dir to a disk with room, raise --step, or use --ram."
                % (scratch_root, free_gb, est_gb)
            )

    try:
        # target_chunk chunks the store along targets so each band read (below)
        # decompresses only its own tracks; combine_pairs sums each stranded pair
        # at write-time -> one column per experiment, in targets_strand_df order.
        # None when nothing collapses (identity) to skip a per-batch gather.
        combine_pairs = store_columns if num_store_targets < num_targets else None
        results = seqnn_model.eval(
            eval_data,
            hi=args.di,
            batch_size=params_train.get("batch_size", 1),
            return_values=True,
            step=args.step,
            zarr_store=zarr_path,
            target_chunk=args.band,
            seq_chunk=args.seq_chunk,
            combine_pairs=combine_pairs,
        )

        if "coverage" not in results:
            raise ValueError(
                "Expected coverage evaluation results, but coverage head is unavailable."
            )

        cov_results = results["coverage"]
        eval_preds = cov_results["preds"]
        eval_targets = cov_results["targets"]

        ###################################################
        # process groups (collapsed / plus-represented frame)

        group_values = targets_strand_df.group.values
        targets_spec = np.full(num_store_targets, np.nan)

        for tg in args.target_groups:
            # column indices into the store (== positions in targets_strand_df)
            col_idx = np.where(group_values == tg)[0]
            num_targets_group = len(col_idx)
            print("%-15s  %4d" % (tg, num_targets_group), flush=True)

            # group_min counts experiments (columns), the cross-track dimension
            if num_targets_group < args.group_min:
                continue

            # streaming quantile-normalized cross-track specificity
            t0 = time.time()
            print(" Quantile normalize + correlate...", flush=True, end="")
            col_pearsonr = spec_norm.group_specificity_pearson(
                eval_preds,
                eval_targets,
                col_idx,
                var_pct=args.var_pct,
                band_size=args.band,
                ncpus=args.ncpus,
            )
            print("DONE in %ds" % (time.time() - t0))

            targets_spec[col_idx] = col_pearsonr

            print(" PearsonR %.4f" % np.nanmean(targets_spec[col_idx]), flush=True)

        # write target-level statistics (one row per experiment, plus-represented)
        targets_acc_df = pd.DataFrame(
            {
                "index": targets_strand_df.index,
                "pearsonr": targets_spec,
                "identifier": targets_strand_df.identifier,
                "description": targets_strand_df.description,
            }
        )
        targets_acc_df.to_csv(
            f"{args.out_dir}/acc.txt", sep="\t", index=False, float_format="%.5f"
        )
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
