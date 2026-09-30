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
import json
import os
import shutil
import torch  # needed due to numpy interaction

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer
from baskerville.helpers import train_gcs

"""
hound_train

Train Hound model using given parameters and data.
"""


def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-w",
        "--whole",
        action="store_true",
        help="Use whole dataset for training, without validation [Default: %(default)s]",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Cross-fold index to hold out as test. Required when data dir contains "
        "examples/fold*.zarr [Default: %(default)s]",
    )
    parser.add_argument(
        "--cross",
        type=int,
        default=0,
        help="Cross replicate index; valid fold = (fold + 1 + cross) %% num_folds "
        "[Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    args = parser.parse_args()

    # detect folds vs legacy split layout from the first data dir
    num_folds = dataset.discover_num_folds(args.data_dirs[0])
    if num_folds is not None and args.fold is None:
        parser.error(
            f"{args.data_dirs[0]}/examples/ contains fold*.zarr ({num_folds} folds); "
            "--fold is required."
        )
    if num_folds is None and args.fold is not None:
        parser.error(
            f"--fold given but {args.data_dirs[0]}/examples/ has no fold*.zarr; "
            "drop --fold or use a folds-mode data directory."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != f"{args.out_dir}/params.json":
        shutil.copy(args.params_file, f"{args.out_dir}/params.json")

    # improves memory usage for different dataset sizes
    if len(args.data_dirs) > 1:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    # optional data cropping
    extra_crop_bp = params["train"].get("extra_crop_bp", None)
    extra_crop_bins = params["train"].get("extra_crop_bins", None)

    # log fold assignment when in folds mode
    if args.fold is not None:
        splits = dataset.compute_fold_splits(num_folds, args.fold, args.cross)
        folds_log = {
            "fold": args.fold,
            "cross": args.cross,
            "num_folds": num_folds,
            "test_fold": splits["test"],
            "valid_fold": splits["valid"],
            "train_folds": splits["train"],
            "data_dirs": [os.path.abspath(d) for d in args.data_dirs],
        }
        print(f"[folds] {json.dumps(folds_log)}")
        with open(f"{args.out_dir}/folds.json", "w") as f:
            json.dump(folds_log, f, indent=2)

    # read datasets
    train_data = []
    eval_data = []
    for data_dir in args.data_dirs:
        if args.whole:
            train_data.append(
                dataset.SeqDataset(
                    data_dir,
                    split_label="*",
                    mode="train",
                    extra_crop_bp=extra_crop_bp,
                    extra_crop_bins=extra_crop_bins,
                    fold=args.fold,
                    cross=args.cross,
                )
            )
        else:
            # load train data
            train_data.append(
                dataset.SeqDataset(
                    data_dir,
                    split_label="train",
                    mode="train",
                    extra_crop_bp=extra_crop_bp,
                    extra_crop_bins=extra_crop_bins,
                    fold=args.fold,
                    cross=args.cross,
                )
            )

            # load eval data
            eval_data.append(
                dataset.SeqDataset(
                    data_dir,
                    split_label="valid",
                    mode="eval",
                    extra_crop_bp=extra_crop_bp,
                    extra_crop_bins=extra_crop_bins,
                    fold=args.fold,
                    cross=args.cross,
                )
            )

    ################################################################

    # GCP backend: restore any prior checkpoint from GCS into out_dir (so the
    # trainer's auto-resume picks it up) and sync out_dir to GCS each epoch.
    # Inert off-GCP (env vars unset).
    checkpoint_callback = None
    prune_callback = None
    gcs_dest = train_gcs.resolve_gcs_dest(args.out_dir)
    if gcs_dest is not None:
        if train_gcs.restore_outdir(args.out_dir, gcs_dest):
            print(f"[gcs] restored prior checkpoint from {gcs_dest}")
        else:
            print(f"[gcs] no prior checkpoint at {gcs_dest}; starting fresh")
        checkpoint_callback = train_gcs.make_sync_callback(args.out_dir, gcs_dest)
        prune_callback = train_gcs.make_prune_callback(gcs_dest)

    # initialize model
    seqnn_model = seqnn.SeqNN(params["model"])

    # initialize trainer
    seqnn_trainer = trainer.Trainer(
        params["train"],
        train_data,
        eval_data,
        args.out_dir,
        checkpoint_callback=checkpoint_callback,
        prune_callback=prune_callback,
    )

    # train
    seqnn_trainer.fit(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
