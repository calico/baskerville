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
import torch
from tqdm import tqdm

from baskerville import dataset
from baskerville import seqnn

"""
hound_updatenorm

Update batch normalization statistics of a trained model by running forward passes
on training data, then save the updated model.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Update batch normalization statistics of a trained model."
    )
    parser.add_argument(
        "-b",
        dest="num_batches",
        default=50,
        type=int,
        help="Number of batches to use for BN update (None = use all) [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Model head(s) to use. Can be: "
        "None (use dataset-specific heads, default), "
        "int (force specific head for all sequences), "
        "-1 (concatenate all heads). [Default: %(default)s]",
    )
    parser.add_argument(
        "-w",
        "--whole",
        action="store_true",
        help="Use whole dataset, without validation split [Default: %(default)s]",
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
        help="Cross replicate index [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file to update")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train data directorie(s) for BN update"
    )
    args = parser.parse_args()

    num_folds = dataset.discover_num_folds(args.data_dirs[0])
    if num_folds is not None and args.fold is None and not args.whole:
        parser.error(
            f"{args.data_dirs[0]}/examples/ contains fold*.zarr ({num_folds} folds); "
            "pass --fold or --whole."
        )
    if num_folds is None and args.fold is not None:
        parser.error(
            f"--fold given but {args.data_dirs[0]}/examples/ has no fold*.zarr."
        )

    # Read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    # Get batch size and num_workers from params
    batch_size = params["train"]["batch_size"]
    num_workers = params["train"].get("num_workers", 0)

    # Read datasets
    train_data = []
    for data_dir in args.data_dirs:
        if args.whole:
            train_data.append(
                dataset.SeqDataset(
                    data_dir,
                    split_label="*",
                    mode="train",
                    fold=args.fold,
                    cross=args.cross,
                )
            )
        else:
            # Load train data only
            train_data.append(
                dataset.SeqDataset(
                    data_dir,
                    split_label="train",
                    mode="train",
                    fold=args.fold,
                    cross=args.cross,
                )
            )

    ################################################################
    # Initialize model and restore weights

    # Initialize model
    seqnn_model = seqnn.SeqNN(params["model"])

    # Restore trained weights
    seqnn_model.restore(args.model_file)

    # Set mixed precision dtype if specified
    mix_dtype = params["train"].get("mix_dtype", "float32")
    if mix_dtype != "float32":
        if mix_dtype == "float16":
            seqnn_model.mix_dtype = torch.float16
        elif mix_dtype == "bfloat16":
            seqnn_model.mix_dtype = torch.bfloat16
        else:
            print(f"Warning: Unrecognized mixed precision dtype {mix_dtype}")

    ################################################################
    # Create data loader

    # Combine training datasets
    train_multidata = dataset.MultiDataset(train_data)

    # Create sampler
    train_sampler = dataset.MultiSampler(
        train_multidata,
        batch_size=batch_size,
        mode="train",
    )

    # Create data loader
    train_dataload = torch.utils.data.DataLoader(
        train_multidata,
        batch_sampler=train_sampler,
        num_workers=num_workers,
    )

    ################################################################
    # Update batch norm statistics

    print("Updating batch normalization statistics...")
    update_bn_stats(
        seqnn_model, train_dataload, model_heads=args.head, num_batches=args.num_batches
    )
    print("Batch normalization statistics updated.")

    ################################################################
    # Save updated model

    # Backup original model by replacing .pth with _orig.pth
    if not args.model_file.endswith(".pth"):
        raise ValueError("Model file must have .pth extension to create backup.")

    backup_file = args.model_file[:-4] + "_orig.pth"
    if os.path.exists(backup_file):
        raise FileExistsError(
            f"Backup file {backup_file} already exists. Clarify before proceeding."
        )

    shutil.copy(args.model_file, backup_file)
    print(f"Original model backed up to {backup_file}")

    torch.save(seqnn_model.model.state_dict(), args.model_file)
    print(f"Updated model saved to {args.model_file}")


def update_bn_stats(seqnn_model, data_loader, model_heads=None, num_batches=None):
    """Update batch normalization statistics.

    Args:
        seqnn_model: SeqNN model object
        data_loader: DataLoader with training data
        model_heads: Model head index to use (None = use dataset index)
        num_batches: Number of batches to use (None = use all)
    """
    # Set model to train mode to update BN running stats
    seqnn_model.model.train()

    # Disable gradient computation
    with torch.no_grad():
        batch_count = 0
        total = num_batches if num_batches is not None else len(data_loader)
        for batch in tqdm(data_loader, total=total, desc="Updating BN statistics"):
            di, example = batch
            di = di[0]
            x, y = example
            x = x.to(seqnn_model.device)

            # Define model head
            if model_heads is None:
                hi = di
            else:
                hi = model_heads

            # Forward pass to update BN statistics
            with torch.autocast(
                device_type=seqnn_model.device, dtype=seqnn_model.mix_dtype
            ):
                seqnn_model.model(x, hi)

            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break

    # Set model back to eval mode
    seqnn_model.model.eval()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
