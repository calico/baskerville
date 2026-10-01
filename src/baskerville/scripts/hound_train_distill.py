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
import glob
import json
import os
import shutil

from natsort import natsorted
import torch

import numpy as np
import pandas as pd

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer
from baskerville.hardware import check_mixed_precision

"""
hound_train_distill

Train a student model using on-the-fly predictions from teacher models (knowledge distillation).

This script differs from hound_train by:
- Loading teacher models from a models directory (e.g., created by hound_train_folds)
- Using SeqDatasetTeacher to compute predictions on-the-fly during training
- Supporting teacher_subset and snp_rate for efficient and robust training
- Not requiring pre-stored target values (y) in the data directory

The teacher models compute soft targets dynamically during training, which provides:
- Memory efficiency (no pre-stored predictions)
- Dynamic augmentation (mutations applied to sequences before teacher prediction)
- Flexible teacher ensemble (can subset teachers per batch)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Train a model using on-the-fly teacher predictions."
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        dest="teacher_rc",
        action="store_true",
        help="Average forward and reverse complement predictions from teachers [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--teacher_subset",
        type=int,
        default=None,
        help="Randomly sample this many teachers for each prediction. "
        "E.g., with 8 teachers, use 2 for each batch. [Default: None (use all)]",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Teacher model head(s) to use for predictions. Can be: "
        "None (use dataset-specific heads, default), "
        "int (force specific head for all sequences), "
        "-1 (concatenate all heads). [Default: %(default)s]",
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
    parser.add_argument("params_file", help="JSON file with student model parameters")
    parser.add_argument(
        "teachers_dir",
        help="Directory containing teacher models (e.g., from hound_train_folds). "
        "Expected structure: teachers_dir/f*c*/train/model_best.pth",
    )
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="Data directories containing sequences (targets not required)",
    )
    args = parser.parse_args()

    num_folds = dataset.discover_num_folds(args.data_dirs[0])
    if num_folds is not None and args.fold is None:
        parser.error(
            f"{args.data_dirs[0]}/examples/ contains fold*.zarr ({num_folds} folds); "
            "--fold is required."
        )
    if num_folds is None and args.fold is not None:
        parser.error(
            f"--fold given but {args.data_dirs[0]}/examples/ has no fold*.zarr."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != f"{args.out_dir}/params.json":
        shutil.copy(args.params_file, f"{args.out_dir}/params.json")

    # read student model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    ################################################################
    # Load teacher models
    teachers = load_teacher_models(
        args.teachers_dir,
        data_dirs=args.data_dirs,
        ensemble_rc=args.teacher_rc,
    )

    ################################################################
    # Create datasets with teacher predictions

    train_data = []
    for data_dir in args.data_dirs:
        train_data.append(
            dataset.SeqDatasetTeacher(
                data_dir,
                split_label="*",
                teachers=teachers,
                mode="train",
                teacher_subset=args.teacher_subset,
                snp_rate=params["train"].get("snp_rate", 0.0),
                del_rate=params["train"].get("del_rate", 0.0),
                head=args.head,
                fold=args.fold,
                cross=args.cross,
            )
        )

    ################################################################
    # Train student model

    # initialize student model
    seqnn_model = seqnn.SeqNN(params["model"])

    # initialize trainer
    seqnn_trainer = trainer.Trainer(
        params["train"], train_data, [], args.out_dir, model_heads=args.head
    )

    # train
    seqnn_trainer.fit(seqnn_model)


def load_teacher_models(teachers_dir, data_dirs=None, ensemble_rc=False):
    """Load teacher models from a directory structure created by hound_train_folds.

    Discovers model files matching the pattern <teachers_dir>/*/train/model_best.pth
    and loads them with shared parameters. All models are assumed to have the same
    architecture and target definitions.

    Args:
        teachers_dir (str): Directory containing teacher model subdirectories.
            Expected structure: teachers_dir/f*c*/train/model_best.pth
            Parameters are loaded from teachers_dir/f0c0/params.json
        data_dirs (list, optional): List of data directories, one per head, to load
            strand pairing information from their targets.txt files.
        ensemble_rc (bool): If True, enables reverse-complement averaging for predictions.

    Returns:
        list[SeqNN]: List of loaded and initialized SeqNN models, ready for inference.
            All models are set to evaluation mode and moved to available GPU if present.

    Raises:
        ValueError: If no model files found or required params.json is missing.
    """
    # Find all model directories
    model_pattern = os.path.join(teachers_dir, "*/train/model_best.pth")
    model_files = natsorted(glob.glob(model_pattern))
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")
    print(f"Found {len(model_files)} teacher models:")

    # Load shared parameters from f0c0 directory
    params_file = os.path.join(teachers_dir, "f0c0", "params.json")
    if not os.path.exists(params_file):
        raise ValueError(f"Params file not found at {params_file}")

    with open(params_file) as f:
        params = json.load(f)
    params_model = params["model"].copy()
    params_train = params["train"].copy()

    # Load strand pairs for each head from each data directory
    strand_pairs = []
    for data_dir in data_dirs:
        data_targets_file = os.path.join(data_dir, "targets.txt")
        targets_df = pd.read_csv(data_targets_file, sep="\t", index_col=0)
        if "strand_pair" in targets_df.columns:
            strand_pairs.append(targets_df.strand_pair.values)
        else:
            strand_pairs.append(None)
    params_model["strand_pair"] = strand_pairs

    # Set mixed precision
    mix_dtype = params_train.get("mix_dtype", "float32")
    if mix_dtype != "float32":
        if not check_mixed_precision():
            print("Warning: Mixed precision not supported on this GPU, using float32")
            mix_dtype = "float32"
        elif mix_dtype not in ["float16", "bfloat16"]:
            print(
                f"Warning: Unrecognized mixed precision dtype {mix_dtype}, using float32"
            )
            mix_dtype = "float32"
    if mix_dtype != "float32":
        print(f"Using mixed precision: {mix_dtype}")

    # Load all teacher models
    teachers = []
    print("\nLoading teacher models:")
    for i, model_file in enumerate(model_files):
        # Initialize model with shared parameters
        teacher = seqnn.SeqNN(params_model)
        teacher.restore(model_file)
        teacher.ensemble_rc = ensemble_rc
        teacher.model.eval()

        # Set mixed precision
        if mix_dtype == "float16":
            teacher.mix_dtype = torch.float16
        elif mix_dtype == "bfloat16":
            teacher.mix_dtype = torch.bfloat16

        # compile
        if params_model.get("compile", False):
            teacher.model = torch.compile(teacher.model)

        teachers.append(teacher)

    if not teachers:
        raise ValueError("No valid models could be loaded")

    return teachers


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
