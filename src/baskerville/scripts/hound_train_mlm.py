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

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer

"""
hound_train_mlm.py

Train Hound model for Masked Language Modeling using given parameters and data.
"""


def main():
    parser = argparse.ArgumentParser(description="Train an MLM model.")
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
        "--has_mask",
        action="store_true",
        help="Dataset has exon masks [Default: %(default)s]",
    )
    parser.add_argument(
        "--has_repeat_mask",
        action="store_true",
        help="Dataset has repeat masks [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != f"{args.out_dir}/params.json":
        shutil.copy(args.params_file, f"{args.out_dir}/params.json")

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)

    # override loss to mlm
    params["train"]["loss"] = "mlm"

    # get mask settings from params or command line
    has_mask = args.has_mask or params["train"].get("has_mask", False)
    has_repeat_mask = args.has_repeat_mask or params["train"].get(
        "has_repeat_mask", False
    )
    augment_rc = params["train"].get("augment_rc", True)

    # read datasets
    train_data = []
    eval_data = []
    for data_dir in args.data_dirs:
        if args.whole:
            train_data.append(
                dataset.SeqDatasetMLM(
                    data_dir,
                    split_label="*",
                    mode="train",
                    has_mask=has_mask,
                    has_repeat_mask=has_repeat_mask,
                    augment_rc=augment_rc,
                )
            )
        else:
            # load train data
            train_data.append(
                dataset.SeqDatasetMLM(
                    data_dir,
                    split_label="train",
                    mode="train",
                    has_mask=has_mask,
                    has_repeat_mask=has_repeat_mask,
                    augment_rc=augment_rc,
                )
            )

            # load eval data (no augmentation)
            eval_data.append(
                dataset.SeqDatasetMLM(
                    data_dir,
                    split_label="valid",
                    mode="eval",
                    has_mask=has_mask,
                    has_repeat_mask=has_repeat_mask,
                    augment_rc=False,
                )
            )

    ################################################################

    # initialize model
    seqnn_model = seqnn.SeqNN(params["model"])

    # initialize trainer
    seqnn_trainer = trainer.Trainer(
        params["train"], train_data, eval_data, args.out_dir
    )

    # train using MLM (loss="mlm" routes fit() through the MLM task path)
    seqnn_trainer.fit(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
