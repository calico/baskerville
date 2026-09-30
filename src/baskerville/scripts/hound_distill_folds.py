#!/usr/bin/env python
# Copyright 2019 Calico Life Sciences LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import argparse
import importlib.util
import json
import os
import pdb
import shutil

from baskerville import utils

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None

from gcprunner.argparse_helpers import make_runner

"""
hound_distill_folds

Train student model replicates using knowledge distillation from teacher models.
Despite the name "folds", this script creates multiple replicates for compatibility
with other fold-based scripts. Each replicate is an independent distillation run.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Train multiple replicates of a student model via distillation."
    )
    parser.add_argument(
        "-c",
        "--crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds (replicates) [Default: %(default)s]",
    )
    parser.add_argument(
        "--checkpoint",
        default=False,
        action="store_true",
        help="Restart training from checkpoint [Default: %(default)s]",
    )
    parser.add_argument(
        "-e",
        "--conda_env",
        default=None,
        help="Conda environment to activate in each job (sources "
        "$BASKERVILLE_CONDA first if set). Default: the current environment.",
    )
    parser.add_argument(
        "-f",
        "--fold_subset",
        default=None,
        type=int,
        help="Run a subset of folds (replicates) [Default: %(default)s]",
    )
    parser.add_argument(
        "--name",
        default="distill",
        help="SLURM name prefix [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="distill_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        type=int,
        help="Number of processes, passed by multi script",
    )
    parser.add_argument(
        "-q",
        "--queue",
        default="rtx4090",
        help="SLURM queue on which to run the jobs [Default: %(default)s]",
    )
    parser.add_argument(
        "-r",
        "--restart",
        default=False,
        action="store_true",
        help="Restart training [Default: %(default)s]",
    )
    parser.add_argument(
        "--setup",
        default=False,
        action="store_true",
        help="Setup folds data directory only [Default: %(default)s]",
    )

    # Distillation arguments
    distill_args = parser.add_argument_group("distillation arguments")
    distill_args.add_argument(
        "--head",
        default=None,
        help="Teacher model head(s) to use for predictions [Default: %(default)s]",
    )
    distill_args.add_argument(
        "--rc",
        dest="teacher_rc",
        action="store_true",
        help="Average forward and reverse complement predictions from teachers [Default: %(default)s]",
    )
    distill_args.add_argument(
        "-s",
        "--teacher_subset",
        type=int,
        default=None,
        help="Randomly sample this many teachers for each prediction [Default: None (use all)]",
    )

    parser.add_argument("params_file", help="JSON file with student model parameters")
    parser.add_argument(
        "teachers_dir",
        help="Directory containing teacher models (e.g., from hound_train_folds)",
    )
    parser.add_argument(
        "data_dirs", nargs="+", help="Data directories containing sequences"
    )
    parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm" if importlib.util.find_spec("slurmrunner") else "local",
    )
    args = parser.parse_args()
    global slurmrunner
    slurmrunner = make_runner(args, slurm_module=slurmrunner)

    #######################################################
    # prep work

    if not args.restart and os.path.isdir(args.out_dir):
        raise ValueError(
            f"Output directory {args.out_dir} exists. Please remove or use --restart flag."
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_train = params["train"]

    # read data parameters
    num_data = len(args.data_dirs)
    data_stats_file = f"{args.data_dirs[0]}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)

    # count folds (replicates)
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # subset folds
    if args.fold_subset is not None:
        num_folds = min(args.fold_subset, num_folds)

    fold_index = [fold_i for fold_i in range(num_folds)]

    # arrange replicate output dirs (no per-rep data dirs; hound_train_distill
    # now resolves splits from --fold/--cross against the source data dir)
    for ci in range(args.crosses):
        for fi in fold_index:
            rep_dir = f"{args.out_dir}/f{fi}c{ci}"
            os.makedirs(rep_dir, exist_ok=True)

    if args.setup:
        return

    #######################################################
    # train

    jobs = []

    for ci in range(args.crosses):
        for fi in fold_index:
            rep_dir = f"{args.out_dir}/f{fi}c{ci}"

            train_dir = f"{rep_dir}/train"
            if args.restart and not args.checkpoint and os.path.isdir(train_dir):
                print(f"{rep_dir} found and skipped.")

            else:
                # copy params into output directory
                shutil.copy(args.params_file, f"{rep_dir}/params.json")

                # train command
                cmd = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"

                cmd += " hound_train_distill"
                cmd += f" -o {rep_dir}/train"
                cmd += f" --fold {fi} --cross {ci}"
                if args.teacher_rc:
                    cmd += " --rc"
                if args.teacher_subset is not None:
                    cmd += f" -s {args.teacher_subset}"
                if args.head is not None:
                    cmd += f" --head {args.head}"
                cmd += f" {rep_dir}/params.json"
                cmd += f" {args.teachers_dir}"
                cmd += f" {' '.join(args.data_dirs)}"

                if slurmrunner is None:
                    # Run locally
                    jobs.append(cmd)
                else:
                    # Submit to SLURM
                    name = f"{args.name}-f{fi}c{ci}"
                    sbf = os.path.abspath(f"{rep_dir}/train.sb")
                    outf = os.path.abspath(f"{rep_dir}/train.out")
                    errf = os.path.abspath(f"{rep_dir}/train.err")

                    j = slurmrunner.Job(
                        cmd,
                        name=name,
                        out_file=outf,
                        err_file=errf,
                        sb_file=sbf,
                        queue=args.queue,
                        cpu=8,
                        gpu=params_train.get("num_gpu", 1),
                        mem=30000,
                        time="60-0:0:0",
                    )
                    jobs.append(j)

    # Execute jobs
    if slurmrunner is None:
        print(
            "Running jobs locally..."
            if args.backend == "local"
            else "SLURM not available, running jobs locally..."
        )
        utils.exec_par(jobs, max_proc=args.processes, verbose=True)
    else:
        print("Submitting jobs to SLURM...")
        slurmrunner.multi_run(
            jobs,
            max_proc=args.processes,
            verbose=True,
            launch_sleep=10,
            update_sleep=60,
        )


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
