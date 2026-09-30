#!/usr/bin/env python

import argparse
import importlib.util
import os
from baskerville import utils

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None

from gcprunner.argparse_helpers import make_runner

"""
hound_grad_folds

Run hound_grad for baskerville model replicates on cross folds using given parameters and data.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run hound_grad for baskerville model replicates on cross folds."
    )

    # grad options
    grad_group = parser.add_argument_group("hound_grad options")
    grad_group.add_argument(
        "--bigwig", default=False, action="store_true", help="Output bigwig files"
    )
    grad_group.add_argument(
        "-f",
        dest="genome_fasta",
        required=True,
        help="Genome FASTA for sequences (required)",
    )
    grad_group.add_argument(
        "--log",
        dest="log_transform",
        action="store_true",
        help="Apply log transformation to sum of coverage",
    )
    grad_group.add_argument("--head", type=int, default=0, help="Model head index")
    grad_group.add_argument(
        "--rc", action="store_true", help="Add reverse complement augmentation"
    )
    grad_group.add_argument(
        "-t", "--targets_file", required=True, help="Targets table (required)"
    )

    # fold options
    fold_group = parser.add_argument_group("cross-fold options")
    fold_group.add_argument(
        "-c",
        "--crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds [Default: %(default)s]",
    )
    fold_group.add_argument(
        "-e",
        "--conda_env",
        default=None,
        help="Conda environment to activate in each job (sources "
        "$BASKERVILLE_CONDA first if set). Default: the current environment.",
    )
    fold_group.add_argument(
        "--folds",
        dest="num_folds",
        default=None,
        type=int,
        help="Number of folds to evaluate",
    )
    fold_group.add_argument(
        "--name", default="grad", help="SLURM name prefix [Default: %(default)s]"
    )
    fold_group.add_argument(
        "-o", "--out_dir", default="grad_out", help="Output directory"
    )
    fold_group.add_argument(
        "-p",
        dest="parallel_jobs",
        default=None,
        type=int,
        help="Maximum number of jobs to run in parallel",
    )
    fold_group.add_argument(
        "-q",
        "--queue",
        default="geforce",
        help="SLURM queue on which to run the jobs [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("models_dir", help="Directory containing trained models")
    parser.add_argument("gene_gtf", help="Gene GTF file")
    parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm" if importlib.util.find_spec("slurmrunner") else "local",
    )
    args = parser.parse_args()
    global slurmrunner
    slurmrunner = make_runner(args, slurm_module=slurmrunner)

    # count folds
    if args.num_folds is None:
        args.num_folds = utils.detect_model_folds(args.models_dir)
        print(f"Found {args.num_folds} folds")
        if args.num_folds == 0:
            raise ValueError(f"No models found in {args.models_dir}")

    if args.queue == "standard":
        num_cpu = 16
        num_gpu = 0
        time_base = 64
    else:
        num_cpu = 4
        num_gpu = 1
        time_base = 24

    ################################################################
    # gradient jobs

    # command base
    cmd_base = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"

    jobs = []

    for ci in range(args.crosses):
        for fi in range(args.num_folds):
            it_dir = f"{args.out_dir}/f{fi}c{ci}"
            os.makedirs(it_dir, exist_ok=True)
            model_file = f"{args.models_dir}/f{fi}c{ci}/train/model_best.pth"
            if os.path.isfile(model_file):
                out_dir = it_dir
                grad_file = f"{out_dir}/scores.h5"
                if os.path.isfile(grad_file):
                    print(f"{grad_file} already exists, skipping.")
                    continue

                cmd_job = f"{cmd_base} time "

                cmd_job += " hound_grad"
                if args.bigwig:
                    cmd_job += " --bigwig"
                cmd_job += f" -f {args.genome_fasta}"
                cmd_job += f" --head {args.head}"
                if args.log_transform:
                    cmd_job += " --log"
                cmd_job += f" -o {out_dir}"
                if args.rc:
                    cmd_job += " --rc"
                cmd_job += f" -t {args.targets_file}"
                cmd_job += f" {args.params_file}"
                cmd_job += f" {model_file}"
                cmd_job += f" {args.gene_gtf}"

                if slurmrunner is None:
                    jobs.append(cmd_job)
                else:
                    name = f"{args.name}-grad-f{fi}c{ci}"
                    job = slurmrunner.Job(
                        cmd_job,
                        name=name,
                        out_file=f"{out_dir}/grad.out",
                        err_file=f"{out_dir}/grad.err",
                        queue=args.queue,
                        cpu=num_cpu,
                        gpu=num_gpu,
                        mem=30000,
                        time=f"{time_base}:00:00",
                    )
                    jobs.append(job)

    # Execute jobs
    if slurmrunner is None:
        utils.exec_par(jobs, max_proc=args.parallel_jobs, verbose=True)
    else:
        slurmrunner.multi_run(
            jobs,
            max_proc=args.parallel_jobs,
            verbose=True,
            launch_sleep=5,
            update_sleep=60,
        )


if __name__ == "__main__":
    main()
