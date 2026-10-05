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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import shutil

import h5py
import numpy as np

try:
    import slurmrunner
except ImportError:
    print("slurmrunner not found, running locally")
    slurmrunner = None

from gcprunner import run_identity
from gcprunner.argparse_helpers import (
    add_argparse_group,
    announce_gcp_image,
    make_runner,
    resolve_image_arg,
)
from gcprunner.batch_spec import DataMount

from baskerville.multi import check_progress_h5
from baskerville import utils
from baskerville.helpers import gcp_output, stage_cache

"""
hound_ism_snp_folds

Perform ISM around SNP variants using an ensemble of cross-fold models.
"""

# Container mount point for a GCP-trained models tree read straight from GCS
# (marker mode). Distinct from the content-cache mount (/workspace/cache).
_GCP_MODELS_MOUNT = "/workspace/models"


################################################################################
# main
################################################################################
def main():
    parser = ArgumentParser(
        description="Perform ISM around SNP variants using cross-fold model ensemble.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # ISM options (pass-through to hound_ism_snp)
    ism_group = parser.add_argument_group("hound_ism_snp options")
    ism_group.add_argument(
        "-d",
        dest="mut_down",
        default=0,
        type=int,
        help="Nucleotides downstream of center sequence to mutate",
    )
    ism_group.add_argument(
        "-f",
        dest="genome_fasta",
        required=True,
        help="Genome FASTA",
    )
    ism_group.add_argument(
        "-g",
        dest="genes_gtf",
        default=None,
        help="GTF for gene annotations. Enables covgene/ and gene/ scoring.",
    )
    ism_group.add_argument(
        "--head",
        dest="head",
        default=0,
        type=int,
        help="Model head with which to predict.",
    )
    ism_group.add_argument(
        "-l",
        dest="mut_len",
        default=128,
        type=int,
        help="Length of centered sequence to mutate",
    )
    ism_group.add_argument(
        "-m",
        "--mix_dtype",
        dest="mix_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Mixed precision dtype",
    )
    ism_group.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile the model with torch.compile",
    )
    ism_group.add_argument(
        "-o",
        dest="out_dir",
        default="ism_snp_out",
        help="Output directory",
    )
    ism_group.add_argument(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions",
    )
    ism_group.add_argument(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts",
    )
    ism_group.add_argument(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save.",
    )
    ism_group.add_argument(
        "-t",
        dest="targets_file",
        required=True,
        type=str,
        help="File specifying target indexes and labels in table format",
    )
    ism_group.add_argument(
        "--targets_gene_file",
        dest="targets_gene_file",
        default=None,
        type=str,
        help="File specifying gene head target indexes and labels",
    )
    ism_group.add_argument(
        "-u",
        dest="mut_up",
        default=0,
        type=int,
        help="Nucleotides upstream of center sequence to mutate",
    )

    # cross-fold options
    fold_group = parser.add_argument_group("cross-fold options")
    fold_group.add_argument(
        "--cross",
        dest="crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds",
    )
    fold_group.add_argument(
        "-e",
        dest="conda_env",
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
        "--f_list",
        dest="fold_subset_list",
        default=None,
        help="Subset of folds to evaluate (encoded as comma-separated string)",
    )
    fold_group.add_argument(
        "--name", dest="name", default="ism_snp", help="SLURM name prefix"
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
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs",
    )

    # positional arguments
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("models_dir", help="Cross-fold models directory")
    parser.add_argument("vcf_file", help="VCF file")
    add_argparse_group(parser)
    args = parser.parse_args()

    ism_snp_folds(args)


def ism_snp_folds(args):
    """Execute cross-fold ISM scoring for SNP variants."""
    global slurmrunner

    gcp_backend = getattr(args, "backend", None) == "gcp"

    # GCP marker mode: if the local models_dir carries a gcp_run.json marker, the
    # models were trained on GCP and their weights live in GCS (the train mirror
    # excludes .pth). Read them straight from there and inherit the run's GCP
    # config, instead of staging a local models tree.
    gcs_models_dir = None
    if gcp_backend:
        # An explicit --gcp_branch/--gcp_image is resolved before the marker so it
        # overrides the training image; the 'main' default is applied after the
        # marker (below) so a training-recorded image wins over it.
        marker = stage_cache.read_run_marker(args.models_dir)
        resolve_image_arg(args, allow_default=False, marker=marker)
        if marker:
            gcs_models_dir = marker.get("output_dir_gcs")
            if not gcs_models_dir:
                raise ValueError(
                    f"{args.models_dir}/{stage_cache.RUN_MARKER_NAME} has no "
                    "'output_dir_gcs' field."
                )
            inherited = stage_cache.apply_run_marker(args, marker)
            print(
                f"[gcp] models from GCS {gcs_models_dir} "
                f"(marker {args.models_dir}/{stage_cache.RUN_MARKER_NAME}; no staging)"
            )
            if inherited:
                print(f"[gcp] inherited from marker: {', '.join(inherited)}")

        # Fall back to the latest built commit on 'main' only if neither an
        # explicit image/branch nor the marker supplied one.
        announce_gcp_image(args, marker)

    # count folds (GCS in marker mode, local models_dir otherwise)
    if args.num_folds is None:
        if gcs_models_dir is not None:
            args.num_folds = stage_cache.detect_model_folds_gcs(gcs_models_dir)
        else:
            args.num_folds = utils.detect_model_folds(args.models_dir)
        print(f"Found {args.num_folds} folds")
        if args.num_folds == 0:
            raise ValueError(f"No models found in {args.models_dir}")

    # subset folds
    if args.fold_subset_list is None:
        fold_index = list(range(args.num_folds))
    else:
        fold_index = [int(fold_str) for fold_str in args.fold_subset_list.split(",")]

    primary_out_dir = args.out_dir

    # GCP: stage local inputs to the content-addressed cache, rewrite paths to
    # their in-container counterparts, and auto-pick an output prefix. Must run
    # before make_runner() so the rewritten paths / gcp_output_dir are baked into
    # the gcprunner.Job kwargs. One job per fold writes scores.h5 directly (no
    # sharding/merge), so there's no per-job collect step.
    if gcp_backend:
        vcf_sha, args.vcf_file = stage_cache.stage_file(args.vcf_file, "vcf")
        models_sha = None
        if gcs_models_dir is None:
            models_sha, args.models_dir = stage_cache.stage_dir(
                args.models_dir, "models", filename="model_best.pth"
            )
        _, args.params_file = stage_cache.stage_file(args.params_file, "params")
        if args.genome_fasta:
            # pysam expects <fasta>.fai next to the FASTA; the cache mount is
            # read-only so it can't build the index in-place.
            fai = args.genome_fasta + ".fai"
            extras = [fai] if os.path.exists(fai) else []
            _, args.genome_fasta = stage_cache.stage_file(
                args.genome_fasta, "fasta", extra_files=extras
            )
        if args.targets_file:
            _, args.targets_file = stage_cache.stage_file(args.targets_file, "targets")
        if args.targets_gene_file:
            _, args.targets_gene_file = stage_cache.stage_file(
                args.targets_gene_file, "targets_gene"
            )
        if args.genes_gtf:
            _, args.genes_gtf = stage_cache.stage_file(args.genes_gtf, "gtf")
        if not args.gcp_output_dir:
            import hashlib

            models_key = (
                models_sha or hashlib.sha256(gcs_models_dir.encode()).hexdigest()
            )
            run_id = stage_cache.build_run_id(vcf_sha, models_key, deterministic=True)
            args.gcp_output_dir = f"{stage_cache.output_prefix()}/ism_snp/{run_id}"
        print("=" * 72)
        print(f"[gcp] run output dir: {args.gcp_output_dir}")
        print("=" * 72)

    if hasattr(args, "backend"):
        slurmrunner = make_runner(args, slurm_module=slurmrunner)

    ################################################################
    # ISM scoring jobs

    # GCP container jobs already have the env active and write under
    # /workspace/out (rsync'd to gcp_output_dir by entry.sh); the content cache
    # (and, in marker mode, the GCS models dir) are mounted read-only via
    # GCSFuse. Local/slurm jobs need conda activation and a $HOSTNAME echo.
    if gcp_backend:
        cmd_base = ""
        data_mounts = [
            DataMount(
                stage_cache.cache_prefix(),
                stage_cache.CONTAINER_CACHE_MOUNT,
                mode="fuse",
            )
        ]
        if gcs_models_dir is not None:
            data_mounts.append(
                DataMount(gcs_models_dir, _GCP_MODELS_MOUNT, mode="fuse")
            )
        gcp_extra = {
            "output_dir_gcs": args.gcp_output_dir,
            "data_mounts": data_mounts,
        }
        # Run-identity labels (per-fold gcprunner_fold added at each Job call) so
        # distinct ISM-SNP runs are distinguishable regardless of --name.
        run_id = run_identity.run_id_from_gcs_dir(args.gcp_output_dir)
    else:
        cmd_base = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"
        gcp_extra = {}

    # Root of the models tree as seen by the scoring command: the GCS mount in
    # marker mode, otherwise args.models_dir (the staged cache path on GCP, or
    # the local path on slurm/local).
    models_root = _GCP_MODELS_MOUNT if gcs_models_dir is not None else args.models_dir

    jobs = []

    for ci in range(args.crosses):
        for fi in fold_index:
            fold_cross = f"f{fi}c{ci}"
            name = f"{args.name}-{fold_cross}"

            model_file = f"{models_root}/{fold_cross}/train/model_best.pth"

            if gcp_backend:
                # In-container output must live under /workspace/out so entry.sh's
                # rsync uploads it to <gcp_output_dir>/<fold_cross>/scores.h5.
                fold_out_dir = f"/workspace/out/{fold_cross}"
                gcp_extra["labels"] = run_identity.identity_labels(
                    run_id, "ism_snp", fold_cross
                )
            else:
                fold_out_dir = f"{primary_out_dir}/{fold_cross}"
            if not gcp_backend:
                os.makedirs(fold_out_dir, exist_ok=True)

            scores_file = f"{fold_out_dir}/scores.h5"
            if gcp_backend:
                # Per-fold resume: skip folds whose scores.h5 already exists in
                # GCS and is marked completed (from a prior attempt).
                gcs_scores = f"{args.gcp_output_dir}/{fold_cross}/scores.h5"
                already_done = stage_cache.check_progress_h5_gcs(
                    gcs_scores, "completed"
                )
            else:
                already_done = check_progress_h5(scores_file, "completed")
            if not already_done:
                cmd_job = cmd_base if gcp_backend else f"{cmd_base} time "
                cmd_job += build_ism_snp_cmd(args, model_file, fold_out_dir)

                if slurmrunner is None:
                    jobs.append(cmd_job)
                else:
                    j = slurmrunner.Job(
                        cmd_job,
                        f"{name}",
                        f"{fold_out_dir}.out",
                        f"{fold_out_dir}.err",
                        f"{fold_out_dir}.sb",
                        queue=args.queue,
                        gpu=1,
                        cpu=4,
                        mem=30000,
                        time="7-0:0:0",
                        **gcp_extra,
                    )
                    jobs.append(j)

    if slurmrunner is None:
        utils.exec_par(jobs, args.parallel_jobs, verbose=True)
    else:
        slurmrunner.multi_run(
            jobs,
            max_proc=args.parallel_jobs,
            verbose=True,
            launch_sleep=10,
            update_sleep=60,
        )

    fold_crosses = [f"f{fi}c{ci}" for ci in range(args.crosses) for fi in fold_index]

    #######################################################
    # GCP: fetch per-fold scores back, then ensemble locally

    if gcp_backend:
        # Each fold's single job wrote scores.h5 (+ targets) directly — no merge.
        # Mirror them to a local dir so the ensemble step below runs unchanged.
        if args.gcp_fetch_output is None:
            local_target = primary_out_dir
        else:
            local_target = args.gcp_fetch_output  # empty string → skip
        if local_target:
            gcp_output.fetch_results(args.gcp_output_dir, local_target, fold_crosses)
            print(f"[fetch] results → {local_target}/")
        print(f"[gcs] full run at {args.gcp_output_dir}")
        if not local_target:
            # Without local files, the ensemble step has nothing to read.
            return
        # ensemble reads from wherever the fetch actually landed
        ensemble_base = local_target
    else:
        #######################################################
        # verify

        for fold_cross in fold_crosses:
            fold_out_dir = f"{primary_out_dir}/{fold_cross}"
            scores_file = f"{fold_out_dir}/scores.h5"
            if not check_progress_h5(scores_file, "completed"):
                raise RuntimeError(f"ISM SNP scoring job failed: {scores_file}")
        ensemble_base = primary_out_dir

    ################################################################
    # ensemble

    ensemble_dir = f"{ensemble_base}/ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)

    fold_dirs = [f"{ensemble_base}/{fold_cross}" for fold_cross in fold_crosses]

    ensemble_scores(ensemble_dir, fold_dirs)


def ensemble_scores(ensemble_dir, fold_dirs):
    """Ensemble ISM scores from multiple folds by averaging.

    Handles both flat datasets and hierarchical ref/alt groups.

    Args:
        ensemble_dir: Directory for ensemble output.
        fold_dirs: List of fold directories containing score HDF5 files.
    """
    # copy fold0 targets
    for targets_name in ["targets_cov.txt", "targets_covgene.txt", "targets_gene.txt"]:
        fold0_targets_file = f"{fold_dirs[0]}/{targets_name}"
        if os.path.exists(fold0_targets_file):
            shutil.copyfile(fold0_targets_file, f"{ensemble_dir}/{targets_name}")

    ensemble_h5_file = f"{ensemble_dir}/scores.h5"
    folds_h5_files = [f"{fold_dir}/scores.h5" for fold_dir in fold_dirs]
    ensemble_h5 = h5py.File(ensemble_h5_file, "w")

    # metadata keys to copy from fold 0 (not averaged)
    copy_keys = {
        "label",
        "chr",
        "start",
        "end",
        "ref/seqs",
        "alt/seqs",
        "gene_ids",
        "snp_idx",
        "gene_idx",
    }
    skip_keys = copy_keys | {"progress_status"}

    snp_stats = []
    stat_shapes = []
    scores0_h5 = h5py.File(folds_h5_files[0], "r")

    def _ensure_parents(name):
        """Create all parent groups for a hierarchical dataset path."""
        parts = name.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            if parent not in ensemble_h5:
                ensemble_h5.create_group(parent)

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name in copy_keys:
                _ensure_parents(name)
                ensemble_h5.create_dataset(name, data=obj)
            elif name not in skip_keys:
                snp_stats.append(name)
                stat_shapes.append(obj.shape)

    scores0_h5.visititems(_visit)
    scores0_h5.close()

    # average stats across folds
    num_folds = len(fold_dirs)
    for si, snp_stat in enumerate(snp_stats):
        scores = np.zeros(shape=stat_shapes[si], dtype="float32")
        for scores_file in folds_h5_files:
            with h5py.File(scores_file, "r") as scores_h5:
                scores += scores_h5[snp_stat][:].astype("float32")
        scores /= num_folds

        _ensure_parents(snp_stat)
        ensemble_h5.create_dataset(snp_stat, data=scores.astype("float16"))

    ensemble_h5.close()


def build_ism_snp_cmd(args, model_file, out_dir):
    """Build command string for hound_ism_snp."""
    cmd_parts = ["hound_ism_snp"]

    if args.mut_down != 0:
        cmd_parts.extend(["-d", str(args.mut_down)])

    if args.genome_fasta:
        cmd_parts.extend(["-f", args.genome_fasta])

    if args.genes_gtf:
        cmd_parts.extend(["-g", args.genes_gtf])

    if args.head != 0:
        cmd_parts.extend(["--head", str(args.head)])

    cmd_parts.extend(["-l", str(args.mut_len)])

    if args.mix_dtype != "float32":
        cmd_parts.extend(["-m", args.mix_dtype])

    if args.compile:
        cmd_parts.append("--compile")

    cmd_parts.extend(["-o", out_dir])

    if args.rc:
        cmd_parts.append("--rc")

    if args.shifts != "0":
        cmd_parts.extend(["--shifts", args.shifts])

    if args.snp_stats != "logSUM":
        cmd_parts.extend(["--stats", args.snp_stats])

    cmd_parts.extend(["-t", args.targets_file])

    if args.targets_gene_file:
        cmd_parts.extend(["--targets_gene_file", args.targets_gene_file])

    if args.mut_up != 0:
        cmd_parts.extend(["-u", str(args.mut_up)])

    # positional arguments
    cmd_parts.extend([args.params_file, model_file, args.vcf_file])

    return " ".join(cmd_parts)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
