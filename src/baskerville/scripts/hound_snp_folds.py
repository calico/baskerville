#!/usr/bin/env python
# Copyright 2023 Calico Life Sciences LLC

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
from gcprunner.argparse_helpers import add_argparse_group, make_runner, resolve_project

from baskerville.multi import collect_scores, check_progress_h5
from baskerville import utils
from baskerville.helpers import gcp_output, stage_cache
from baskerville.vcf import VCF

"""
hound_snp_folds

Compute variant effect predictions for SNPs in a VCF file, using an ensemble of cross-fold models.

This script processes SNPs in parallel across multiple jobs and model folds, enabling efficient
large-scale variant effect prediction. It supports optional normalization using previously scored
datasets to calibrate quantiles when working with small variant sets.

Key features:
- Cross-fold ensemble prediction for robust variant effect estimates
- Parallel job processing for scalability
- Optional normalization against large reference datasets
- Support for both gene-specific and genome-wide scoring modes
"""


################################################################################
# main
################################################################################
def main():
    parser = ArgumentParser(
        description="Compute variant effect predictions for SNPs in a VCF file using cross-fold model ensemble.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # snp options
    snp_group = parser.add_argument_group("hound_snp options")
    snp_group.add_argument(
        "-c",
        dest="cluster_pct",
        default=0,
        type=float,
        help="Cluster SNPs (or genes) within a %% of the seq length to make a single ref pred",
    )
    snp_group.add_argument(
        "-f",
        dest="genome_fasta",
        required=True,
        help="Genome FASTA for sequences",
    )
    snp_group.add_argument(
        "-g",
        dest="genes_gtf",
        default=None,
        help="GTF for gene annotations. Enables gene scoring (coverage + gene head logFC).",
    )
    snp_group.add_argument(
        "--center_gene",
        dest="center_gene",
        default=False,
        action="store_true",
        help="Center sequences on genes instead of variants (requires -g)",
    )
    snp_group.add_argument(
        "--head",
        dest="head",
        default=0,
        type=int,
        help="Model head with which to predict.",
    )
    snp_group.add_argument(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts",
    )
    snp_group.add_argument(
        "--local_window",
        default=2048,
        type=int,
        help="Local window size in bp for targets with window=local",
    )
    snp_group.add_argument(
        "-m",
        "--mix_dtype",
        dest="mix_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Mixed precision dtype",
    )
    snp_group.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile the model with torch.compile",
    )
    snp_group.add_argument(
        "-n",
        "--norm",
        dest="norm_subdir",
        default=None,
        help="Model directory subdirectory containing normalization HDF5 files for each fold",
    )
    snp_group.add_argument(
        "-o",
        dest="out_dir",
        default="snp_out",
        help="Output directory for tables and plots",
    )
    parser.add_argument(
        "--pregrouped_seqs",
        default=False,
        action="store_true",
        help="Use pre-grouped sequences for scoring, defined in the VCF",
    )
    snp_group.add_argument(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions",
    )
    snp_group.add_argument(
        "--shifts",
        dest="shifts",
        default="0",
        type=str,
        help="Ensemble prediction shifts",
    )
    snp_group.add_argument(
        "--gene_cov_t",
        type=float,
        default=0.5,
        help="Minimum gene coverage fraction to include [Default: %(default)s]",
    )
    snp_group.add_argument(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="In gene scoring mode, aggregate entire gene span",
    )
    snp_group.add_argument(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save.",
    )
    snp_group.add_argument(
        "-t",
        dest="targets_file",
        required=True,
        help="File specifying target indexes and labels in table format",
    )
    snp_group.add_argument(
        "--targets_gene",
        dest="targets_gene_file",
        default=None,
        type=str,
        help="Gene targets file (for gene/ scores). Required when using gene/ stats.",
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
        "--embed",
        default=False,
        action="store_true",
        help="Embed output in the models directory",
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
        "--name", dest="name", default="snp", help="SLURM name prefix"
    )
    fold_group.add_argument(
        "-p",
        dest="parallel_jobs",
        default=None,
        type=int,
        help="Maximum number of jobs to run in parallel",
    )
    fold_group.add_argument(
        "-j",
        dest="job_size",
        default=128,
        type=int,
        help="Number of SNPs to process per job",
    )
    fold_group.add_argument(
        "-q",
        dest="queue",
        default="geforce",
        help="SLURM queue on which to run the jobs",
    )

    # Positional arguments
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("models_dir", help="Cross-fold models directory")
    parser.add_argument("vcf_file", help="VCF file with SNPs to score")
    add_argparse_group(parser)
    args = parser.parse_args()

    # modularize for use in other scripts
    snp_folds(args)


def snp_folds(args):
    """Execute cross-fold SNP scoring with optional normalization.

    This function orchestrates the entire cross-fold scoring pipeline:
    1. Discovers available model folds and creates parallel jobs
    2. Runs SNP scoring jobs across all folds with fold-specific normalization
    3. Verifies job completion and collects results
    4. Creates ensemble predictions from all folds

    Args:
        args: Parsed command line arguments containing:
            - Model and data file paths
            - Job parallelization settings
            - Optional normalization subdirectory for quantile calibration
            - Cross-fold ensemble parameters
    """
    global slurmrunner

    #######################################################
    # prep work

    # validate: gene-prefixed stats require gene mode
    gene_prefixed = [
        s
        for s in args.snp_stats.split(",")
        if s.startswith("covgene/") or s.startswith("gene/")
    ]
    if gene_prefixed and not getattr(args, "genes_gtf", None):
        raise ValueError(
            f"Stats {gene_prefixed} require gene scoring mode (-g <gtf_file>)."
        )

    # count folds (uses local models_dir)
    if args.num_folds is None:
        args.num_folds = utils.detect_model_folds(args.models_dir)
        print(f"Found {args.num_folds} folds")
        if args.num_folds == 0:
            raise ValueError(f"No models found in {args.models_dir}")

    # subset folds
    if args.fold_subset_list is None:
        fold_index = [fold_i for fold_i in range(args.num_folds)]
    else:
        fold_index = [int(fold_str) for fold_str in args.fold_subset_list.split(",")]

    # save primary output directory
    primary_out_dir = args.out_dir

    # count SNPs and determine number of jobs (uses local vcf_file)
    vcf_obj = VCF(args.vcf_file)
    num_snps = len(vcf_obj.snps)
    num_jobs = (num_snps + args.job_size - 1) // args.job_size

    # create job bounds
    job_bounds = []
    for job_i in range(num_jobs):
        start_i = job_i * args.job_size
        end_i = min((job_i + 1) * args.job_size, num_snps)
        job_bounds.append((start_i, end_i))

    print(f"Processing {num_snps} SNPs in {num_jobs} jobs of size ~{args.job_size}")

    # GCP: stage local inputs to the content-addressed cache, rewrite paths
    # to their in-container counterparts, and auto-pick an output prefix.
    # Must run before make_runner() so the rewritten gcp_data_dir /
    # gcp_output_dir are baked into the gcprunner.Job kwargs.
    if getattr(args, "backend", None) == "gcp":
        # Fail before uploading anything if the GCP settings are missing.
        try:
            resolve_project(args.gcp_project)
            stage_cache.cache_prefix()
            if not args.gcp_output_dir:
                stage_cache.output_prefix()
        except (ValueError, RuntimeError) as e:
            parser.error(str(e))
        if args.embed:
            raise ValueError(
                "--embed is incompatible with --backend gcp: the staged models "
                "directory is mounted read-only inside the container."
            )
        vcf_sha, args.vcf_file = stage_cache.stage_file(args.vcf_file, "vcf")
        models_sha, args.models_dir = stage_cache.stage_dir(
            args.models_dir, "models", filename="model_best.pth"
        )
        _, args.params_file = stage_cache.stage_file(args.params_file, "params")
        if getattr(args, "genome_fasta", None):
            # pysam expects <fasta>.fai next to the FASTA; the cache mount is
            # read-only so it can't build the index in-place.
            fai = args.genome_fasta + ".fai"
            extras = [fai] if os.path.exists(fai) else []
            _, args.genome_fasta = stage_cache.stage_file(
                args.genome_fasta, "fasta", extra_files=extras
            )
        if getattr(args, "targets_file", None):
            _, args.targets_file = stage_cache.stage_file(args.targets_file, "targets")
        if getattr(args, "genes_gtf", None):
            _, args.genes_gtf = stage_cache.stage_file(args.genes_gtf, "gtf")
        if not args.gcp_output_dir:
            # Deterministic run_id so a re-run with the same VCF, models and job
            # command (options, job size) lands in the same output prefix, enabling
            # per-shard resume; any change starts fresh instead of reusing shards.
            cmd_sha = stage_cache.hash_text(
                build_snp_cmd(args, "MODEL", *job_bounds[0], "OUT", "FOLD")
            )
            run_id = stage_cache.build_run_id(
                vcf_sha, models_sha, cmd_sha, deterministic=True
            )
            args.gcp_output_dir = f"{stage_cache.output_prefix()}/snp/{run_id}"
        args.gcp_data_dir = stage_cache.cache_prefix()
        args.gcp_data_local = stage_cache.CONTAINER_CACHE_MOUNT
        # Run-identity labels (per-fold gcprunner_fold added at each Job call) so
        # distinct SNP-scoring runs are distinguishable regardless of --name.
        run_id = run_identity.run_id_from_gcs_dir(args.gcp_output_dir)
        print(f"[stage] inputs cached; output → {args.gcp_output_dir}")

    if hasattr(args, "backend"):
        slurmrunner = make_runner(args, slurm_module=slurmrunner)

    ################################################################
    # SNP scoring jobs

    # command base — local/slurm jobs need conda activation and hostname
    # logging; GCP container jobs already have the env active and don't
    # benefit from $HOSTNAME or shell-level timing. (GPU passthrough setup
    # lives in entry.sh so it's shared across all fold scripts.)
    gcp_backend = getattr(args, "backend", None) == "gcp"
    if gcp_backend:
        cmd_base = ""
    else:
        cmd_base = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"

    jobs = []

    for ci in range(args.crosses):
        for fi in fold_index:
            fold_cross = f"f{fi}c{ci}"
            name = f"{args.name}-{fold_cross}"

            # choose model
            fold_dir = f"{args.models_dir}/{fold_cross}"
            model_file = f"{fold_dir}/train/model_best.pth"

            # make jobs
            for job_i, (start_i, end_i) in enumerate(job_bounds):
                if gcp_backend:
                    # In-container output must live under /workspace/out so
                    # entry.sh's rsync uploads it. We deliberately drop the
                    # primary_out_dir prefix here so the GCS layout matches
                    # what finalize_fold expects: <gcp_output_dir>/<fold>/
                    # job<N>/scores.h5.
                    job_out_dir = f"/workspace/out/{fold_cross}/job{job_i}"
                elif args.embed:
                    job_out_dir = (
                        f"{args.models_dir}/{fold_cross}/{primary_out_dir}/job{job_i}"
                    )
                else:
                    job_out_dir = f"{primary_out_dir}/{fold_cross}/job{job_i}"
                if not gcp_backend:
                    os.makedirs(job_out_dir, exist_ok=True)

                scores_file = f"{job_out_dir}/scores.h5"
                if gcp_backend:
                    # Per-shard resume: skip shards whose scores.h5 already
                    # exists in GCS and is marked completed (from a prior
                    # attempt where some shards finished and others didn't).
                    gcs_scores = (
                        f"{args.gcp_output_dir}/{fold_cross}/job{job_i}/scores.h5"
                    )
                    already_done = stage_cache.check_progress_h5_gcs(
                        gcs_scores, "completed"
                    )
                else:
                    already_done = check_progress_h5(scores_file, "completed")
                if not already_done:
                    # create command with explicit start/end indices
                    cmd_job = cmd_base if gcp_backend else f"{cmd_base} time "
                    cmd_job += build_snp_cmd(
                        args,
                        model_file,
                        start_i,
                        end_i,
                        job_out_dir,
                        fold_cross,
                    )

                    if slurmrunner is None:
                        jobs.append(cmd_job)
                    else:
                        gcp_extra = (
                            {
                                "labels": run_identity.identity_labels(
                                    run_id, "snp", fold_cross
                                )
                            }
                            if gcp_backend
                            else {}
                        )
                        j = slurmrunner.Job(
                            cmd_job,
                            f"{name}_job{job_i}",
                            f"{job_out_dir}.out",
                            f"{job_out_dir}.err",
                            f"{job_out_dir}.sb",
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
    quantile_copy = args.norm_subdir is not None

    if getattr(args, "backend", None) == "gcp":
        # Per-job scores.h5 files live in GCS at <gcp_output_dir>/<fold>/jobN/.
        # Merge each fold (download shards → collect_scores → upload merged),
        # writing the merged per-fold files straight into the local mirror in the
        # same pass so the ensemble step below can run unchanged — no redundant
        # re-download of what we just uploaded.
        if args.gcp_fetch_output is None:
            local_target = primary_out_dir
        else:
            local_target = args.gcp_fetch_output  # empty string → skip
        for fold_cross in fold_crosses:
            print(f"[finalize] merging {fold_cross}")
            local_fold_dir = (
                os.path.join(local_target, fold_cross) if local_target else None
            )
            gcp_output.finalize_fold(
                f"{args.gcp_output_dir}/{fold_cross}",
                len(job_bounds),
                quantile_copy,
                local_fold_dir,
            )
        if local_target:
            print(f"[fetch] merged results → {local_target}/")
        print(f"[gcs] full run (incl. intermediates) at {args.gcp_output_dir}")
        if not local_target:
            # Without local files, the ensemble step has nothing to read.
            return
        # ensemble reads from wherever the fetch actually landed
        ensemble_base = local_target
    else:
        #######################################################
        # verify

        for fold_cross in fold_crosses:
            for job_i in range(len(job_bounds)):
                if args.embed:
                    job_out_dir = (
                        f"{args.models_dir}/{fold_cross}/{primary_out_dir}/job{job_i}"
                    )
                else:
                    job_out_dir = f"{primary_out_dir}/{fold_cross}/job{job_i}"
                scores_file = f"{job_out_dir}/scores.h5"
                if not check_progress_h5(scores_file, "completed"):
                    raise RuntimeError(f"SNP scoring job failed: {scores_file}")

        #######################################################
        # collect output

        for fold_cross in fold_crosses:
            if args.embed:
                fold_out_dir = f"{args.models_dir}/{fold_cross}/{primary_out_dir}"
            else:
                fold_out_dir = f"{primary_out_dir}/{fold_cross}"
            collect_scores(fold_out_dir, len(job_bounds), quantile_copy)
        ensemble_base = primary_out_dir

    ################################################################
    # ensemble

    if args.embed:
        ensemble_dir = f"{args.models_dir}/ensemble/{primary_out_dir}"
    else:
        ensemble_dir = f"{ensemble_base}/ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)

    # collect all fold score files
    fold_dirs = []
    for fold_cross in fold_crosses:
        if args.embed:
            fold_out_dir = f"{args.models_dir}/{fold_cross}/{primary_out_dir}"
        else:
            fold_out_dir = f"{ensemble_base}/{fold_cross}"
        fold_dirs.append(fold_out_dir)

    # create final ensemble
    ensemble_scores(ensemble_dir, fold_dirs)


def ensemble_scores(ensemble_dir: str, fold_dirs):
    """Ensemble SNP scores from multiple files into a single file.

    Args:
      ensemble_dir (str): Directory for ensemble output.
      fold_dirs ([str]): List of fold directories containing score HDF5 files.
    """
    # copy fold0 targets
    for targets_name in ["targets_cov.txt", "targets_covgene.txt", "targets_gene.txt"]:
        fold0_targets_file = f"{fold_dirs[0]}/{targets_name}"
        if os.path.exists(fold0_targets_file):
            shutil.copyfile(fold0_targets_file, f"{ensemble_dir}/{targets_name}")

    # open ensemble
    ensemble_h5_file = f"{ensemble_dir}/scores.h5"
    folds_h5_files = [f"{fold_dir}/scores.h5" for fold_dir in fold_dirs]
    ensemble_h5 = h5py.File(ensemble_h5_file, "w")

    # transfer base
    base_keys = [
        "alt_allele",
        "chr",
        "pos",
        "ref_allele",
        "snp",
    ]

    # keys that are the same across folds (gene-specific mode)
    gene_mode_keys = [
        "gene_ids",
        "snp_idx",
        "gene_idx",
    ]

    copy_keys = set(base_keys) | set(gene_mode_keys) | {"quantiles"}
    skip_keys = copy_keys | {"progress_status"}
    snp_stats = []
    sad_shapes = []
    scores0_h5 = h5py.File(folds_h5_files[0], "r")

    # copy metadata and discover stat datasets
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name in copy_keys:
                ensemble_h5.create_dataset(name, data=obj)
            elif name not in skip_keys:
                snp_stats.append(name)
                sad_shapes.append(obj.shape)

    scores0_h5.visititems(_visit)
    scores0_h5.close()

    # average stats across folds
    num_folds = len(fold_dirs)
    for si, snp_stat in enumerate(snp_stats):
        # initialize ensemble array
        snp_scores = np.zeros(shape=sad_shapes[si], dtype="float32")

        # read and add folds
        for scores_file in folds_h5_files:
            with h5py.File(scores_file, "r") as scores_h5:
                snp_scores += scores_h5[snp_stat][:].astype("float32")

        # normalize and downcast
        snp_scores /= num_folds
        snp_scores = snp_scores.astype("float16")

        # save (create parent groups if needed)
        ensemble_h5.create_dataset(snp_stat, data=snp_scores)

    ensemble_h5.close()


def build_snp_cmd(args, model_file, start_i, end_i, out_dir=None, fold_cross=None):
    """Build command string for hound_snp with explicit options.

    Args:
        args: Parsed command line arguments containing options
        model_file: Path to model file
        start_i: Start index for SNP processing (0-based, inclusive)
        end_i: End index for SNP processing (0-based, exclusive)
        out_dir: Output directory for this job
        fold_cross: Fold/cross identifier (e.g., "f0c0") for normalization file selection

    Returns:
        str: Complete command string for hound_snp
    """
    cmd_parts = ["hound_snp"]

    # Add options as command line flags
    if hasattr(args, "cluster_pct") and args.cluster_pct != 0:
        cmd_parts.extend(["-c", str(args.cluster_pct)])

    if hasattr(args, "genome_fasta") and args.genome_fasta:
        cmd_parts.extend(["-f", args.genome_fasta])

    if hasattr(args, "genes_gtf") and args.genes_gtf:
        cmd_parts.extend(["-g", args.genes_gtf])

    if hasattr(args, "center_gene") and args.center_gene:
        cmd_parts.append("--center_gene")

    if hasattr(args, "indel_stitch") and args.indel_stitch:
        cmd_parts.append("--indel_stitch")

    if hasattr(args, "local_window") and args.local_window != 2048:
        cmd_parts.extend(["--local_window", str(args.local_window)])

    if hasattr(args, "mix_dtype") and args.mix_dtype != "float32":
        cmd_parts.extend(["-m", args.mix_dtype])

    if hasattr(args, "compile") and args.compile:
        cmd_parts.append("--compile")

    # Add normalization file if specified
    if hasattr(args, "norm_subdir") and args.norm_subdir and fold_cross is not None:
        norm_file = f"{args.models_dir}/{fold_cross}/{args.norm_subdir}/scores.h5"
        cmd_parts.extend(["-n", norm_file])

    if out_dir:
        cmd_parts.extend(["-o", out_dir])
    elif hasattr(args, "out_dir") and args.out_dir:
        cmd_parts.extend(["-o", args.out_dir])

    if hasattr(args, "pregrouped_seqs") and args.pregrouped_seqs:
        cmd_parts.append("--pregrouped_seqs")

    if hasattr(args, "rc") and args.rc:
        cmd_parts.append("--rc")

    if hasattr(args, "shifts") and args.shifts != "0":
        cmd_parts.extend(["--shifts", args.shifts])

    if hasattr(args, "gene_cov_t") and args.gene_cov_t > 0:
        cmd_parts.extend(["--gene_cov_t", str(args.gene_cov_t)])

    if hasattr(args, "span") and args.span:
        cmd_parts.append("--span")

    if hasattr(args, "snp_stats") and args.snp_stats:
        cmd_parts.extend(["--stats", args.snp_stats])

    if hasattr(args, "targets_file") and args.targets_file:
        cmd_parts.extend(["-t", args.targets_file])

    if hasattr(args, "targets_gene_file") and args.targets_gene_file:
        cmd_parts.extend(["--targets_gene", args.targets_gene_file])

    # Add explicit start and end indices
    cmd_parts.extend(["--index_start", str(start_i)])
    cmd_parts.extend(["--index_end", str(end_i)])

    # Add positional arguments
    cmd_parts.extend([args.params_file, model_file, args.vcf_file])

    return " ".join(cmd_parts)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
