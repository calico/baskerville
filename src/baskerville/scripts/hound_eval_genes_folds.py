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
import hashlib
import json
import os

from baskerville import utils

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None

from gcprunner import run_identity
from gcprunner.argparse_helpers import (
    add_argparse_group,
    announce_gcp_image,
    gcp_location_str,
    make_runner,
    resolve_image_arg,
)
from gcprunner.batch_spec import DataMount
from baskerville.helpers import fold_utils, stage_cache
from baskerville.helpers.fold_utils import (
    model_present as _model_present,
    read_fold_splits as _read_fold_splits,
    resolve_model_file as _resolve_model_file,
)
from baskerville.helpers.gcs_utils import (
    download_folder_from_gcs,
    gcs_file_exist,
    read_json_gcs,
)

"""
hound_eval_genes_folds

Measure gene-level accuracy for baskerville model replicates on cross folds.
"""

# Container mount point for a GCP-trained models tree read straight from GCS
# (marker mode — see eval_genes_folds). Distinct from the dataset
# (/workspace/data) and content-cache (/workspace/cache) mounts.
_GCP_MODELS_MOUNT = fold_utils.GCP_MODELS_MOUNT

# GCP: --span aggregates whole-gene coverage profiles and needs ~90 GB RAM, so
# span jobs are force-pinned to a high-mem GPU VM regardless of --queue (which
# still governs the lighter per-bin gene eval jobs).
_GCP_SPAN_QUEUE = "l4-large"  # g2-standard-32: 1× L4, 32 vCPU, ~122 GB RAM


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Measure gene-level accuracy for model replicates on cross folds."
    )

    # eval options
    parser.add_argument(
        "-g",
        "--genes_gtf",
        required=True,
        help="GTF file with gene annotations.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="models",
        help="Training output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--pseudo_qtl",
        default=None,
        type=float,
        help="Quantile of coverage to add as pseudo counts to genes [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save_span",
        default=False,
        action="store_true",
        help="Store predicted/measured gene span coverage profiles [Default: %(default)s]",
    )
    parser.add_argument(
        "--seq_step",
        default=1,
        type=int,
        help="Compute only every seq_step sequence [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        type=str,
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--valid",
        default=False,
        action="store_true",
        help="Evaluate each replicate on its valid fold instead of its test fold "
        "[Default: %(default)s]",
    )

    # replication options
    parser.add_argument(
        "-c",
        "--crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds [Default: %(default)s]",
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
        help="Run a subset of folds [Default: %(default)s]",
    )
    parser.add_argument(
        "--local",
        default=False,
        action="store_true",
        help="Run jobs locally rather than on SLURM [Default: %(default)s]",
    )
    parser.add_argument(
        "--name",
        default="evalg",
        help="SLURM name prefix [Default: %(default)s]",
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
        default="titan_rtx",
        help="SLURM queue on which to run the jobs [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    add_argparse_group(parser)
    args = parser.parse_args()

    # modularize for use in other scripts / tests
    eval_genes_folds(args)


def eval_genes_folds(args):
    """Build and run cross-fold gene-level evaluation jobs (slurm/local or GCP)."""
    global slurmrunner

    # Fail fast on a missing -o, else it silently stages nothing and runs no jobs.
    if not os.path.isdir(args.out_dir):
        raise FileNotFoundError(
            f"Models directory {args.out_dir!r} not found "
            "(eval expects model_best.pth + folds.json under <out_dir>/f*c*/train/)."
        )

    # GCP: the dataset lives in a (zonal) bucket mounted read-only via GCSFuse;
    # the params file, GTF and the local models tree (model_best.pth files) are
    # content-addressed and staged to the cache bucket. The data dirs become
    # bare names resolved against the in-container mount. Must run before
    # make_runner() so the rewritten paths bake into the gcprunner.Job kwargs.
    gcp_backend = getattr(args, "backend", None) == "gcp"
    container_params = None
    container_models_dir = None  # staged-models mode (local tree → cache)
    container_data_dirs = None
    container_targets = None
    container_gtf = None
    data_names = None
    gcs_models_dir = None  # marker mode (weights read straight from GCS)
    if gcp_backend:
        # If the local -o dir carries a gcp_run.json marker, the models were
        # trained on GCP and their weights live in GCS (the train mirror
        # excludes .pth). Read them straight from there and inherit the run's
        # GCP config, instead of staging a local models tree to the cache.
        # An explicit --gcp_branch/--gcp_image is resolved before the marker so it
        # overrides the training image; the 'main' default is applied after the
        # marker (below) so a training-recorded image wins over it.
        marker = stage_cache.read_run_marker(args.out_dir)
        resolve_image_arg(args, allow_default=False, marker=marker)
        if marker:
            gcs_models_dir = marker.get("output_dir_gcs")
            if not gcs_models_dir:
                raise ValueError(
                    f"{args.out_dir}/{stage_cache.RUN_MARKER_NAME} has no "
                    "'output_dir_gcs' field."
                )
            inherited = stage_cache.apply_run_marker(args, marker)
            print(
                f"[gcp] models from GCS {gcs_models_dir} "
                f"(marker {args.out_dir}/{stage_cache.RUN_MARKER_NAME}; no staging)"
            )
            if inherited:
                print(f"[gcp] inherited from marker: {', '.join(inherited)}")

        # Fall back to the latest built commit on 'main' only if neither an
        # explicit image/branch nor the marker supplied one.
        announce_gcp_image(args, marker)

        if not args.gcp_data_dir:
            raise ValueError(
                "--backend gcp requires --gcp_data_dir gs://<bucket>/<prefix> "
                "(the dataset directory, pre-uploaded with 'gcloud storage rsync'). "
                f"It can also be recorded in {stage_cache.RUN_MARKER_NAME}."
            )
        params_sha, container_params = stage_cache.stage_file(
            args.params_file, "params"
        )
        gtf_sha, container_gtf = stage_cache.stage_file(args.genes_gtf, "gtf")
        models_sha = None
        if gcs_models_dir is None:
            models_sha, container_models_dir = stage_cache.stage_dir(
                args.out_dir, "models", filename="model_best.pth"
            )
        if args.targets_file:
            _, container_targets = stage_cache.stage_file(args.targets_file, "targets")
        data_names = [os.path.basename(d.rstrip("/")) for d in args.data_dirs]
        container_data_dirs = [f"{args.gcp_data_local}/{n}" for n in data_names]
        if not args.gcp_output_dir:
            # Deterministic run id = hash(models, params, gtf, data URIs). Same
            # inputs → same GCS location → per-replicate resume (skip finished
            # gene_metrics.tsv). Marker mode has no staged models_sha; key on the
            # GCS models dir. The GTF hash is included so different annotations
            # don't collide in the same output prefix.
            data_uris = sorted(f"{args.gcp_data_dir}/{n}" for n in data_names)
            data_uri_sha = hashlib.sha256("\n".join(data_uris).encode()).hexdigest()
            models_key = (
                models_sha or hashlib.sha256(gcs_models_dir.encode()).hexdigest()
            )
            run_id = stage_cache.build_run_id(
                models_key, params_sha, gtf_sha, data_uri_sha, deterministic=True
            )
            args.gcp_output_dir = f"{stage_cache.output_prefix()}/evalg/{run_id}"
        print("=" * 72)
        print(f"[gcp] run output dir: {args.gcp_output_dir}")
        loc = gcp_location_str(args)
        print(
            f"[gcp] data: {args.gcp_data_dir} ({', '.join(data_names)})  location: {loc}"
        )
        print("=" * 72)

        # On GCP, mem is the VM's hard RAM (not a soft Slurm request) and the
        # task gets the whole VM. --span jobs are auto-pinned to a high-mem VM
        # (see the eval loop); here we only sanity-check the lighter per-bin
        # jobs, which honor --queue, and surface the span auto-override.
        from gcprunner import resolve_gpu

        if args.span and args.queue != _GCP_SPAN_QUEUE:
            print(
                f"[gcp] NOTE: --span jobs need ~90 GB RAM, so they're pinned to "
                f"--queue {_GCP_SPAN_QUEUE} automatically. Your --queue "
                f"{args.queue!r} still applies to the per-bin gene eval jobs."
            )
        elif not args.span:
            prof = resolve_gpu(args.queue)
            if prof.mem_mib < 30000:
                print(
                    f"[gcp] WARNING: gene eval needs ~30 GB RAM but --queue "
                    f"{args.queue} ({prof.machine_type}, ~{prof.mem_mib // 1000} GB "
                    "usable) is too small — jobs will be rejected. Use a larger "
                    "GPU profile, e.g. --queue l4."
                )

    slurmrunner = make_runner(args, slurm_module=slurmrunner)

    # local backend resolves absolute paths; GCP uses staged/mounted paths.
    if gcp_backend:
        params_file = container_params
        data_dirs = container_data_dirs
        genes_gtf = container_gtf
        targets_file = container_targets
    else:
        params_file = os.path.abspath(args.params_file)
        data_dirs = [os.path.abspath(data_dir) for data_dir in args.data_dirs]
        genes_gtf = os.path.abspath(args.genes_gtf)
        targets_file = args.targets_file

    #######################################################
    # prep work

    # read data parameters
    num_data = len(args.data_dirs)
    if gcp_backend:
        data_stats = read_json_gcs(
            f"{args.gcp_data_dir}/{data_names[0]}/statistics.json"
        )
    else:
        with open(f"{data_dirs[0]}/statistics.json") as data_stats_open:
            data_stats = json.load(data_stats_open)

    # count folds
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # focus on initial folds
    if args.fold_subset is None:
        num_folds_score = num_folds
    else:
        num_folds_score = min(args.fold_subset, num_folds)
    fold_index = [fold_i for fold_i in range(num_folds_score)]

    if args.queue == "standard":
        num_cpu = 16
        num_gpu = 0
        time_base = 64
    else:
        num_cpu = 4
        num_gpu = 1
        time_base = 24

    #######################################################
    # evaluate folds

    # GCP container jobs already have the env active and write under
    # /workspace/out (rsync'd to gcp_output_dir by entry.sh); the dataset and
    # content-cache buckets are mounted read-only via GCSFuse. Local/slurm jobs
    # need conda activation and a $HOSTNAME echo.
    if gcp_backend:
        cmd_base = "hound_eval_genes"
        data_mounts = [
            DataMount(args.gcp_data_dir, args.gcp_data_local, mode="fuse"),
            DataMount(
                stage_cache.cache_prefix(),
                stage_cache.CONTAINER_CACHE_MOUNT,
                mode="fuse",
            ),
        ]
        # marker mode: also mount the GCS models dir read-only so model_best.pth
        # is read in place (no staging).
        if gcs_models_dir is not None:
            data_mounts.append(
                DataMount(gcs_models_dir, _GCP_MODELS_MOUNT, mode="fuse")
            )
        gcp_extra = {
            "output_dir_gcs": args.gcp_output_dir,
            "data_mounts": data_mounts,
        }
        # Run-identity labels (per-fold gcprunner_fold added at each Job call) so
        # distinct eval-genes runs are distinguishable regardless of --name.
        run_id = run_identity.run_id_from_gcs_dir(args.gcp_output_dir)
    else:
        env_base = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"
        cmd_base = f"{env_base} hound_eval_genes"
        gcp_extra = {}

    jobs = []
    found_model = False

    for ci in range(args.crosses):
        for fi in fold_index:
            fold_cross = f"f{fi}c{ci}"
            it_dir = f"{args.out_dir}/{fold_cross}"
            train_dir = f"{it_dir}/train"

            # model existence: GCS in marker mode (weights aren't local), the
            # staged/local tree otherwise.
            if not _model_present(gcp_backend, gcs_models_dir, train_dir, fold_cross):
                continue
            found_model = True

            # authoritative fold assignment from training (folds.json), so eval
            # matches how the model was actually split. Gene eval runs on the
            # held-out test fold by default (--valid switches to the valid fold).
            test_fold, valid_fold = _read_fold_splits(train_dir, num_folds, fi, ci)
            split_fold = valid_fold if args.valid else test_fold

            model_file = _resolve_model_file(
                gcp_backend, gcs_models_dir, container_models_dir, train_dir, fold_cross
            )

            for di in range(num_data):
                evalg_sub = "evalg" if num_data == 1 else f"evalg{di}"
                rel = f"{fold_cross}/{evalg_sub}"

                # check if done (GCS for gcp, local otherwise). gene_metrics.tsv
                # is the last file hound_eval_genes writes.
                if gcp_backend:
                    done_loc = f"{args.gcp_output_dir}/{rel}/gene_metrics.tsv"
                    already_done = gcs_file_exist(done_loc)
                    eval_dir = f"/workspace/out/{rel}"
                else:
                    eval_dir = f"{it_dir}/{evalg_sub}"
                    done_loc = f"{eval_dir}/gene_metrics.tsv"
                    already_done = os.path.isfile(done_loc)
                if already_done:
                    print(f"{done_loc} already generated.")
                    continue

                # hound evaluate genes
                cmd = cmd_base
                cmd += f" --head {di}"
                cmd += f" -o {eval_dir}"
                cmd += f" --split fold{split_fold}"
                cmd += f" --seq_step {args.seq_step}"
                if args.pseudo_qtl is not None:
                    cmd += f" --pseudo_qtl {args.pseudo_qtl:.2f}"
                if args.rc:
                    cmd += " --rc"
                if args.save_span:
                    cmd += " --save_span"
                if args.shifts:
                    cmd += f" --shifts {args.shifts}"
                if args.span:
                    cmd += " --span"
                if targets_file:
                    cmd += f" -t {targets_file}"
                cmd += f" {params_file}"
                cmd += f" {model_file}"
                cmd += f" {data_dirs[di]}"
                cmd += f" {genes_gtf}"

                job_mem = 90000 if args.span else 30000

                if args.local or slurmrunner is None:
                    # Run locally
                    jobs.append(cmd)
                else:
                    # Submit to SLURM / GCP Batch
                    name = f"{args.name}-evalg-{fold_cross}"
                    if gcp_backend:
                        out_file = f"{args.gcp_output_dir}/{rel}.out"
                        err_file = f"{args.gcp_output_dir}/{rel}.err"
                        gcp_extra["labels"] = run_identity.identity_labels(
                            run_id, "evalg", fold_cross
                        )
                    else:
                        out_file = f"{eval_dir}.out"
                        err_file = f"{eval_dir}.err"
                    job = slurmrunner.Job(
                        cmd,
                        name=name,
                        out_file=out_file,
                        err_file=err_file,
                        # --span needs ~90 GB; on GCP force a high-mem GPU VM
                        # (l4-large) regardless of --queue. mem=None → the task
                        # gets the whole VM (sized via --queue). Per-bin jobs
                        # honor --queue.
                        queue=(
                            _GCP_SPAN_QUEUE
                            if (gcp_backend and args.span)
                            else args.queue
                        ),
                        cpu=num_cpu,
                        gpu=num_gpu,
                        # On GCP, mem is the VM's hard memoryMib, not a soft
                        # request — passing the Slurm value would exceed the GPU
                        # profile's RAM and reject the job. mem=None gives the
                        # (single) task the whole VM; size it via --queue.
                        mem=None if gcp_backend else job_mem,
                        time=f"{4 * time_base}:00:00",
                        **gcp_extra,
                    )
                    jobs.append(job)

    # dir exists but had no replicate weights (vs. all jobs already complete)
    if not found_model:
        where = gcs_models_dir if gcs_models_dir is not None else args.out_dir
        raise FileNotFoundError(
            f"No model_best.pth found under {where} (expected at f*c*/train/). "
            "Check the models directory."
        )

    # Execute jobs
    if args.local or slurmrunner is None:
        utils.exec_par(jobs, max_proc=args.processes, verbose=True)
    else:
        slurmrunner.multi_run(
            jobs,
            max_proc=args.processes,
            verbose=True,
            launch_sleep=10,
            update_sleep=60,
        )

    #######################################################
    # GCP: fetch results back into the local out dir

    if gcp_backend:
        # Each gene eval output is self-contained (no merge step). Mirror the GCS
        # run tree into the local out dir so results land next to the models:
        # <out_dir>/<fold_cross>/<evalg_sub>/gene_metrics.tsv.
        if args.gcp_fetch_output is None:
            local_target = args.out_dir
        else:
            local_target = args.gcp_fetch_output  # empty string → skip
        if local_target:
            download_folder_from_gcs(args.gcp_output_dir, local_target)
            print(f"[fetch] results → {local_target}/")
        print(f"[gcs] full run at {args.gcp_output_dir}")


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
