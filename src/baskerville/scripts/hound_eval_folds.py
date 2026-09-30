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
import pdb

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
hound_eval_folds

Evaluate baskerville model replicates on cross folds using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baskerville model replicates on cross folds using given parameters and data."
    )

    # eval/spec options
    parser.add_argument(
        "--aggregate_genes",
        action="store_true",
        help="Aggregate predictions per unique gene across sequences [Default: %(default)s]",
    )
    parser.add_argument(
        "--band",
        default=None,
        type=int,
        help="Tracks read per band during spec streaming normalization; lower "
        "to reduce peak RAM [Default: hound_eval_spec's default]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="models",
        help="Training output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--rank",
        dest="rank_corr",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--seq_chunk",
        default=None,
        type=int,
        help="Zarr seq-axis chunk for preds/targets store; [Default: batch_size]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        type=str,
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Spatial step for specificity/spearmanr [Default: %(default)s]",
    )
    parser.add_argument(
        "--test",
        dest="test_only",
        default=False,
        action="store_true",
        help="Evaluate only the test set [Default: %(default)s]",
    )
    parser.add_argument(
        "--valid",
        dest="valid_only",
        default=False,
        action="store_true",
        help="Evaluate only the validation set [Default: %(default)s]",
    )
    parser.add_argument(
        "-tg",
        "--targets_gene_file",
        default=None,
        help="Gene targets file [Default: {data_dir}/targets_gene.txt]",
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
        "--name",
        default="fold",
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
    parser.add_argument(
        "-r",
        "--restart",
        default=False,
        action="store_true",
        help="Restart evaluation [Default: %(default)s]",
    )
    parser.add_argument(
        "--ram",
        default=False,
        action="store_true",
        help="Hold spec preds/targets in RAM instead of streaming to a Zarr "
        "store on disk (big-memory nodes only) [Default: %(default)s]",
    )
    parser.add_argument(
        "--scratch_dir",
        default=None,
        help="Directory for the spec preds/targets Zarr store; must be a real "
        "disk with room [Default: the job output directory]",
    )
    parser.add_argument(
        "--spec",
        default=False,
        action="store_true",
        help="Specificity evaluation [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    add_argparse_group(parser)
    args = parser.parse_args()

    # modularize for use in other scripts / tests
    eval_folds(args)


def eval_folds(args):
    """Build and run cross-fold evaluation jobs (slurm/local or GCP Batch)."""
    global slurmrunner

    # Fail fast on a missing -o, else it silently stages nothing and runs no jobs.
    if not os.path.isdir(args.out_dir):
        raise FileNotFoundError(
            f"Models directory {args.out_dir!r} not found "
            "(eval expects model_best.pth + folds.json under <out_dir>/f*c*/train/)."
        )

    # GCP: the dataset lives in a (zonal) bucket mounted read-only via GCSFuse;
    # the params file and the local models tree (model_best.pth files) are
    # content-addressed and staged to the cache bucket. The data dirs become
    # bare names resolved against the in-container mount. Must run before
    # make_runner() so the rewritten paths bake into the gcprunner.Job kwargs.
    gcp_backend = getattr(args, "backend", None) == "gcp"
    container_params = None
    container_models_dir = None  # staged-models mode (local tree → cache)
    container_data_dirs = None
    container_targets_gene = None
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
        models_sha = None
        if gcs_models_dir is None:
            models_sha, container_models_dir = stage_cache.stage_dir(
                args.out_dir, "models", filename="model_best.pth"
            )
        if args.targets_gene_file:
            _, container_targets_gene = stage_cache.stage_file(
                args.targets_gene_file, "targets_gene"
            )
        data_names = [os.path.basename(d.rstrip("/")) for d in args.data_dirs]
        container_data_dirs = [f"{args.gcp_data_local}/{n}" for n in data_names]
        if not args.gcp_output_dir:
            # Deterministic run id = hash(models, params, data URIs). Same inputs
            # → same GCS location → per-shard resume (skip finished acc.txt).
            # Marker mode has no staged models_sha; key on the GCS models dir.
            data_uris = sorted(f"{args.gcp_data_dir}/{n}" for n in data_names)
            data_uri_sha = hashlib.sha256("\n".join(data_uris).encode()).hexdigest()
            models_key = (
                models_sha or hashlib.sha256(gcs_models_dir.encode()).hexdigest()
            )
            run_id = stage_cache.build_run_id(
                models_key, params_sha, data_uri_sha, deterministic=True
            )
            args.gcp_output_dir = f"{stage_cache.output_prefix()}/eval/{run_id}"
        print("=" * 72)
        print(f"[gcp] run output dir: {args.gcp_output_dir}")
        loc = gcp_location_str(args)
        print(
            f"[gcp] data: {args.gcp_data_dir} ({', '.join(data_names)})  location: {loc}"
        )
        print("=" * 72)

    slurmrunner = make_runner(args, slurm_module=slurmrunner)

    # local backend resolves absolute paths; GCP uses staged/mounted paths.
    if gcp_backend:
        params_file = container_params
        data_dirs = container_data_dirs
    else:
        params_file = os.path.abspath(args.params_file)
        data_dirs = [os.path.abspath(data_dir) for data_dir in args.data_dirs]

    #######################################################
    # prep work

    # read data parameters
    num_data = len(args.data_dirs)
    if gcp_backend:
        data_stats = read_json_gcs(
            f"{args.gcp_data_dir}/{data_names[0]}/statistics.json"
        )
    else:
        data_stats_file = f"{data_dirs[0]}/statistics.json"
        with open(data_stats_file) as data_stats_open:
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
        num_cpu = 8
        num_gpu = 1
        time_base = 24

    #######################################################
    # evaluate folds

    # GCP container jobs already have the env active and write under
    # /workspace/out (rsync'd to gcp_output_dir by entry.sh); the dataset and
    # content-cache buckets are mounted read-only via GCSFuse. Local/slurm jobs
    # need conda activation and a $HOSTNAME echo.
    if gcp_backend:
        cmd_base = "hound_eval"
        spec_cmd_base = "hound_eval_spec"
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
                DataMount(gcs_models_dir, fold_utils.GCP_MODELS_MOUNT, mode="fuse")
            )
        gcp_extra = {
            "output_dir_gcs": args.gcp_output_dir,
            "data_mounts": data_mounts,
        }
        # Run-identity labels (per-fold gcprunner_fold added at each Job call) so
        # distinct eval runs are distinguishable regardless of --name.
        run_id = run_identity.run_id_from_gcs_dir(args.gcp_output_dir)
    else:
        env_base = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"
        cmd_base = f"{env_base} hound_eval"
        spec_cmd_base = f"{env_base} hound_eval_spec"
        gcp_extra = {}

    jobs = []
    found_model = False
    test_links = []  # (rel_eval_dir, test_fold) pairs to symlink after a GCP fetch

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
            # matches how the model was actually split (valid != fi+1 if cross>0).
            test_fold, valid_fold = _read_fold_splits(train_dir, num_folds, fi, ci)

            model_file = _resolve_model_file(
                gcp_backend, gcs_models_dir, container_models_dir, train_dir, fold_cross
            )

            for di in range(num_data):
                eval_sub = "eval" if num_data == 1 else f"eval{di}"

                # SLURM needs this dir to exist to write each job's .out/.err.
                if not gcp_backend:
                    os.makedirs(f"{it_dir}/{eval_sub}", exist_ok=True)

                # convenience symlinks to the held-out test fold's metrics + log.
                # The test fold isn't evaluated under --valid_only, so skip it.
                if not args.valid_only:
                    if gcp_backend:
                        # outputs are remote; link after the fetch-back below,
                        # relative to whatever local dir results are fetched into
                        test_links.append((f"{fold_cross}/{eval_sub}", test_fold))
                    else:
                        _link_test_metrics(f"{it_dir}/{eval_sub}", test_fold)

                for ei in range(num_folds):
                    if args.test_only and ei != test_fold:
                        continue
                    if args.valid_only and ei != valid_fold:
                        continue

                    rel = f"{fold_cross}/{eval_sub}/fold{ei}"

                    # check if done (GCS for gcp, local otherwise)
                    if gcp_backend:
                        acc_loc = f"{args.gcp_output_dir}/{rel}/acc.txt"
                        already_done = gcs_file_exist(acc_loc)
                        eval_fold_dir = f"/workspace/out/{rel}"
                    else:
                        eval_fold_dir = f"{it_dir}/{eval_sub}/fold{ei}"
                        acc_loc = f"{eval_fold_dir}/acc.txt"
                        already_done = os.path.isfile(acc_loc)
                    if already_done:
                        print(f"{acc_loc} already generated.")
                        continue

                    # hound evaluate
                    cmd = cmd_base
                    cmd += f" --dataset {di}"
                    cmd += f" -o {eval_fold_dir}"
                    if args.rank_corr:
                        cmd += " --rank"
                    if args.rc:
                        cmd += " --rc"
                    if args.save:
                        cmd += " --save"
                    if args.shifts:
                        cmd += f" --shifts {args.shifts}"
                    if args.aggregate_genes:
                        cmd += " --aggregate_genes"
                    targets_gene = (
                        container_targets_gene
                        if gcp_backend
                        else args.targets_gene_file
                    )
                    if targets_gene:
                        cmd += f" --targets_gene_file {targets_gene}"
                    cmd += f" --split fold{ei}"
                    if args.rank_corr or args.save:
                        cmd += f" --step {args.step}"
                    cmd += f" {params_file}"
                    cmd += f" {model_file}"
                    cmd += f" {data_dirs[di]}"

                    if args.save or args.aggregate_genes:
                        job_mem = 60000
                    else:
                        job_mem = 30000

                    if slurmrunner is None:
                        # Run locally
                        jobs.append(cmd)
                    else:
                        # Submit to SLURM / GCP Batch
                        name = f"{args.name}-eval-f{fi}e{ei}"
                        if gcp_backend:
                            out_file = f"{args.gcp_output_dir}/{rel}.out"
                            err_file = f"{args.gcp_output_dir}/{rel}.err"
                            gcp_extra["labels"] = run_identity.identity_labels(
                                run_id, "eval", fold_cross
                            )
                        else:
                            out_file = f"{eval_fold_dir}.out"
                            err_file = f"{eval_fold_dir}.err"
                        job = slurmrunner.Job(
                            cmd,
                            name=name,
                            out_file=out_file,
                            err_file=err_file,
                            queue=args.queue,
                            cpu=num_cpu,
                            gpu=num_gpu,
                            # On GCP, mem is the VM's hard memoryMib, not a soft
                            # request — passing the Slurm value would exceed the
                            # GPU profile's RAM and reject the job. mem=None gives
                            # the (single) task the whole VM; size it via --queue.
                            mem=None if gcp_backend else job_mem,
                            time=f"{time_base}:00:00",
                            **gcp_extra,
                        )
                        jobs.append(job)

    #######################################################
    # evaluate test specificity

    if args.spec:
        for ci in range(args.crosses):
            for fi in fold_index:
                fold_cross = f"f{fi}c{ci}"
                it_dir = f"{args.out_dir}/{fold_cross}"
                train_dir = f"{it_dir}/train"

                if not _model_present(
                    gcp_backend, gcs_models_dir, train_dir, fold_cross
                ):
                    continue

                # use the test fold training actually held out (not just fi)
                test_fold, _ = _read_fold_splits(train_dir, num_folds, fi, ci)

                model_file = _resolve_model_file(
                    gcp_backend,
                    gcs_models_dir,
                    container_models_dir,
                    train_dir,
                    fold_cross,
                )

                for di in range(num_data):
                    spec_sub = "spec" if num_data == 1 else f"spec{di}"
                    rel = f"{fold_cross}/{spec_sub}"

                    # check if done (GCS for gcp, local otherwise)
                    if gcp_backend:
                        acc_loc = f"{args.gcp_output_dir}/{rel}/acc.txt"
                        already_done = gcs_file_exist(acc_loc)
                        out_dir = f"/workspace/out/{rel}"
                    else:
                        out_dir = f"{it_dir}/{spec_sub}"
                        acc_loc = f"{out_dir}/acc.txt"
                        already_done = os.path.isfile(acc_loc)
                    if already_done:
                        print(f"{acc_loc} already generated.")
                        continue

                    cmd = spec_cmd_base
                    cmd += f" --dataset {di}"
                    cmd += f" -o {out_dir}"
                    cmd += f" --step {args.step}"
                    cmd += f" --split fold{test_fold}"
                    cmd += f" --ncpus {num_cpu}"
                    if args.band is not None:
                        cmd += f" --band {args.band}"
                    if args.seq_chunk is not None:
                        cmd += f" --seq_chunk {args.seq_chunk}"
                    if args.ram:
                        cmd += " --ram"
                    if args.scratch_dir is not None:
                        cmd += f" --scratch_dir {args.scratch_dir}"
                    if args.rc:
                        cmd += " --rc"
                    if args.shifts:
                        cmd += f" --shifts {args.shifts}"
                    cmd += f" {params_file}"
                    cmd += f" {model_file}"
                    cmd += f" {data_dirs[di]}"

                    if slurmrunner is None:
                        # Run locally
                        jobs.append(cmd)
                    else:
                        # Submit to SLURM / GCP Batch
                        name = f"{args.name}-spec-{fold_cross}"
                        spec_extra = dict(gcp_extra)
                        if gcp_backend:
                            out_file = f"{args.gcp_output_dir}/{rel}.out"
                            err_file = f"{args.gcp_output_dir}/{rel}.err"
                            spec_extra["boot_disk_gb"] = 200
                            spec_extra["labels"] = run_identity.identity_labels(
                                run_id, "eval", fold_cross
                            )
                        else:
                            out_file = f"{out_dir}.out"
                            err_file = f"{out_dir}.err"
                        job = slurmrunner.Job(
                            cmd,
                            name=name,
                            out_file=out_file,
                            err_file=err_file,
                            queue=args.queue,
                            cpu=num_cpu,
                            gpu=num_gpu,
                            mem=None if gcp_backend else 64000,
                            time=f"{3 * time_base}:00:00",
                            **spec_extra,
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
    if slurmrunner is None:
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
        # Each acc.txt is self-contained (no merge step). Mirror the GCS run
        # tree into the local out dir so eval results land next to the models:
        # <out_dir>/<fold_cross>/<eval_sub>/fold<ei>/acc.txt.
        if args.gcp_fetch_output is None:
            local_target = args.out_dir
        else:
            local_target = args.gcp_fetch_output  # empty string → skip
        if local_target:
            download_folder_from_gcs(args.gcp_output_dir, local_target)
            print(f"[fetch] results → {local_target}/")
            # restore convenience test-fold symlinks in the fetched tree
            for rel_eval_dir, test_fold in test_links:
                _link_test_metrics(f"{local_target}/{rel_eval_dir}", test_fold)
        print(f"[gcs] full run at {args.gcp_output_dir}")


################################################################################
# helpers
################################################################################
# The shared fold helpers (_read_fold_splits, _model_present, _resolve_model_file)
# now live in baskerville.helpers.fold_utils and are imported above under
# their original private names so existing call sites and tests are unchanged.


def _link_test_metrics(eval_dir, test_fold):
    """Symlink test -> fold{test_fold} and test.out -> fold{test_fold}.out.

    Relative targets so the links resolve wherever eval_dir lives. Targets may
    be dangling until eval completes. A stale link (e.g. left by the pre-GCP
    script that linked test -> fold{fi}, wrong when test_fold != fi for
    cross > 0) is repointed; a real file/dir at the link path is left alone.
    """
    os.makedirs(eval_dir, exist_ok=True)
    for name, target in (
        ("test", f"fold{test_fold}"),
        ("test.out", f"fold{test_fold}.out"),
    ):
        link = f"{eval_dir}/{name}"
        if os.path.islink(link):
            if os.readlink(link) == target:
                continue  # already correct
            os.unlink(link)  # stale link → repoint below
        elif os.path.lexists(link):
            continue  # real file/dir squatting the name → don't clobber
        os.symlink(target, link)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
