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
import glob
import hashlib
import json
import os
import pdb
import subprocess
import tempfile
import threading
import time

from baskerville import utils

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None

from gcprunner.argparse_helpers import (
    add_argparse_group,
    gcp_location_str,
    make_runner,
    resolve_image_arg,
    resolve_project,
    resolve_region,
    resolve_zone,
)
from gcprunner import resolve_image
from gcprunner import run_identity
from gcprunner.batch_spec import DataMount
from baskerville.helpers import stage_cache
from baskerville.helpers.gcs_utils import (
    download_folder_from_gcs,
    gcs_file_exist,
    read_json_gcs,
)

"""
hound_train_folds

Train baskerville model replicates on cross folds using given parameters and data.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(description="Train a cross-fold set of models.")
    parser.add_argument(
        "-c",
        "--crosses",
        default=1,
        type=int,
        help="Number of cross-fold rounds [Default: %(default)s]",
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
        help="Run a subset of folds [Default: %(default)s]",
    )
    parser.add_argument(
        "--name",
        default="fold",
        help="SLURM name prefix [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
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
        default="titan_rtx",
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
    parser.add_argument(
        "--transfer",
        help="Transfer learn model directory",
    )
    parser.add_argument(
        "-w",
        "--whole",
        default=False,
        action="store_true",
        help="Use whole dataset for training, without validation [Default: %(default)s]",
    )
    parser.add_argument(
        "--gcp_max_rounds",
        default=10,
        type=int,
        help="GCP backend: max resubmit rounds for relaunching crashed/preempted "
        "folds before giving up [Default: %(default)s]",
    )
    parser.add_argument(
        "--gcp_retry_count",
        default=3,
        type=int,
        help="GCP backend: Batch in-place task retries per fold (resumes from "
        "checkpoint) before the outer resubmit loop takes over [Default: %(default)s]",
    )
    parser.add_argument(
        "--conclude",
        default=False,
        action="store_true",
        help="GCP backend: stop this run and pull it down. Cancel the run's "
        "active Batch jobs (gracefully) and mirror the full GCS run dir "
        "(including .pth weights) to the local -o dir. Ctrl-C the orchestrator "
        "first, then re-run the launch command with --conclude appended. GCS is "
        "left intact, so the run stays resumable [Default: %(default)s]",
    )
    parser.add_argument(
        "--extend",
        default=False,
        action="store_true",
        help="GCP backend: continue the run recorded in -o/gcp_run.json after "
        "raising train_epochs_max. Reuses its GCS output dir and image (explicit "
        "--gcp_output_dir / --gcp_image win) [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    add_argparse_group(parser)
    args = parser.parse_args()

    # GCP: the dataset lives in a (zonal) bucket mounted read-only via GCSFuse;
    # only the small params file is staged to the content cache. The data dirs
    # become bare names resolved against the in-container mount. Must run before
    # make_runner() so the rewritten kwargs land in the gcprunner.Job.
    gcp_backend = getattr(args, "backend", None) == "gcp"
    container_data_dirs = None
    container_transfer_dir = None
    if (args.conclude or args.extend) and not gcp_backend:
        parser.error("--conclude/--extend are only supported with --backend gcp")
    if args.extend:
        # Edited params hash to a new run id, so take the run dir and image
        # from the marker instead. Before resolve_image_arg, so the pinned
        # image beats --gcp_branch.
        marker = stage_cache.read_run_marker(args.out_dir)
        if not marker:
            parser.error(f"--extend needs {args.out_dir}/gcp_run.json")
        stage_cache.apply_run_marker(args, marker)
        args.gcp_output_dir = args.gcp_output_dir or marker["output_dir_gcs"]
        print(f"[gcp] extending {args.gcp_output_dir} on {args.gcp_image}")
    if gcp_backend and args.conclude:
        # Stop-and-pull path: resolve the run from the -o marker (or explicit
        # --gcp_output_dir), cancel its Batch jobs, and mirror everything down.
        # Runs before any param/transfer staging so conclude never uploads.
        _conclude_gcp_run(args)
        return
    if gcp_backend:
        # Resolve --gcp_branch (or the 'main' default) to one concrete, digest-
        # pinned image before make_runner() builds the Job kwargs and before the
        # run marker is written, so every fold and the marker share it. Explicit
        # --gcp_image / GCPRUNNER_IMAGE bypass the lookup.
        try:
            resolved_image = resolve_image_arg(args)
        except Exception as e:  # ImageBranchError and friends → clean CLI error
            parser.error(str(e))
        if resolved_image:
            print(f"[gcp] image: {resolved_image}")
        # Fail before uploading anything if the GCP settings are missing.
        try:
            resolve_project(args.gcp_project)
            stage_cache.cache_prefix()
            if not args.gcp_output_dir:
                stage_cache.output_prefix()
        except (ValueError, RuntimeError) as e:
            parser.error(str(e))
        if not args.gcp_data_dir:
            parser.error(
                "--backend gcp requires --gcp_data_dir gs://<bucket>/<prefix> "
                "(the dataset directory, pre-uploaded with 'gcloud storage rsync')."
            )
        if not resolve_zone(args.gcp_zone):
            print(
                "[gcp] no --gcp_zone: allocating region-wide. This relies on the "
                "dataset bucket's Rapid Cache covering every zone in "
                f"{resolve_region(args.gcp_region)}; otherwise cross-zone reads will be slow/costly. "
                "Pin --gcp_zone to force a single zone."
            )
        # Transfer learning: stage the foundation weights to the content cache so
        # the workers can seed from them. Only the model_best.pth files are
        # uploaded (content-addressed, cached across runs); they land under the
        # /workspace/cache mount the train job already binds. The per-fold
        # pretrained_model paths are embedded into the staged params below.
        transfer_sha = ""
        if args.transfer:
            transfer_sha, container_transfer_dir = stage_cache.stage_dir(
                args.transfer, "models", filename="model_best.pth"
            )
        # data dirs are bare names under the mount (e.g. hg38 mm10)
        data_names = [os.path.basename(d.rstrip("/")) for d in args.data_dirs]
        container_data_dirs = [f"{args.gcp_data_local}/{n}" for n in data_names]
        if not args.gcp_output_dir:
            # Deterministic run id = hash(params content, transfer weights, data
            # URIs). Independent of -o (local-only). Same inputs → same GCS
            # location → resume. The 366 GB data dirs are NOT hashed; only their
            # gs:// URIs are. transfer_sha is "" without --transfer, so the
            # non-transfer run id is unchanged.
            params_sha = stage_cache.hash_file(args.params_file)
            data_uris = sorted(f"{args.gcp_data_dir}/{n}" for n in data_names)
            data_uri_sha = hashlib.sha256("\n".join(data_uris).encode()).hexdigest()
            run_id = stage_cache.build_run_id(
                params_sha, transfer_sha, data_uri_sha, deterministic=True
            )
            args.gcp_output_dir = f"{stage_cache.output_prefix()}/train/{run_id}"
        print("=" * 72)
        print(f"[gcp] run output dir: {args.gcp_output_dir}")
        loc = gcp_location_str(args)
        print(
            f"[gcp] data: {args.gcp_data_dir} ({', '.join(data_names)})  location: {loc}"
        )
        print("=" * 72)

    global slurmrunner
    slurmrunner = make_runner(args, slurm_module=slurmrunner)

    #######################################################
    # prep work

    if not gcp_backend:
        if not args.restart and os.path.isdir(args.out_dir):
            raise ValueError(
                f"Output directory {args.out_dir} exists. Please remove or use --restart flag."
            )
        os.makedirs(args.out_dir, exist_ok=True)

    # read model parameters (params_file stays local even on GCP)
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_train = params.get("train", {})

    # read data parameters
    num_data = len(args.data_dirs)
    if gcp_backend:
        data_stats = read_json_gcs(
            f"{args.gcp_data_dir}/{data_names[0]}/statistics.json"
        )
    else:
        data_stats_file = f"{args.data_dirs[0]}/statistics.json"
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)

    # count folds
    num_folds = len([dkey for dkey in data_stats if dkey.startswith("fold")])

    # subset folds
    if args.fold_subset is not None:
        num_folds = min(args.fold_subset, num_folds)

    fold_index = [fold_i for fold_i in range(num_folds)]

    # arrange replicate output dirs (no per-rep data dirs anymore;
    # fold/cross are passed to hound_train, which resolves the splits).
    # On GCP the per-rep output lives in GCS, not on the submitter.
    if not gcp_backend:
        for ci in range(args.crosses):
            for fi in fold_index:
                fold_cross = f"f{fi}c{ci}"
                rep_dir = f"{args.out_dir}/{fold_cross}"
                os.makedirs(rep_dir, exist_ok=True)

                # write params.json (inject pretrained path under --transfer)
                if args.transfer:
                    pretrained_model = (
                        f"{args.transfer}/{fold_cross}/train/model_best.pth"
                    )
                else:
                    pretrained_model = None
                make_rep_params(args.params_file, rep_dir, pretrained_model)

    if args.setup:
        return

    #######################################################
    # train

    fold_crosses = [f"f{fi}c{ci}" for ci in range(args.crosses) for fi in fold_index]

    # GCP backend: status-aware resubmit loop (see _run_gcp_training).
    if gcp_backend:
        container_params_by_fold = _stage_train_params(
            args, fold_crosses, container_transfer_dir
        )
        _run_gcp_training(
            args,
            params_train,
            container_params_by_fold,
            container_data_dirs,
            fold_crosses,
        )
        return

    #######################################################
    # slurm / local backend

    jobs = []
    for ci in range(args.crosses):
        for fi in fold_index:
            fold_cross = f"f{fi}c{ci}"
            rep_dir = f"{args.out_dir}/{fold_cross}"

            train_dir = f"{rep_dir}/train"
            if args.restart and not args.checkpoint and os.path.isdir(train_dir):
                print(f"{rep_dir} found and skipped.")

            else:
                # train command
                cmd = utils.conda_activate(args.conda_env) + "echo $HOSTNAME;"

                cmd += " hound_train"
                cmd += f" -o {rep_dir}/train"
                cmd += f" --fold {fi} --cross {ci}"
                if args.whole:
                    cmd += " -w"
                cmd += f" {rep_dir}/params.json"
                cmd += f" {' '.join(args.data_dirs)}"

                if slurmrunner is None:
                    # Run locally
                    jobs.append(cmd)
                else:
                    # Submit to SLURM
                    name = f"{args.name}-train-{fold_cross}"
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
                        mem=45000,
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
# GCP backend: status-aware resubmit loop
################################################################################

# Bucket→local mirror cadence and what to skip. The mirror is a convenience so
# the user can follow progress (log.txt, progress.json, train.out/err) in the
# local -o dir without hand-running gcloud; the large .pth checkpoints stay in
# GCS as the source of truth and are deliberately not pulled every interval.
_MIRROR_INTERVAL_S = 300
_MIRROR_EXCLUDE = r".*\.pth$"


def _mirror_once(gcs_dir, local_dir, exclude=_MIRROR_EXCLUDE):
    """One pull of the GCS run dir → local dir via the GCS Python client.

    ``exclude`` is a regex of paths to skip; the periodic mirror passes the
    default (``.pth`` weights) to stay light, while the conclude path passes
    ``None`` for a full pull that includes the weights.

    Uses the google-cloud-storage client (honors ``GOOGLE_APPLICATION_CREDENTIALS``)
    rather than ``gcloud storage rsync``, whose reauth-gated user ADC silently
    stalled the mirror once the reauth window lapsed. Failures raise here.
    """
    os.makedirs(local_dir, exist_ok=True)
    download_folder_from_gcs(gcs_dir.rstrip("/"), local_dir, exclude_regex=exclude)


def _start_output_mirror(gcs_dir, local_dir, interval_s=_MIRROR_INTERVAL_S):
    """Background daemon mirroring the GCS run dir → local out_dir periodically.

    Returns a stop() callable. Errors (e.g. transient auth/network) are logged
    and retried on the next tick rather than killing the run.
    """
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            try:
                _mirror_once(gcs_dir, local_dir)
            except Exception as e:  # never let the mirror crash the orchestrator
                print(f"[gcp] output mirror warning: {e}")
            stop.wait(interval_s)

    t = threading.Thread(target=_loop, daemon=True, name="gcs-output-mirror")
    t.start()

    def _stop():
        stop.set()
        t.join(timeout=15)

    return _stop


################################################################################
# GCP backend: --conclude (stop a run and pull it down)
################################################################################

# How long to wait for cancelled jobs to actually leave the active set before
# doing the final pull, so their SIGTERM EXIT traps finish uploading. Polled in
# _CONCLUDE_WAIT_TICKS steps of _CONCLUDE_WAIT_SLEEP_S seconds.
_CONCLUDE_WAIT_TICKS = 18
_CONCLUDE_WAIT_SLEEP_S = 10


def _conclude_gcp_run(args):
    """Stop a GCP fold run and mirror it (weights included) to the local -o dir.

    Resolves the run's GCS location + project/region (explicit flags win, else
    the ``gcp_run.json`` marker the launch wrote into -o), cancels the run's
    active Batch jobs gracefully, waits for them to stop, then does a full pull.
    GCS is left intact, so the run stays resumable by re-running the launch
    command. Idempotent: safe to run again if interrupted.
    """
    import gcprunner

    marker = stage_cache.read_run_marker(args.out_dir)
    if marker:
        inherited = stage_cache.apply_run_marker(args, marker)
        if inherited:
            print(f"[gcp] inherited from run marker: {', '.join(inherited)}")
    gcs_dir = args.gcp_output_dir or (marker or {}).get("output_dir_gcs")
    if not gcs_dir:
        raise SystemExit(
            "[gcp] --conclude could not locate the run's GCS dir. Pass "
            "--gcp_output_dir gs://... (find it via 'gcloud batch jobs describe'), "
            "or run from the same -o dir that holds gcp_run.json."
        )
    project = resolve_project(args.gcp_project)
    region = resolve_region(args.gcp_region)

    print("=" * 72)
    print(f"[gcp] concluding run: {gcs_dir}")
    print(f"[gcp] project={project} region={region}")
    print("=" * 72)

    # Cancel this run's active Batch jobs, scoped strictly by run-identity labels
    # (not --name), so conclude can never touch a different run that shares a name.
    run_id = run_identity.run_id_from_gcs_dir(gcs_dir)
    run_labels = run_identity.identity_labels(run_id, "train")
    cancelled = gcprunner.cancel_active_jobs(project, region, labels=run_labels)
    if cancelled:
        print(f"[gcp] cancel requested for {len(cancelled)} job(s):")
        for jn in cancelled:
            print(f"    {jn.rsplit('/', 1)[-1]}")
        # Wait for the cancels to land so the SIGTERM EXIT traps upload final state.
        for _ in range(_CONCLUDE_WAIT_TICKS):
            time.sleep(_CONCLUDE_WAIT_SLEEP_S)
            still = [
                j
                for j in gcprunner.list_active_jobs(project, region)
                if all(j.get("labels", {}).get(k) == v for k, v in run_labels.items())
            ]
            if not still:
                break
            print(f"[gcp] waiting for {len(still)} job(s) to stop...")
        else:
            print("[gcp] warning: jobs still active after wait; pulling anyway.")
    else:
        print("[gcp] no active jobs to cancel (already stopped).")

    # Full pull, including the .pth weights the periodic mirror skips.
    _pull_full_run(gcs_dir, args.out_dir)
    _warn_if_orchestrator_alive(args.out_dir)


def _pull_full_run(gcs_dir, out_dir):
    """Full mirror (weights included) → local, then summarize.

    Shared by --conclude and successful completion; the periodic mirror skips
    .pth, so this is what puts checkpoints where eval/eqtl expect them.
    """
    print(f"[gcp] pulling {gcs_dir} -> {out_dir} (full, includes .pth)")
    _mirror_once(gcs_dir, out_dir, exclude=None)
    _print_conclude_summary(out_dir)


def _print_conclude_summary(out_dir):
    """Per-fold snapshot of the pulled run: last epoch + which weights are local."""
    train_dirs = sorted(glob.glob(os.path.join(out_dir, "f*c*", "train")))
    if not train_dirs:
        print(f"[gcp] pulled to {out_dir} (no fold dirs present yet).")
        return
    print(f"\n[gcp] local snapshot in {out_dir}:")
    for td in train_dirs:
        fc = os.path.basename(os.path.dirname(td))
        epoch = None
        pj = os.path.join(td, "progress.json")
        if os.path.isfile(pj):
            try:
                with open(pj) as f:
                    epoch = json.load(f).get("epoch")
            except (json.JSONDecodeError, OSError):
                pass
        have = [
            w
            for w in ("model_best.pth", "checkpoint.pth")
            if os.path.isfile(os.path.join(td, w))
        ]
        epoch_s = f"epoch {epoch}" if epoch is not None else "epoch ?"
        print(f"    {fc:10s} {epoch_s:12s} {', '.join(have) or 'no .pth yet'}")


def _warn_if_orchestrator_alive(out_dir):
    """Best-effort footgun guard: warn if another orchestrator still owns this -o.

    A live orchestrator would relaunch the folds we just cancelled. Purely
    advisory — never fails the conclude.
    """
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False
        ).stdout
    except Exception:
        return
    mypid = str(os.getpid())
    for line in ps.splitlines():
        if "hound_train_folds" not in line or "--conclude" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[0] == mypid:
            continue
        pid, cmdline = parts
        if f"-o {out_dir}" in cmdline or f"--out_dir {out_dir}" in cmdline:
            print(
                f"\n[gcp] WARNING: an orchestrator is still running (pid {pid}) for "
                f"-o {out_dir}. It will relaunch the folds you just cancelled — "
                f"kill it:\n    kill {pid}"
            )


def _read_progress_epoch(gcs_progress_uri):
    """Read the epoch from a fold's tiny progress.json in GCS, or None."""
    try:
        return read_json_gcs(gcs_progress_uri).get("epoch")
    except Exception:
        return None


def _fold_state(run_dir, fold_cross, active, epochs_max):
    """Infer a fold's state from GCS markers + the set of active Batch jobs.

    Returns (state, detail) where state is one of:
      complete   — COMPLETE marker present (done, skip)
      failed     — FAILED marker present (terminal-bad, e.g. loss_nan)
      running    — a Batch job for this fold is currently active
      incomplete — a checkpoint exists but no terminal marker (resume); detail=epoch
      new        — nothing yet (fresh start)

    A COMPLETE fold that stopped at max_epochs below ``epochs_max`` counts as
    incomplete, so raising train_epochs_max extends it.
    """
    base = f"{run_dir}/{fold_cross}/train"
    if gcs_file_exist(f"{base}/COMPLETE"):
        done = read_json_gcs(f"{base}/COMPLETE")
        if done.get("reason") != "max_epochs" or done["epoch"] + 1 >= epochs_max:
            return ("complete", None)
    if gcs_file_exist(f"{base}/FAILED"):
        return ("failed", None)
    if fold_cross in active:
        return ("running", None)
    if gcs_file_exist(f"{base}/checkpoint.pth"):
        return ("incomplete", _read_progress_epoch(f"{base}/progress.json"))
    return ("new", None)


def _active_fold_crosses(project, region, fold_crosses, run_id):
    """Set of fold_crosses with a currently-active Batch job for THIS run.

    Matches on the run-identity labels each train job carries
    (``gcprunner_run``/``gcprunner_kind``/``gcprunner_fold``), not on the job
    name — so a different run that happens to share ``--name`` (or the fragile
    ``train-<fold_cross>-`` substring) never counts as active here. Jobs without
    these labels (e.g. launched by pre-label code) are ignored.
    """
    import gcprunner

    active = set()
    running = gcprunner.list_active_jobs(project, region)
    want = run_identity.identity_labels(run_id, "train")
    for j in running:
        labels = j.get("labels", {})
        if any(labels.get(k) != v for k, v in want.items()):
            continue
        fc = labels.get(run_identity.FOLD_KEY)
        if fc in fold_crosses:
            active.add(fc)
    return active


def _print_status_table(states, round_i, max_rounds):
    print(f"\n[gcp] fold status (round {round_i + 1}/{max_rounds}):")
    for fc in sorted(states):
        state, detail = states[fc]
        extra = f" (epoch {detail})" if state == "incomplete" and detail else ""
        print(f"    {fc:10s} {state}{extra}")


def _build_gcp_train_job(
    args, params_train, container_params, container_data_dirs, fold_cross
):
    """Build one per-fold GCP training Job (resumable, retried, zone-pinned)."""
    fi, ci = fold_cross[1:].split("c")
    cmd = "hound_train -o /workspace/out/train"
    cmd += f" --fold {fi} --cross {ci}"
    if args.whole:
        cmd += " -w"
    cmd += f" {container_params} {' '.join(container_data_dirs)}"

    fold_gcs = f"{args.gcp_output_dir}/{fold_cross}"
    run_id = run_identity.run_id_from_gcs_dir(args.gcp_output_dir)
    # Two read-only fuse mounts: the dataset bucket (large, its own bucket) at
    # /workspace/data, and the content cache holding the staged params at
    # /workspace/cache. gcp_job_kwargs binds only the dataset mount (from
    # --gcp_data_dir), so pass both explicitly here — without the cache mount the
    # staged params_hydra.json isn't present in the container.
    data_mounts = [
        DataMount(args.gcp_data_dir, args.gcp_data_local, mode="fuse"),
        DataMount(
            stage_cache.cache_prefix(), stage_cache.CONTAINER_CACHE_MOUNT, mode="fuse"
        ),
    ]
    # mem=None → Batch memoryMib defaults to the queue profile's usable VM memory
    # (the task is the only one on the VM), instead of Batch's 2000 MiB default.
    # time capped at 7d (Batch's hard VM limit); the resubmit loop continues runs
    # beyond that.
    return slurmrunner.Job(
        cmd,
        name=f"{args.name}-train-{fold_cross}",
        out_file=f"{fold_gcs}/train.out",
        err_file=f"{fold_gcs}/train.err",
        queue=args.queue,
        cpu=8,
        gpu=params_train.get("num_gpu", 1),
        mem=None,
        time="7-0:0:0",
        output_dir_gcs=fold_gcs,
        retry_count=args.gcp_retry_count,
        data_mounts=data_mounts,
        labels=run_identity.identity_labels(run_id, "train", fold_cross),
    )


def _stage_train_params(args, fold_crosses, container_transfer_dir):
    """Stage per-fold params to the content cache; return {fold_cross: container_params}.

    Without ``--transfer`` every fold shares one staged params file. With
    ``--transfer`` each fold gets a params file whose ``model.pretrained_model``
    points at the staged foundation weights for that fold_cross (under the
    ``/workspace/cache`` mount the train job binds). Identical params across folds
    dedupe automatically via the content hash.
    """
    if not args.transfer:
        _, container_params = stage_cache.stage_file(args.params_file, "params")
        return {fc: container_params for fc in fold_crosses}

    mapping = {}
    with tempfile.TemporaryDirectory() as scratch:
        for fold_cross in fold_crosses:
            pretrained_model = (
                f"{container_transfer_dir}/{fold_cross}/train/model_best.pth"
            )
            rep_dir = os.path.join(scratch, fold_cross)
            os.makedirs(rep_dir, exist_ok=True)
            make_rep_params(args.params_file, rep_dir, pretrained_model)
            _, container_params = stage_cache.stage_file(
                f"{rep_dir}/params.json", "params"
            )
            mapping[fold_cross] = container_params
    return mapping


def _run_gcp_training(
    args, params_train, container_params_by_fold, container_data_dirs, fold_crosses
):
    """Run the resubmit loop, mirroring the GCS run dir to the local -o dir.

    The background mirror skips .pth for visibility; the final mirror is a full
    pull (weights included) on success, a light catch-up otherwise.
    """
    # Record the run's GCS origin + config in the local -o dir, so downstream
    # commands (e.g. hound_eval_folds) can read the weights straight from GCS
    # — the local mirror excludes .pth — and inherit these GCP settings. Write
    # it before the mirror starts so it's present even if the run is interrupted.
    if args.out_dir:
        # Record the *resolved* config (not the raw args) so a downstream command
        # inherits the effective project/region/zone/image even when the run was
        # configured via env vars (GCPRUNNER_*), which leave the args at None.
        resolved_region = resolve_region(args.gcp_region)
        resolved_project = resolve_project(args.gcp_project)
        marker_path = stage_cache.write_run_marker(
            args.out_dir,
            backend="gcp",
            output_dir_gcs=args.gcp_output_dir,
            gcp_data_dir=args.gcp_data_dir,
            gcp_image=resolve_image(
                args.gcp_image, region=resolved_region, project=resolved_project
            ),
            gcp_project=resolved_project,
            gcp_region=resolved_region,
            gcp_zone=resolve_zone(args.gcp_zone),
            transfer=args.transfer,
        )
        print(f"[gcp] wrote run marker {marker_path}")

    mirror_stop = (
        _start_output_mirror(args.gcp_output_dir, args.out_dir)
        if args.out_dir
        else None
    )
    if mirror_stop is not None:
        print(
            f"[gcp] mirroring {args.gcp_output_dir} -> {args.out_dir} "
            f"every {_MIRROR_INTERVAL_S // 60} min (excludes .pth)"
        )
    completed = False
    try:
        completed = _resubmit_loop(
            args,
            params_train,
            container_params_by_fold,
            container_data_dirs,
            fold_crosses,
        )
    finally:
        if mirror_stop is not None:
            mirror_stop()
            # A mirror hiccup at shutdown must not mask the run's outcome:
            # surface it and carry on. GCS is intact; --conclude re-pulls.
            try:
                if completed:
                    _pull_full_run(args.gcp_output_dir, args.out_dir)
                else:
                    # light catch-up only; --conclude pulls an interrupted run
                    _mirror_once(args.gcp_output_dir, args.out_dir)
            except Exception as e:
                print(
                    f"[gcp] final mirror to {args.out_dir} failed: {e}\n"
                    f"[gcp] GCS is intact at {args.gcp_output_dir}; re-run with "
                    "--conclude to pull it down."
                )


def _resubmit_loop(
    args, params_train, container_params_by_fold, container_data_dirs, fold_crosses
):
    """Status-aware resubmit loop: launch incomplete folds, wait, re-scan, repeat.

    GCS is the source of truth: each fold's COMPLETE/FAILED/checkpoint markers
    determine whether it's done, terminally broken, resumable, or fresh. Each
    launched fold resumes from its last checkpoint (hound_train restores from
    GCS on start). Idempotent: re-running this command picks up where it left
    off.

    Returns:
        bool: True iff every fold reached COMPLETE (caller pulls the full run,
        weights included). False on any abort/incomplete/foreign-run outcome.
    """
    run_dir = args.gcp_output_dir
    run_id = run_identity.run_id_from_gcs_dir(run_dir)
    project = resolve_project(args.gcp_project)
    region = resolve_region(args.gcp_region)
    epochs_max = params_train.get("train_epochs_max", 10000)  # Trainer default

    previous_states = {}

    for round_i in range(args.gcp_max_rounds):
        active = _active_fold_crosses(project, region, fold_crosses, run_id)
        states = {
            fc: _fold_state(run_dir, fc, active, epochs_max) for fc in fold_crosses
        }
        _print_status_table(states, round_i, args.gcp_max_rounds)

        # loss_nan etc. is terminal — stop and surface, do not relaunch
        failed = [fc for fc, (s, _) in states.items() if s == "failed"]
        if failed:
            print(
                f"\n[gcp] ABORTING — fold(s) marked FAILED (e.g. loss_nan): "
                f"{', '.join(sorted(failed))}"
            )
            print(
                "[gcp] These will not be retried; investigate params/data. "
                f"Logs under {run_dir}/<fold>/train/."
            )
            return False

        stuck = []
        for fc, (previous_state, previous_epoch) in previous_states.items():
            state, epoch = states[fc]
            if state == "new" or (
                state == previous_state == "incomplete"
                and (epoch is None or previous_epoch is None or epoch <= previous_epoch)
            ):
                stuck.append(fc)
        if stuck:
            print(
                f"\n[gcp] ABORTING — submitted fold(s) made no checkpoint progress: "
                f"{', '.join(sorted(stuck))}"
            )
            print(
                "[gcp] No new checkpoint or advancing epoch was observed. Inspect "
                "the Batch job status and, if the container started, "
                f"{run_dir}/<fold>/train.err before re-running."
            )
            return False

        if all(s == "complete" for s, _ in states.values()):
            print(f"\n[gcp] all {len(fold_crosses)} folds COMPLETE → {run_dir}")
            return True

        pending = [fc for fc, (s, _) in states.items() if s in ("incomplete", "new")]
        if not pending:
            running = [fc for fc, (s, _) in states.items() if s == "running"]
            print(
                f"\n[gcp] {len(running)} fold(s) still running from another "
                f"invocation: {', '.join(sorted(running))}. Re-run this command "
                "to confirm completion once they finish."
            )
            return False

        resuming = [fc for fc in pending if states[fc][0] == "incomplete"]
        if resuming:
            print(
                "[gcp] resuming from checkpoint: "
                + ", ".join(f"{fc}@epoch{states[fc][1]}" for fc in sorted(resuming))
            )
        print(
            f"[gcp] round {round_i + 1}: launching {len(pending)} fold(s): "
            + ", ".join(sorted(pending))
        )

        jobs = [
            _build_gcp_train_job(
                args,
                params_train,
                container_params_by_fold[fc],
                container_data_dirs,
                fc,
            )
            for fc in pending
        ]
        slurmrunner.multi_run(
            jobs,
            max_proc=args.processes,
            verbose=True,
            launch_sleep=10,
            update_sleep=60,
            raise_on_failure=False,
        )
        # remember what we launched so the next round can detect no-progress
        previous_states = {fc: states[fc] for fc in pending}

    # exhausted rounds
    active = _active_fold_crosses(project, region, fold_crosses, run_id)
    states = {fc: _fold_state(run_dir, fc, active, epochs_max) for fc in fold_crosses}
    incomplete = [fc for fc, (s, _) in states.items() if s != "complete"]
    if incomplete:
        print(
            f"\n[gcp] WARNING: {len(incomplete)} fold(s) still incomplete after "
            f"{args.gcp_max_rounds} rounds: {', '.join(sorted(incomplete))}. "
            "Re-run to continue (resumes from checkpoint)."
        )
        return False
    print(f"\n[gcp] all folds COMPLETE → {run_dir}")
    return True


def make_rep_params(params_file, rep_dir, pretrained_model):
    """Copy params file, including pretained model path.

    Args:
        params_file (str): Path to the original params file.
        rep_dir (str): Directory where the new params file will be created.
        pretrained_model (str): Path to the pretrained model, if any.
    """
    rep_params_file = f"{rep_dir}/params.json"
    with open(rep_params_file, "w") as rep_params_open:
        for line in open(params_file):
            print(line, file=rep_params_open, end="")
            if line.strip() == '"model": {':
                if pretrained_model is not None:
                    print(
                        f'        "pretrained_model": "{pretrained_model}",',
                        file=rep_params_open,
                    )


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
