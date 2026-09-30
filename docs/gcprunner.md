# gcprunner

A Google Cloud Batch–based supplement to `slurmrunner` (a non-public Slurm job runner)
for dispatching the `hound_*_folds.py` workloads (training, eval, ISM,
SNP scoring) onto ephemeral GPU VMs in GCP.

This document covers: why it exists, how it works, how to set it up, how to
use it from the CLI and programmatically, and what to watch out for.

---

## 1. Why this exists

`slurmrunner` puts jobs on our local Slurm cluster. That cluster is shared and
sometimes capacity-constrained. `gcprunner` gives us a second backend so the
same `hound_*_folds.py` scripts can run in GCP without rewriting them.

Three things drove the design:

1. **Cost discipline.** Baseline cost when nothing is running must be **zero**.
   No persistent cluster, no idle controller node, no leftover VMs after a
   crash. GCP Batch is the right primitive for this: VMs are created when a
   job starts and deleted automatically when the entrypoint exits — success
   or crash.
2. **Heterogeneous GPUs.** SNP scoring fans out into many small jobs and runs
   well on cheap L4s. Training wants A100/H100. We need to pick GPU type
   per job, not per cluster.
3. **API compatibility.** `slurmrunner.Job(...)` and `slurmrunner.multi_run(...)`
   are already woven into every `hound_*_folds.py`. The GCP backend mirrors
   that API so call sites don't change.

We evaluated **Cluster Director** as an alternative. It's nicer for true
multi-node A3/A4X training with topology-aware NCCL, but it requires a
persistent cluster that bills 24/7 unless actively torn down. That violates
the zero-baseline cost requirement, so it's deferred until we have a workload
that genuinely needs it.

---

## 2. Architecture at a glance

```
            ┌─────────────────────────────────────────────────────────────┐
            │  Your laptop / a login node                                 │
            │                                                             │
            │  hound_snp_folds --backend gcp ...                       │
            │       │                                                     │
            │       ▼                                                     │
            │  gcprunner.Job(cmd=..., queue="l4", gpu=1, ...)             │
            │       │                                                     │
            │       │  .launch()  → batch_v1.CreateJobRequest             │
            │       │              (built from JobSpec dict)              │
            │       ▼                                                     │
            └───────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
                ┌───────────────────────────────────────────────┐
                │  GCP Batch service                            │
                │  - Provisions a Compute Engine VM             │
                │  - Pulls the container from Artifact Registry │
                │  - Runs the entrypoint                        │
                │  - Streams logs to Cloud Logging              │
                │  - Reaps the VM on exit (any exit code)       │
                └───────────────────────────────────────────────┘
                                        │
                                        ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │  Ephemeral VM (e.g. g2-standard-8 with one nvidia-l4)               │
        │                                                                     │
        │   /usr/local/bin/gcprunner-entry           ◀── entry.sh             │
        │   ├─ optionally gcsfuse mount  gs://…  →  /workspace/data           │
        │   ├─ optionally gcloud storage rsync   gs://…  →  /workspace/stage  │
        │   ├─ run user command (tee stdout/stderr to local files)            │
        │   └─ EXIT trap (best-effort):                                       │
        │       ├─ gcloud storage rsync /workspace/out  →  gs://…/out         │
        │       ├─ gcloud storage cp stdout.log         →  gs://…/run.out     │
        │       └─ gcloud storage cp stderr.log         →  gs://…/run.err     │
        │                                                                     │
        │   VM is then deleted by Batch. Billing stops.                       │
        └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The package layout

```
src/gcprunner/
├── __init__.py            # re-exports Job, multi_run, …
├── job.py                 # Job class (submit, poll, delete)
├── runner.py              # multi_run, multi_update_status, list_active_jobs
├── gpus.py                # GPU alias table  (l4, a100-80, h100, …)
├── batch_spec.py          # JobSpec → Batch CreateJobRequest dict
├── entry.sh               # container entrypoint
└── argparse_helpers.py    # CLI integration for *_folds.py
```

Tests: [`src/tests/test_gcprunner.py`](../src/tests/test_gcprunner.py) — offline unit
tests. Run with `pytest src/tests/test_gcprunner.py`.

---

## 4. Component walkthrough

### 4.1 `gpus.py` — GPU profiles

A single dict maps a friendly alias to a `(machine_type, accelerator_type,
accelerator_count, mem_mib)` tuple. This is the GCP analogue of
slurmrunner's `GPU_TRANSLATIONS`.

| Alias       | Machine type   | GPU               | Count | Notes                                    |
| ----------- | -------------- | ----------------- | ----- | ---------------------------------------- |
| `cpu`       | n2-standard-8  | —                 | 0     | CPU-only                                 |
| `cpu-large` | n2-standard-32 | —                 | 0     | CPU-only                                 |
| `t4`        | n1-standard-8  | nvidia-tesla-t4   | 1     | Cheap, available widely                  |
| `l4`        | g2-standard-8  | nvidia-l4         | 1     | **Default for SNP/ISM/eval/grad**        |
| `l4-large`  | g2-standard-32 | nvidia-l4         | 1     | When you need more CPU/RAM beside the L4 |
| `v100`      | n1-standard-8  | nvidia-tesla-v100 | 1     |                                          |
| `a100-40`   | a2-highgpu-1g  | nvidia-tesla-a100 | 1     | A100 40GB                                |
| `a100-40x4` | a2-highgpu-4g  | nvidia-tesla-a100 | 4     | 4-way A100                               |
| `a100-80`   | a2-ultragpu-1g | nvidia-a100-80gb  | 1     | A100 80GB                                |
| `h100`      | a3-highgpu-1g  | nvidia-h100-80gb  | 1     |                                          |
| `h100x8`    | a3-highgpu-8g  | nvidia-h100-80gb  | 8     | 8-way H100                               |

Plus a `SLURM_ALIASES` table that maps existing slurmrunner queue names
(`rtx4090`, `titan_rtx`, `p100`, etc.) to the closest GCP profile. So an
existing `Job(..., queue="rtx4090")` keeps working when you switch backend:
`rtx4090` resolves to `l4` on GCP.

`resolve_gpu(queue)` does the lookup and raises `ValueError` on unknown
names. To add a new profile, edit `GPU_PROFILES` — no call sites change.

### 4.2 `batch_spec.py` — pure-dict spec builder

`JobSpec` is a dataclass that holds everything we need to describe one job
(command, image, GPU profile, CPU/mem, runtime, Spot, data mounts, env, GCS
output paths, service account, network, labels).

`build_batch_job_dict(spec)` converts it to a plain `dict` that matches the
Batch [CreateJobRequest REST schema](https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs#Job).
It deliberately returns a dict, not a protobuf object, so the spec builder
is unit-testable without the SDK installed and easy to log, persist, or
diff.

The dict includes:

- A single task group with `task_count=1` and `parallelism=1`. (We don't use
  Batch arrays — `multi_run` does the fan-out at the orchestration level, in
  parallel with `slurmrunner`.)
- One runnable per task with the container image, entrypoint script, and
  `--privileged` (required so the entrypoint can run `gcsfuse`).
- `cpu_milli` and optional `memory_mib` on `compute_resource`.
- `max_run_duration` (parsed from slurmrunner's `D-HH:MM:SS` strings).
- An instance policy with the GPU spec and, unless `provisioning="standard"`,
  `provisioning_model` (`SPOT` or `FLEX_START`; Flex Start also sets
  `reservation: NO_RESERVATION`, which Google requires to grant it).
- `logs_policy: CLOUD_LOGGING` so stdout/stderr stream to Cloud Logging
  automatically, surviving VM reaping.
- `labels`: an auto `gcprunner_name` (derived from the job name) plus any
  caller-supplied labels from `JobSpec.labels` / `Job(labels=...)`. Keys and
  values are sanitized to GCP's label rules (lowercase `[a-z0-9_-]`, ≤63 chars;
  keys must start with a letter). `gcprunner_name` is always written last, so a
  caller can't shadow it. The fold scripts use this to stamp **run-identity**
  labels (`gcprunner_run`/`gcprunner_kind`/`gcprunner_fold`) — see the fold-run
  docs — which let concurrent distinct runs be told apart by exact label match
  rather than by the human-readable `--name`.

The run-identity schema itself lives in `run_identity.py` — the single source of
truth for the label keys, the `run_id` derivation (`run_id_from_gcs_dir`), and
`identity_labels(run_id, kind, fold)`. Both the job builders and the matchers
(double-launch detection, `--conclude`) go through it, so the keys and the label
sanitization stay identical on both sides of every comparison.

Data mounts are encoded as a tab-separated environment variable
`GCPRUNNER_DATA_MOUNTS` that the entrypoint script reads. The format is
`mode\tgcs_uri\tlocal_path` per line, where mode is either `fuse` or
`stage_local` (see §6).

### 4.3 `entry.sh` — container entrypoint

This script is the heart of the cost-discipline story. The container image
installs it as `/usr/local/bin/gcprunner-entry` and Batch invokes it instead
of your command directly. It:

1. Registers `trap cleanup EXIT` before input setup. Setup failures stop
   execution. Cleanup cannot run after SIGKILL or loss of the VM.
2. Sets up data mounts in declaration order:
   - `fuse` → `gcsfuse --implicit-dirs -o ro [--only-dir prefix] bucket mountpoint`
   - `stage_local` → `gcloud storage rsync --recursive gs://… /local/…`
3. Runs the user command with `tee` to local log files.
4. On EXIT (success or any non-zero):
   - `gcloud storage rsync --recursive ${OUTPUT_DIR_LOCAL} ${OUTPUT_DIR_GCS}` if set.
   - Uploads the captured stdout/stderr to the optional `out_file`/`err_file`
     GCS paths.
   - Unmounts any active gcsfuse mounts.

After the entrypoint exits, Batch deletes the VM. Billing stops.

### 4.4 `job.py` — Job class

`Job(...)` accepts every slurmrunner kwarg (`cmd`, `name`, `out_file`,
`err_file`, `sb_file`, `queue`, `cpu`, `mem`, `time`, `gpu`) plus GCP-only
ones as keyword arguments:

- `image` — Artifact Registry URI; defaults to `$GCPRUNNER_IMAGE`.
- `project` — GCP project; defaults to `$GCPRUNNER_PROJECT`.
- `region` — Batch region; defaults to `us-central1` or `$GCPRUNNER_REGION`.
- `spot` — bool; default False.
- `data_mounts` — list of `DataMount(gcs_uri, local_path, mode)`.
- `env` — dict of extra environment variables.
- `output_dir_gcs`, `output_dir_local` — what to sync to GCS on exit.
- `service_account`, `network`, `subnetwork` — Compute Engine knobs.

Status strings are normalized to slurmrunner vocab (`PENDING`, `RUNNING`,
`COMPLETED`, `FAILED`) via `_STATE_MAP` so callers don't branch on backend.

Slurmrunner's `time="7-0:0:0"` format is parsed to seconds via
`_parse_slurm_time` and emitted as Batch `max_run_duration: "604800s"`.

`Job` is a thin wrapper. The Batch SDK isn't imported until `.launch()`
actually runs, so you can build, inspect, and unit-test `JobSpec`s without
having credentials configured.

### 4.5 `runner.py` — orchestration

`multi_run(jobs, max_proc, ...)` is API-compatible with `slurmrunner.multi_run`:

- Launches up to `max_proc` jobs at a time, with `launch_sleep` between
  submissions.
- Polls each active job's state every `update_sleep` seconds.
- A failed job is logged and counted as finished — Batch already reaped the
  VM, there's nothing for us to clean up.
- Raises after all jobs finish if any failed. Training uses
  `raise_on_failure=False` and checks its GCS markers to decide whether to resume.

`multi_update_status(jobs)` refreshes a batch of jobs. Batch has no batched
status RPC, so this just iterates per-job. Kept as a separate function to
match slurmrunner's surface.

`list_active_jobs(project, region)` is the **cost-watchdog helper**. Call
it any time to confirm nothing is left running and billing in that
project/region. Returns a list of `{"name", "state", "labels"}` dicts for
non-terminal jobs only.

`cancel_active_jobs(project, region, name_substr=None, labels=None)` gracefully
cancels (SIGTERM, not delete) the active jobs matching `name_substr` **and**
every key/value in `labels`. At least one filter is required — a bare call
raises rather than cancelling everything. Passing `labels={"gcprunner_run": …}`
scopes a cancel to exactly one run, so it can never reach a different run that
happens to share a `--name`.

### 4.6 `argparse_helpers.py` — CLI integration

Three helpers are used by the `hound_*_folds.py` scripts:

- `add_argparse_group(parser)` — adds the `--backend` flag plus
  GCP-specific knobs (`--gcp_project`, `--gcp_region`, `--gcp_zone`,
  `--gcp_image`, `--gcp_output_dir`, `--gcp_data_dir`, `--gcp_stage_dir`,
  `--gcp_provisioning`, `--gcp_service_account`, …). `--gcp_zone`
  pins the job to a single zone (vs. the whole region) so the VM can sit
  beside a zonal **Rapid Cache** of the dataset bucket — see §6.4.
- `make_runner(args, slurm_module)` — returns a runner facade
  matching slurmrunner's public API. For `--backend gcp` the returned
  object's `Job` constructor has the GCP-specific kwargs pre-bound (via a
  `_PartialRunner` wrapper), so existing `runner.Job(...)` call sites work
  unchanged. For `--backend slurm` returns the slurmrunner module directly.
- `gcp_job_kwargs(args)` — translates the parsed argparse namespace into
  kwargs for `Job(...)`.

---

## 5. Lifecycle of one job (cradle to grave)

```
  caller (login node)                 GCP Batch              VM            GCS
  ───────────────────────             ─────────              ──            ───

  Job(cmd="python score.py …",
      name="snp-f0-c0-job17",
      queue="l4", gpu=1, cpu=4,
      mem=30000, time="2:00:00",
      image=…, project=…,
      data_mounts=[Fuse(gs://b/d, /workspace/data)],
      output_dir_gcs=gs://b/out/job17)

  .launch()           ───────────►   CreateJob
                                     ├─ allocate VM
                                     ├─ pull image
                                     └─ start container ───────►
                                                                gcprunner-entry
                                                                 ├─ trap EXIT
                                                                 ├─ gcsfuse mount
                                                                 └─ run cmd:
                                                                    python score.py
                                                                    │
                                                                    │ (writes
                                                                    │  outputs to
                                                                    │  /workspace/out)
                                                                    │
                                                                    │ exit 0  or  exit !=0
                                                                    │
                                                                    EXIT trap fires:
                                                                    ├─ gcloud storage rsync ──► gs://b/out/job17
                                                                    ├─ gcloud storage cp stdout ──► gs://b/out/job17.out
                                                                    └─ unmount gcsfuse
                                                                 entrypoint exits
                                     ◄──── container exit code
                                     │
                                     ├─ delete VM
                                     └─ mark job SUCCEEDED/FAILED

  poll: .update_status()   ────►   GetJob
                                    └─ state=SUCCEEDED/FAILED ──►
  status="COMPLETED"/"FAILED"
```

The important property: **the EXIT trap runs before the container exit code
is reported to Batch**, so output and logs are always captured, even on
crash. The VM is then reaped by Batch, so billing stops within a minute or
two.

---

## 6. Data strategies

Two modes, chosen per data source per job:

### 6.1 `stage_local` — copy to local SSD

```python
DataMount("gs://my-bucket/train/fold0", "/workspace/stage", mode="stage_local")
```

At container start, the entrypoint runs `gcloud storage rsync --recursive` from the GCS
prefix to the local path. Use this for **training/distill** where the
dataset is too large or accessed too randomly for FUSE to keep up. Pair it
with an `a2-*` or `a3-*` machine type that has local SSDs (or attach one
explicitly).

Trade-off: pays the download cost once per VM lifetime. Training runs for
hours-to-days so it's amortized. SNP/ISM jobs are too short for this to
make sense.

### 6.2 `fuse` — GCSFuse read-only

```python
DataMount("gs://my-bucket/data", "/workspace/data", mode="fuse")
```

Mounts the bucket (or prefix) at `local_path` as a read-only filesystem via
[GCSFuse](https://cloud.google.com/storage/docs/cloud-storage-fuse/overview).
Reads are pulled lazily from GCS. Use this for **eval / grad / ISM / SNP**
where the input is small or read sparsely (FASTA, VCF chunk, targets file,
model checkpoint).

Trade-off: variable throughput, occasional stalls. Fine for sparse reads,
bad for shuffled training over hundreds of GB.

### 6.3 `fuse` + Rapid Cache — the training-scale strategy

`stage_local` (§6.1) forces you to size a local SSD to the whole dataset,
which is exactly the disk-math we want to stop doing for the 300–500 GB
training corpora. The alternative is to keep the dataset in a **regional
bucket**, mount it read-only via GCSFuse (§6.2), and put a
[**Rapid Cache**](https://docs.cloud.google.com/storage/docs/rapid/rapid-cache)
(formerly _Anywhere Cache_) in front of it — a fully-managed, SSD-backed
**zonal** read cache. First read of a 2 MB chunk pulls it from the regional
bucket into the in-zone cache; every later read (next epoch, next fold,
sibling job) is served from SSD at low latency with no per-op Class-B charge
and no cross-zone egress. No VM disk to size; the cache scales itself.

The one constraint that shapes everything: **a Rapid Cache is zonal and only
serves VMs in the same zone.** That's why `hound_train_folds --backend gcp`
**requires `--gcp_zone`** and pins the Batch VM to it. You give up the
region-wide GPU fallback (a pinned zone can be GPU-exhausted with no
auto-retry elsewhere) in exchange for cache locality. For training — a few
long jobs, not a fan-out of hundreds — that trade is usually right.

**Setup (one-time, out of band):**

```bash
# 1. Upload the dataset to a regional bucket (region must contain your zone).
gcloud storage rsync --recursive \
    data/hg38 \
    gs://my-data-bucket/2-10/hg38
gcloud storage rsync --recursive \
    data/mm10 \
    gs://my-data-bucket/2-10/mm10

# 2. Create a Rapid Cache for that bucket in the zone you'll train in.
#    --enable-ingest-on-write is optional; here data already exists, so it
#    won't help — the first epoch warms the cache instead.
gcloud storage buckets anywhere-caches create gs://my-data-bucket us-central1-a \
    --ttl=7d

# 3. (Optional) Pre-warm before the first epoch so epoch 1 isn't cache-cold:
#    read the dataset once from a VM in that zone (e.g. a cheap throwaway
#    instance, or just accept the first-epoch fill).

# Inspect / tear down:
gcloud storage buckets anywhere-caches list gs://my-data-bucket
gcloud storage buckets anywhere-caches disable gs://my-data-bucket/us-central1-a
```

Idle caches in unused zones cost nothing, so you can also create one per
candidate zone. But since we pin a single zone for training, one cache in
that zone is enough.

**AI zones.** Google also offers dedicated _AI zones_ with concentrated
GPU/TPU capacity and Rapid Cache as the recommended performance layer
(see [Cloud Storage AI zones](https://docs.cloud.google.com/storage/docs/ai-zones)).
These have special zone names — `us-central1-ai1a`, `europe-west4-ai1a`,
`us-south1-ai1b` — rather than the usual `-a/-b/-c`. They pass `--gcp_zone`
unchanged (the region prefix still matches). They're Preview and must be
enabled per project; if you have access, pinning to an AI zone is the
highest-capacity place to land a GPU training job beside its cache.

**Fallback.** If Rapid Cache turns out slow or unavailable, the documented
backups are (a) a [zonal _Rapid Storage_ bucket](https://docs.cloud.google.com/storage/docs/rapid/rapid-storage)
(premium per-GB, but co-located and POSIX-ish), or (b) the older
`stage_local` full-copy to a local SSD (§6.1). Both require more disk-size
thinking than Rapid Cache, which is why it's the default recommendation.

### 6.4 Outputs

Outputs are always written to a local working directory inside the container
(default `/workspace/out`). On exit, the entrypoint syncs that directory to
the GCS path you set as `output_dir_gcs`. The sync runs in both the success
and the crash paths via the EXIT trap.

Training writes to `/workspace/out/train`, synced to
`<gcp_output_dir>/f{i}c{j}/train/`. Re-running skips folds with a `COMPLETE`
marker and resumes incomplete folds from `checkpoint.pth`.

---

## 7. Cost behavior

The cost model is a deliberate design point. Here's what you pay for and
when:

| State                           | Billing                                      |
| ------------------------------- | -------------------------------------------- |
| Nothing running                 | **$0** — no persistent infrastructure        |
| Job queued, no VM yet           | $0                                           |
| VM provisioning (pre-container) | A few seconds of VM time                     |
| Container running               | VM time at the machine type's rate           |
| Container exited (any code)     | VM is reaped within ~1 minute; billing stops |
| Job sitting "FAILED" in Batch   | $0 — VM is gone, Batch job record is free    |

The cost guardrails built into gcprunner:

1. **On-demand by default** everywhere (`--gcp_provisioning standard`, set in
   [`argparse_helpers.py`](../src/gcprunner/argparse_helpers.py)). Pass
   `--gcp_provisioning spot` for cheaper, preemptible VMs, or
   `--gcp_provisioning flex_start` to queue for capacity and then run
   uninterrupted for up to 7 days at the deepest discount.
2. **Hard time cap** on every job. `time="7-0:0:0"` becomes Batch's
   `max_run_duration: "604800s"`. Anything past the cap is killed and the
   VM reaped.
3. **No persistent resources created by the runner.** Buckets, Artifact
   Registry, IAM, networks — you provision those externally. `gcprunner`
   only creates Batch jobs, which are billing-free records once the VM dies.
4. **`list_active_jobs(project, region)`** for a quick "is anything still
   running?" sanity check after a session.

The Batch service itself has a hard **7-day VM cap**. Anything longer than
that needs your training script to support checkpoint-resume so the
orchestrator can re-launch. (This is a Batch limitation, not a gcprunner
choice. Cluster Director doesn't have this cap but trades it for a
continuously-billing controller.)

---

## 8. Setup checklist

You need these in place before the first run:

### 8.1 GCP project setup (one-time)

1. **Enable APIs** in the project where Batch jobs will run (your
   `GCPRUNNER_PROJECT`, e.g. `my-gcp-project`):
   ```bash
   gcloud services enable batch.googleapis.com \
       storage.googleapis.com compute.googleapis.com \
       logging.googleapis.com artifactregistry.googleapis.com
   ```
   (Skip `artifactregistry` if your images live in a registry in another
   project — see §8.2.)
2. **Create a GCS bucket** you own for the input cache and outputs, e.g.
   `gs://my-bucket`, and point `GCPRUNNER_CACHE_PREFIX` /
   `GCPRUNNER_OUTPUT_PREFIX` at prefixes in it (§8.3). There are no built-in
   defaults; commands fail fast if these are unset.
3. **Batch runtime service account.** Batch VMs run under a service account;
   by default that's the **Compute Engine default service account**
   (`<project-number>-compute@developer.gserviceaccount.com`), which every
   GCP project gets automatically. It usually has `roles/editor` project-wide
   (covers GCS writes, log emission, Batch reporting) and can pull from an
   Artifact Registry repo in the same project. If your registry lives in a
   different project, grant this SA `roles/artifactregistry.reader` on that
   repo. Many organizations disable that automatic `Editor` grant
   (`iam.automaticIamGrantsForDefaultServiceAccounts`); if yours does, grant
   the default SA the roles listed in the next paragraph.

   _If you want tighter scoping later_ (the default SA is broad — `Editor`
   covers far more than this workload needs), create a dedicated SA, grant
   it `roles/storage.objectAdmin` on your bucket, `roles/logging.logWriter`,
   `roles/batch.agentReporter` and `roles/artifactregistry.reader` on the
   image repo, and pass it via `--gcp_service_account`. Not needed for a
   first run.

4. **Your own permissions.** The account you submit from needs
   `roles/batch.jobsEditor` on the project and `roles/iam.serviceAccountUser`
   on the runtime service account, plus write access to your bucket.

5. **GPU quota**. Each GPU family (L4, A100, H100) has its own quota in each
   region. Request quota in the GCP console before your first job — first
   submissions will fail with quota errors otherwise.

### 8.2 Container image

The Dockerfile at [`dockerfiles/baskerville.Dockerfile`](../dockerfiles/baskerville.Dockerfile)
installs `gcsfuse`, the `google-cloud-cli` (for `gcloud storage`), and the
`gcprunner-entry` script at the right location.

**To publish an image**, build it and push it to an Artifact Registry repo
you control. By default gcprunner looks for images at
`<region>-docker.pkg.dev/<GCPRUNNER_PROJECT>/baskerville/baskerville`;
set `GCPRUNNER_IMAGE_BASE` if your repo lives elsewhere.

```bash
gcloud artifacts repositories create baskerville \
    --repository-format=docker --location=us-central1 --project=my-gcp-project
gcloud auth configure-docker us-central1-docker.pkg.dev

IMAGE=us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville
docker build -f dockerfiles/baskerville.Dockerfile \
    -t "$IMAGE:commit-$(git rev-parse HEAD)" -t "$IMAGE:latest" .
docker push --all-tags "$IMAGE"
```

Or build remotely with Cloud Build, no local Docker needed (about 10 minutes).
Run it from a full clone, not a `git worktree`: setuptools_scm reads `.git`.

```bash
gcloud services enable cloudbuild.googleapis.com --project=my-gcp-project
gcloud builds submit --project=my-gcp-project --region=us-central1 \
    --config=dockerfiles/cloudbuild.yaml --ignore-file=.dockerignore \
    --substitutions=_IMAGE=$IMAGE,_SHA=$(git rev-parse HEAD) .
```

The `commit-<full-git-sha>` tag is what `--gcp_branch` (below) looks up; tag
CI-built images the same way if you automate this.

**To use that image with gcprunner**, set the image on the caller side:

```bash
export GCPRUNNER_IMAGE=<tag>            # bare tag, expanded against the base
# or a full URI:
export GCPRUNNER_IMAGE=$IMAGE:<tag>
```

or pass `--gcp_image <…>` per invocation. If your CI tags images with a
numeric run id behind a prefix (e.g. `build-123`), set
`GCPRUNNER_IMAGE_TAG_PREFIX=build-` so `--gcp_image 123` expands to it.

#### Selecting an image by git branch (`--gcp_branch`)

Usually what you want is "the latest image built from branch X." Use
`--gcp_branch <name>` for that. There is **no per-branch tag** in the
registry: tag every image you push with `commit-<full-git-sha>` (as above) and
optionally a moving `latest`. `--gcp_branch` correlates by commit: it walks the
branch's history newest→oldest and picks the newest commit that has a
`commit-<sha>` image, resolving to a **digest-pinned** URI
(`…/baskerville@sha256:<digest>`) for reproducibility.

```bash
# latest built image on my-feature
hound_train_folds ... --backend gcp --gcp_branch my-feature

# no image args at all → defaults to the latest built commit on main
hound_train_folds ... --backend gcp
```

Precedence: `--gcp_image` > `--gcp_branch` > `GCPRUNNER_IMAGE` > the `main`
default. Notes and caveats:

- Requires `gcloud` on PATH with registry read access (already a Dockerfile
  dependency); the lookup runs once at launch. Pass `--gcp_image`/set
  `GCPRUNNER_IMAGE` to skip it entirely (e.g. offline).
- Only commits you actually built and pushed have images, so the newest commit
  and the newest **built** commit can differ — the resolver reports which
  commit/digest it chose. If no commit on the branch was ever built, it fails
  fast telling you to build one or pass `--gcp_image`.
- The image reflects a pushed, built commit — **not** your local uncommitted
  edits.
- The registry queried is `GCPRUNNER_IMAGE_BASE` if set, else
  `<GCPRUNNER_REGION>-docker.pkg.dev/<GCPRUNNER_PROJECT>/baskerville/baskerville`.
  Set `GCPRUNNER_IMAGE_BASE` when images live in a different project or region
  than your Batch jobs.

**Pull-side permission:** the Batch runtime SA from §8.1 needs read access to
the image repo. Without it, jobs fail at the image-pull step before the
container starts.

#### 8.2.1 Local debug builds (optional)

To iterate on the Dockerfile itself, push to a personal tag (e.g.
`dev-<name>`) with the commands above and pass it via `--gcp_image`.

### 8.3 Caller environment

On the machine that calls `hound_*_folds.py --backend gcp`:

```bash
# Auth — pick one of these
gcloud auth application-default login            # interactive
# or
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

# Required — there are no built-in defaults
export GCPRUNNER_PROJECT=my-gcp-project
export GCPRUNNER_CACHE_PREFIX=gs://my-bucket/cache
export GCPRUNNER_OUTPUT_PREFIX=gs://my-bucket/output

# Optional
export GCPRUNNER_REGION=us-central1         # default us-central1
export GCPRUNNER_IMAGE_BASE=us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville
export GCPRUNNER_IMAGE=<tag>                 # else --gcp_branch / latest built on main
```

Install the package:

```bash
pip install -e /path/to/baskerville
```

---

## 9. CLI usage

Every `hound_*_folds.py` now accepts `--backend {local,slurm,gcp}` (default
`slurm` if `slurmrunner` is installed, else `local`) and a "runner backend"
argparse group. The `--gcp_*` flags are inert unless `--backend gcp`.

### 9.1 SNP scoring (the headline use case)

```bash
hound_snp_folds \
    --backend gcp \
    --gcp_project my-gcp-project \
    -q l4 \
    -j 1024 \
    -p 64 \
    --crosses 1 \
    -f ~/refs/hg38.fa \
    -t ~/refs/targets.txt \
    -o snp_out \
    ~/configs/params.json ~/models/borzoi-v3 ~/vcfs/test.vcf
```

Note all positional and `-f` / `-t` paths are **local on the laptop**. The
script handles uploading them for you:

- **Auto-staging.** Each input (`vcf_file`, `models_dir`, `params_file`,
  `genome_fasta`, `targets_file`) is SHA256-hashed and uploaded to
  `$GCPRUNNER_CACHE_PREFIX/<type>/<sha>/…` if not already there. Re-runs with
  the same inputs are no-ops on the upload side (file hashes are memoized
  in `~/.cache/baskerville/stage_hashes.json`, so re-hashing a 2 GB
  weight file is one stat() call).
- **Auto-output.** When `--gcp_output_dir` is omitted (default), results
  go to `$GCPRUNNER_OUTPUT_PREFIX/snp/<vcf8>-<models8>/` (deterministic, so a
  re-run resumes). The script prints the full URI on startup.
- **Path rewriting.** Before each Batch job is built, the local paths in
  `args` are rewritten to their in-container counterparts under a GCSFuse
  mount of `$GCPRUNNER_CACHE_PREFIX` at `/workspace/cache`. So the
  in-container `hound_snp` command points at
  `/workspace/cache/vcf/<sha>/test.vcf`, etc.
- **Post-job merge.** After all jobs finish, the orchestrator downloads
  each fold's per-job `scores.h5` files, calls `collect_scores()` to merge
  them, and uploads the consolidated `scores.h5` back to GCS — same shape
  the slurm backend produces.
- **Local mirror.** By default the merged per-fold `scores.h5` files are
  fetched to `-o snp_out/<fold_cross>/scores.h5` on the laptop (skipping
  the per-job `jobN/` intermediates, which stay in GCS). Override the
  target with `--gcp_fetch_output OTHER_DIR`, or skip the fetch with
  `--gcp_fetch_output ""`.
- **Concurrency.** Runs at most 64 jobs concurrently (`-p 64`) on L4
  on-demand VMs by default; pass `--gcp_provisioning spot` for Spot.

The cache and output locations come from `GCPRUNNER_CACHE_PREFIX` /
`GCPRUNNER_OUTPUT_PREFIX` (§8.3); both must point at a bucket you own.

### 9.2 Cross-fold training

Training mounts the dataset from a regional bucket via GCSFuse and relies on a
**Rapid Cache** for throughput (see §6.3), so there's no local SSD to size. Two
things differ from SNP scoring:

- The dataset is **not** auto-staged. You upload it once with
  `gcloud storage rsync` (§6.3) and pass `--gcp_data_dir gs://…` pointing at
  the parent prefix. The positional `data_dirs` become **bare names** under
  that prefix.
- `--gcp_zone` is **optional**. If you created a Rapid Cache in **every** zone
  of the region, omit it: the job allocates region-wide and Batch lands GPU
  capacity in whichever zone is available (best for scarce a100 spot). Pin
  `--gcp_zone` only if the cache covers a single zone — the VM must co-locate
  with a zonal cache or reads fall back to slow/costly cross-zone access.

```bash
hound_train_folds \
    --backend gcp \
    --gcp_project my-gcp-project \
    --gcp_region us-west1 \
    --gcp_image us-west1-docker.pkg.dev/my-gcp-project/baskerville/baskerville:latest \
    --gcp_data_dir gs://my-data-bucket/2-10-26 \
    -q a100-80 \
    -c 1 \
    -o models_hydra \
    params.json hg38 mm10
```

What happens:

- Only `params.json` is staged to the content cache; `hg38` and `mm10` are
  resolved to `/workspace/data/hg38` and `/workspace/data/mm10` inside the
  container, served from `gs://my-data-bucket/2-10-26/{hg38,mm10}` through the
  Rapid Cache.
- One Batch job per `(fold, cross)` is launched. With no `--gcp_zone` they
  allocate region-wide; pin a zone to force one. `hound_train` derives its
  train/valid/test fold split from `--fold`/`--cross` and logs it to
  `folds.json`.
- **`-o` is local-only.** The GCS output dir defaults to a **deterministic
  pure hash of (params, data)**: `$GCPRUNNER_OUTPUT_PREFIX/train/<params8>-<data8>/`,
  printed loudly at launch. Same params+data → same location → resume. Override
  with `--gcp_output_dir` for corner cases (e.g. code changed but params/data
  unchanged, and you want a fresh location). Each fold lands at
  `…/<run_id>/f{i}c{j}/train/`.
- **A run marker is written to the local `-o` dir.** Training drops
  `<-o>/gcp_run.json` recording the GCS output dir and this run's GCP config
  (`gcp_data_dir`, `gcp_image`, `gcp_project`, `gcp_region`, `gcp_zone`). The
  local mirror excludes `.pth` (the weights stay in GCS), so this marker is how
  a later `hound_eval_folds` finds the weights and inherits the run's GCP
  settings without you re-specifying them — see §9.3. The marker does not
  record `GCPRUNNER_CACHE_PREFIX` / `GCPRUNNER_OUTPUT_PREFIX` /
  `GCPRUNNER_IMAGE_BASE`, so keep those set for downstream commands.
- On-demand VMs by default; pass `--gcp_provisioning spot` or
  `flex_start` — the resume machinery below makes either viable.

#### 9.2.1 Crash-resilient resume

Long training jobs crash — Spot preemption, hardware, and Batch's hard 7-day VM
cap all guarantee it. The GCS run dir is the **single source of truth** for
per-fold status, and the command is **idempotent**: re-run it any time and it
picks up where it left off. Two layers cooperate:

1. **Within a job.** The trainer checkpoints every epoch
   (`checkpoint.pth` = model + optimizer + scheduler + epoch + early-stop
   state, written atomically) and `hound_train` syncs the output dir to GCS
   right after each epoch checkpoint — so `checkpoint.pth`, `model_best.pth`,
   `progress.json`, **and `log.txt`** land in the bucket continuously (pull
   `log.txt` to follow along; adjust the upload cadence via steps-per-epoch).
   On start it restores any prior checkpoint from GCS, so a relaunched fold
   resumes mid-training instead of from epoch 0. Batch task-retry
   (`--gcp_retry_count`, default 3) relaunches a crashed/preempted task in place
   on a fresh VM.
2. **Orchestrator.** Every invocation scans the run dir and prints a per-fold
   status table — `complete` / `failed` / `running` / `incomplete (epoch N)` /
   `new` — telling you which prior checkpoints will be picked up. It then loops
   (`--gcp_max_rounds`, default 10): launch the incomplete folds, wait, re-scan,
   relaunch any that died. An active-job check prevents double-launching a fold
   another invocation is already running.

Completion vs failure is explicit: the trainer writes a `COMPLETE` marker on a
clean finish and a **`FAILED`** marker on `loss_nan`. A `FAILED` fold is
**terminal** — the orchestrator does **not** retry it (a `loss_nan` recurs);
it stops and tells you to investigate the params/data. Crashes/preemptions
leave no marker and are retried.

The orchestrator stops if a submitted fold produces no checkpoint, or a
checkpointed fold finishes a round without advancing its recorded epoch.
Missing epoch metadata also stops resubmission when progress cannot be verified.
Inspect the Batch job status and any available `train.err` before re-running.
Submission errors always raise with their original error text after other
submitted jobs finish; they are not treated as training failures.

```bash
# Same command, run again after a crash → resumes incomplete folds, skips done ones.
hound_train_folds --backend gcp \
    --gcp_data_dir gs://my-data-bucket/2-10-26 -q a100-80 -o models_hydra \
    params.json hg38 mm10
```

To train a finished run longer, raise `train_epochs_max` and re-run with
`--extend`. The edited params hash to a new run id, so `--extend` takes the GCS
run dir and image from `<-o>/gcp_run.json` instead; the pinned image beats
`--gcp_branch`. Folds whose `COMPLETE` marker says `max_epochs` below the new
limit count as incomplete and resume from `checkpoint.pth`; early-stopped folds
stay complete.

`--transfer MODELS_DIR` stages the local `model_best.pth` files and injects
each fold's pretrained model path into its staged parameters.

### 9.3 Cross-fold eval

`hound_eval_folds --backend gcp` mounts the dataset from a regional bucket via
GCSFuse (like training), evaluates each replicate's `model_best.pth` on its
folds, writes per-fold `acc.txt` to GCS, and mirrors the results back to the
local `-o` dir. On-demand by default; pass `--gcp_provisioning spot` for Spot.

How it finds the **weights** depends on where they live, and the
fold-evaluation split is read from each model's `folds.json` so eval always
matches how the model was actually trained (in particular `valid_fold = (fold +
1 + cross) % num_folds`, which is _not_ `fold + 1` once `cross > 0`).

**Marker mode — models trained on GCP (the common case).** A GCP training run
leaves `<-o>/gcp_run.json` in the local models dir (§9.2). When eval sees that
marker it:

- reads the weights **straight from the recorded GCS output dir** (mounted
  read-only at `/workspace/models`) — no download, no re-upload, no content-cache
  round-trip. This matters because the train mirror excludes `.pth`, so the
  local tree has `folds.json` but not the weights;
- **inherits** `--gcp_data_dir`, `--gcp_image`, `--gcp_project`, `--gcp_zone`,
  and `--gcp_region` from the marker (anything you pass explicitly wins;
  `--gcp_region` inherits when unset);
- writes `acc.txt` to a separate eval prefix and fetches it back next to the
  models locally.

So evaluating a model set you just trained is essentially flag-free:

```bash
cd my_experiment      # contains models_gcp/gcp_run.json
hound_eval_folds \
    --backend gcp \
    -q l4 \
    -o models_gcp \
    params_hydra.json hg38 mm10
```

`--gcp_data_dir/_image/_project/_region` come from `models_gcp/gcp_run.json`.
Results land under `models_gcp/f*/eval*/`. Re-running skips folds whose
`acc.txt` already exists in GCS.

**Staged mode — local weights (Slurm-trained, or no marker).** With no marker,
eval falls back to the SNP-style flow: the local models tree (`model_best.pth`
files) and `params.json` are SHA256-staged to `$GCPRUNNER_CACHE_PREFIX/…` and
mounted at `/workspace/cache`. Use this when the weights are actually on disk.

**Common flags:**

- `--test` / `--valid` evaluate only the held-out test / validation fold per
  model (from `folds.json`); the default evaluates every fold.
- `--save` / `--aggregate_genes` need ~60 GB RAM — use `-q l4-large`.
- `--spec` (specificity) is **auto-pinned to `-q l4-large`** (g2-standard-32,
  32 vCPU, ~122 GB) regardless of `-q`, since quantile normalization is RAM- and
  CPU-hungry; it runs with `--ncpus 32`. Your `-q` still governs the lighter
  coverage-eval jobs.
- On GCP, `mem` is the VM's hard RAM (the task owns the whole VM), not a soft
  Slurm request — size it via `-q`, not a mem flag. Eval warns if the chosen
  profile is too small.
- `--gcp_fetch_output OTHER_DIR` redirects the local results mirror;
  `--gcp_fetch_output ""` skips it (results stay in GCS only).

### 9.4 ISM

ISM fold scripts stage local inputs and write results to GCS, like SNP scoring.
They default to on-demand; pass `--gcp_provisioning spot` for Spot. Gradient and distillation
fold scripts support only `--backend local` and `--backend slurm`.

### 9.5 Cost watchdog

After a session, sanity-check that nothing is left running:

```python
import gcprunner
for j in gcprunner.list_active_jobs("my-gcp-project", "us-central1"):
    print(j)
# [] means clean — no idle billing.
```

---

## 10. Programmatic API

The same `Job` and `multi_run` you'd use in scripts. Equivalent to
`slurmrunner` line-for-line:

```python
import gcprunner
from gcprunner.batch_spec import DataMount

jobs = []
for chunk_i in range(num_chunks):
    cmd = f"hound_snp ... --chunk {chunk_i}"
    jobs.append(
        gcprunner.Job(
            cmd=cmd,
            name=f"snp-job{chunk_i}",
            queue="l4",
            gpu=1,
            cpu=4,
            mem=30000,
            time="2:00:00",
            project="my-gcp-project",
            region="us-central1",
            image="us-central1-docker.pkg.dev/my-gcp-project/baskerville/baskerville:latest",
            data_mounts=[
                DataMount("gs://my-bucket/refs", "/workspace/data", mode="fuse"),
            ],
            output_dir_gcs=f"gs://my-bucket/runs/2026-05/job{chunk_i}",
            out_file=f"gs://my-bucket/runs/2026-05/job{chunk_i}.out",
            err_file=f"gs://my-bucket/runs/2026-05/job{chunk_i}.err",
            provisioning="spot",
        )
    )

gcprunner.multi_run(jobs, max_proc=64, verbose=True, update_sleep=60)
```

`Job` is constructed eagerly (cheap, no SDK calls). The SDK is only imported
when `.launch()` runs. To inspect what would be submitted without actually
submitting:

```python
import json
print(json.dumps(jobs[0].to_batch_dict(), indent=2))
```

This is also how the unit tests work — they build dicts and assert their
structure, with no SDK or auth needed.

---

## 11. Logging & debugging

### 11.1 Where logs live

Three places, by design:

1. **Cloud Logging** (primary). Batch streams stdout/stderr automatically.
   View in the GCP console: Batch → Jobs → your job → Logs. Or via CLI:
   ```bash
   gcloud logging read \
       'resource.type="batch.googleapis.com/Job" AND labels.job_uid="<uid>"' \
       --limit 200
   ```
2. **GCS log copies** (optional). If you set `out_file`/`err_file` to
   `gs://…` paths (or use `--gcp_output_dir`), the entrypoint also `tee`s
   stdout/stderr to local files and uploads them in the EXIT trap. Handy
   for quick `gcloud storage cat`.
3. **Batch job record**. `gcloud batch jobs describe <name> --location us-central1`
   shows state, exit code, and timing.

### 11.2 Debugging a failed job

The cycle:

1. `update_status()` returns `FAILED`. Check Cloud Logging for the
   container's stderr — the user command's traceback is there.
2. If the entrypoint itself failed (gcsfuse mount, staging rsync), you'll
   see `[gcprunner] …` log lines from `entry.sh`. The trap still ran, so
   partial outputs/logs were uploaded if possible.
3. If the job never made it to RUNNING (e.g. quota error, image pull
   failure), the failure is on the Batch job status itself, not in
   container logs. `gcloud batch jobs describe` is the right tool.

### 11.3 Local dry-run

You can simulate the entrypoint on a local Linux machine without GCP:

```bash
# To skip the uploads, shadow gcloud with a no-op earlier on PATH. An alias
# will not do: entry.sh runs as a child process, which does not inherit one.
mkdir -p /tmp/fakebin
printf '#!/bin/sh\nexit 0\n' > /tmp/fakebin/gcloud && chmod +x /tmp/fakebin/gcloud

PATH=/tmp/fakebin:$PATH \
GCPRUNNER_OUTPUT_DIR_LOCAL=/tmp/out \
GCPRUNNER_OUTPUT_DIR_GCS=gs://my-bucket/test \
bash src/gcprunner/entry.sh /bin/bash -c "echo hello && touch /tmp/out/marker"
```

---

## 12. Failure behavior

Input mount or staging failures stop the entrypoint before the user command.
On ordinary exit, cleanup attempts to upload outputs and logs. A failed output
sync makes an otherwise successful job fail; an existing command failure keeps
its original exit code. Optional log uploads remain best-effort.

An EXIT trap cannot guarantee uploads after SIGKILL, preemption, or VM loss.
Training also syncs checkpoints after each epoch so it can resume without an
exit-time upload. Batch retries failed tasks according to the job's retry count;
the training orchestrator can then resume checkpointed folds in another round.

Stopping the orchestrator does not stop its submitted jobs. For training, stop
the orchestrator before running `hound_train_folds --conclude` to cancel jobs
and download the run. Cancellation does not guarantee a final checkpoint.

---

## 13. Gotchas

- **`--privileged` container**. Required for `gcsfuse` (it touches
  `/dev/fuse`). If your image is hardened or your org disallows privileged
  containers, switch all data mounts to `stage_local` and remove the
  `--privileged` from `batch_spec.py`.
- **GPU drivers**. The instance policy sets `install_gpu_drivers: True`, so
  Batch installs NVIDIA drivers on the host before starting the container.
  Adds ~30s to job startup.
- **Region/zone for the GPU**. Jobs allocate within the resolved region.
  Use `--gcp_zone` or `GCPRUNNER_ZONE` to pin a zone; GPU profiles do not
  determine placement.
- **No batched status RPC**. `multi_update_status` iterates per-job. With
  hundreds of concurrent jobs that's still fine (Batch quota is generous),
  but if you push into the thousands consider lengthening `update_sleep`.
- **`time` parsing**. Slurmrunner accepts `7-0:0:0`. We parse that.
  `7d`, `1w`, `48h` etc. are **not** accepted — keep the slurmrunner format.
- **Image versioning**. Images come from whatever you push to your registry
  (§8.2). Pin `GCPRUNNER_IMAGE` to a specific tag for reproducibility — don't rely
  on a moving `:latest` between runs.
- **`--gcp_branch` semantics**. There is no per-branch tag; `--gcp_branch`
  resolves to the newest commit on that branch with a `commit-<sha>` image and
  pins by digest. The concrete digest is recorded in the run marker so eval
  inherits the _exact_ training image — the branch name is deliberately not
  persisted (persisting it would let eval drift to a newer commit). Passing
  `--gcp_branch` at eval time overrides the inherited training image (a warning
  is printed). With no image args, training defaults to the latest built commit
  on `main`, which needs `gcloud` auth/network at launch; `--gcp_image` or
  `GCPRUNNER_IMAGE` bypasses the lookup.
- **Local execution is `--backend local`**. In-process execution via
  `utils.exec_par` is now selected with `--backend local`, alongside
  `--backend slurm` and `--backend gcp` (the former standalone `--local`
  flag was folded into `--backend`). `--backend slurm` still falls back to
  local automatically when `slurmrunner` is not installed.
- **Spot preemption mid-training**. Training defaults to on-demand
  precisely because Spot preemption mid-epoch wastes more than it saves —
  checkpoints are written only at epoch boundaries. Prefer
  `--gcp_provisioning flex_start`: it is cheaper than Spot and cannot be
  reclaimed inside its 7-day window.
- **GCS egress**. Cross-region reads cost money. Keep your bucket, your
  Artifact Registry repo, and your Batch region in the same continent (or
  same region).

---

## 14. Comparison with slurmrunner

| Concern             | slurmrunner                         | gcprunner                                          |
| ------------------- | ----------------------------------- | -------------------------------------------------- |
| Submit              | `sbatch` via shell                  | `batch_v1.BatchServiceClient.create_job`           |
| Status              | `sacct` parse                       | `BatchServiceClient.get_job`                       |
| GPU selection       | Partition + `--gres` translation    | Machine type + accelerator type in instance policy |
| Resource units      | CPU count, mem MB, time `D-H:M:S`   | Same — translated internally                       |
| Concurrency control | `multi_run(max_proc=…)` loop        | Same — identical loop, different status backend    |
| Dependencies        | None                                | None                                               |
| Job arrays          | None — caller builds N jobs         | Same                                               |
| Failure semantics   | Job stays in `sacct`; node freed    | VM deleted; job record stays in Batch (free)       |
| Cost when idle      | Cluster bills regardless            | $0                                                 |
| Long-running cap    | Partition `--time` (sometimes days) | 7-day Batch VM cap                                 |
| Logs                | Files on shared FS                  | Cloud Logging + optional GCS copies                |

The public API surface (`Job`, `multi_run`, `multi_update_status`,
`get_job_summary`) is identical so that the `hound_*_folds.py` scripts can
swap backends with a one-line shim.

---

## 15. Files referenced

- Package: [`src/gcprunner/`](../src/gcprunner/)
- Dockerfile: [`dockerfiles/baskerville.Dockerfile`](../dockerfiles/baskerville.Dockerfile)
- Tests: [`src/tests/test_gcprunner.py`](../src/tests/test_gcprunner.py)
- Fold scripts: [`src/baskerville/scripts/hound_*_folds.py`](../src/baskerville/scripts/)
- GCS helpers reused by the fold scripts: [`src/baskerville/helpers/gcs_utils.py`](../src/baskerville/helpers/gcs_utils.py)
