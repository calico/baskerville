#!/usr/bin/env bash
# gcprunner container entrypoint.
#
# Wraps the user command with:
#   1. Optional GCSFuse read-only mounts (mode=fuse)
#   2. Optional gcloud storage rsync to local SSD (mode=stage_local)
#   3. Run the user command, tee'ing stdout/stderr to local logs
#   4. On EXIT (success OR crash): sync the local output dir to GCS and
#      upload the captured logs. SIGKILL or VM loss can prevent cleanup.
#
# Inputs (env vars set by gcprunner.batch_spec.build_batch_job_dict):
#   GCPRUNNER_DATA_MOUNTS   - newline-separated  "<mode>\t<gcs_uri>\t<local_path>"
#   GCPRUNNER_OUTPUT_DIR_LOCAL - local dir whose contents get uploaded
#   GCPRUNNER_OUTPUT_DIR_GCS   - gs:// dir to upload to (optional)
#   GCPRUNNER_STDOUT_GCS       - gs:// path for stdout copy (optional)
#   GCPRUNNER_STDERR_GCS       - gs:// path for stderr copy (optional)
#
# Note: stdout/stderr also go to Cloud Logging via Batch — these uploads are
# a redundant copy at the user-supplied path.

set -euo pipefail

OUTPUT_DIR_LOCAL="${GCPRUNNER_OUTPUT_DIR_LOCAL:-/workspace/out}"

declare -a FUSE_MOUNTS=()

cleanup() {
  local rc=$?
  echo "[gcprunner] entrypoint exit=${rc}, running cleanup" >&2

  if [[ -n "${GCPRUNNER_OUTPUT_DIR_GCS:-}" && -d "${OUTPUT_DIR_LOCAL}" ]]; then
    echo "[gcprunner] syncing ${OUTPUT_DIR_LOCAL} -> ${GCPRUNNER_OUTPUT_DIR_GCS}" >&2
    gcloud storage rsync --recursive "${OUTPUT_DIR_LOCAL}" "${GCPRUNNER_OUTPUT_DIR_GCS}" || {
      [[ "$rc" != 0 ]] || rc=1
    }
  fi

  if [[ -n "${GCPRUNNER_STDOUT_GCS:-}" && -f "${STDOUT_LOG:-}" ]]; then
    gcloud storage cp "${STDOUT_LOG}" "${GCPRUNNER_STDOUT_GCS}" || true
  fi
  if [[ -n "${GCPRUNNER_STDERR_GCS:-}" && -f "${STDERR_LOG:-}" ]]; then
    gcloud storage cp "${STDERR_LOG}" "${GCPRUNNER_STDERR_GCS}" || true
  fi

  # ${a[@]+...}: bash < 4.4 treats an empty array as unbound under set -u.
  for mp in ${FUSE_MOUNTS[@]+"${FUSE_MOUNTS[@]}"}; do
    fusermount -u "$mp" 2>/dev/null || true
  done
  exit "$rc"
}
trap cleanup EXIT

LOG_DIR="$(mktemp -d -t gcprunner-logs-XXXXXX)"
STDOUT_LOG="${LOG_DIR}/stdout.log"
STDERR_LOG="${LOG_DIR}/stderr.log"
# Create the log files up front so the cleanup upload never skips them: tee
# creates them lazily on first write, so a job that dies before emitting any
# output would otherwise leave no file for the trap's -f guard to find.
: > "${STDOUT_LOG}"
: > "${STDERR_LOG}"
mkdir -p "${OUTPUT_DIR_LOCAL}"

# Set up data mounts. Lines are tab-separated: mode\tgcs_uri\tlocal_path
if [[ -n "${GCPRUNNER_DATA_MOUNTS:-}" ]]; then
  while IFS=$'\t' read -r mode gcs_uri local_path; do
    [[ -z "${mode:-}" ]] && continue
    case "$mode" in
      fuse)
        bucket="${gcs_uri#gs://}"
        bucket="${bucket%%/*}"
        prefix="${gcs_uri#gs://${bucket}}"
        prefix="${prefix#/}"
        mkdir -p "$local_path"
        echo "[gcprunner] gcsfuse mount $gcs_uri -> $local_path" >&2
        if [[ -n "$prefix" ]]; then
          gcsfuse --implicit-dirs -o ro --only-dir "$prefix" "$bucket" "$local_path"
        else
          gcsfuse --implicit-dirs -o ro "$bucket" "$local_path"
        fi
        FUSE_MOUNTS+=("$local_path")
        ;;
      stage_local)
        mkdir -p "$local_path"
        echo "[gcprunner] staging $gcs_uri -> $local_path" >&2
        gcloud storage rsync --recursive "$gcs_uri" "$local_path"
        ;;
      *)
        echo "[gcprunner] unknown mount mode: $mode" >&2
        exit 64
        ;;
    esac
  done <<< "${GCPRUNNER_DATA_MOUNTS}"
fi

# Expose host NVIDIA userspace if Batch bind-mounted it (GPU jobs only).
# Batch's COS-GPU image is in CDI mode where --gpus / --runtime=nvidia
# don't work; the supported path is bind-mounting /var/lib/nvidia/{lib64,bin}
# (see gcprunner.batch_spec). Without /usr/local/nvidia/lib64 on the loader
# path, the PyTorch image can't dlopen libcuda.so and torch.cuda silently
# falls back to CPU.
if [[ -d /usr/local/nvidia/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
  export PATH="/usr/local/nvidia/bin:${PATH}"
  echo "[gcprunner] nvidia libs exposed from host" >&2
fi

# Mirror both streams to the console (→ Cloud Logging) AND to local files.
# We use FIFOs + backgrounded tee rather than process substitution ">(tee …)":
# process substitution gives no reapable PID, so the EXIT trap races the tee
# processes and uploads a half-flushed log — truncating the tail, which is
# exactly the traceback we care about. With real PIDs we can wait for tee to
# drain before the trap runs.
OUT_FIFO="${LOG_DIR}/out.fifo"
ERR_FIFO="${LOG_DIR}/err.fifo"
mkfifo "${OUT_FIFO}" "${ERR_FIFO}"
tee -a "${STDOUT_LOG}" < "${OUT_FIFO}" & TEE_OUT_PID=$!
tee -a "${STDERR_LOG}" < "${ERR_FIFO}" >&2 & TEE_ERR_PID=$!

echo "[gcprunner] running user command" >&2
# "$@" comes from Batch container.commands; we forward it. Capture the exit
# status explicitly so cleanup can preserve it.
rc=0
"$@" > "${OUT_FIFO}" 2> "${ERR_FIFO}" || rc=$?

# The command's exit closed the FIFO write ends → tee sees EOF. Wait for both
# to finish draining and flush to the log files before the EXIT trap uploads.
wait "${TEE_OUT_PID}" "${TEE_ERR_PID}" 2>/dev/null || true
exit "${rc}"
