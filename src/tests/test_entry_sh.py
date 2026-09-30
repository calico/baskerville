"""Behavioral tests for the gcprunner container entrypoint (entry.sh).

These exercise the stdout/stderr capture + upload path without touching GCP: a
fake ``gcloud`` on PATH records invocations and copies the local log file to a
capture dir (translating the gs:// dest to a local path). This guards against
the process-substitution race that used to truncate crash logs (losing the
traceback tail) and drop instant-failure logs entirely.
"""

import os
import stat
import subprocess

import gcprunner
import pytest

ENTRY_SH = os.path.join(os.path.dirname(gcprunner.__file__), "entry.sh")


def _write_gcloud_stub(bin_dir, capture_dir):
    """Fake gcloud: `gcloud storage cp <src> gs://.../<name>` copies <src> to
    capture_dir/<name>, and `gcloud storage rsync` exits with $RSYNC_EXIT.

    Exits 64 on an rsync invocation that has lost `--recursive`, so a mangled
    flag fails a test instead of passing silently. Only a real `gcloud` can
    tell us the flags are *accepted*; this just pins what entry.sh sends.
    """
    stub = os.path.join(bin_dir, "gcloud")
    with open(stub, "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "set -u\n"
            'if [[ "$2" == "rsync" ]]; then\n'
            '  [[ "$3" == "--recursive" ]] || exit 64\n'
            '  exit "${RSYNC_EXIT:-0}"\n'
            "fi\n"
            'if [[ "$2" == "cp" ]]; then\n'
            '  src="$3"; dest="$4"\n'
            '  cp "$src" "%s/$(basename "$dest")"\n'
            "fi\n"
            "exit 0\n" % capture_dir
        )
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_entry(tmp_path, cmd_args, **env_overrides):
    """Run entry.sh with a gcloud stub and gs:// log dests; return (proc, capture_dir)."""
    bin_dir = tmp_path / "bin"
    capture_dir = tmp_path / "captured"
    bin_dir.mkdir()
    capture_dir.mkdir()
    _write_gcloud_stub(str(bin_dir), str(capture_dir))

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GCPRUNNER_STDOUT_GCS"] = "gs://fake/train.out"
    env["GCPRUNNER_STDERR_GCS"] = "gs://fake/train.err"
    env.pop("GCPRUNNER_OUTPUT_DIR_GCS", None)
    env.pop("GCPRUNNER_DATA_MOUNTS", None)
    env["GCPRUNNER_OUTPUT_DIR_LOCAL"] = str(tmp_path / "out")

    env.update(env_overrides)
    proc = subprocess.run(
        ["bash", ENTRY_SH, *cmd_args],
        env=env,
        capture_output=True,
        text=True,
    )
    return proc, capture_dir


def test_entry_sh_syntax():
    subprocess.run(["bash", "-n", ENTRY_SH], check=True)


def test_full_output_and_tail_uploaded(tmp_path):
    """A long-running crash uploads complete logs — including the tail traceback."""
    proc, capture_dir = _run_entry(
        tmp_path,
        [
            "bash",
            "-c",
            "for i in $(seq 1 5000); do echo out $i; done; "
            "echo TAIL_MARKER; echo boom >&2; exit 3",
        ],
    )
    assert proc.returncode == 3  # command exit status propagated

    out = (capture_dir / "train.out").read_text()
    err = (capture_dir / "train.err").read_text()
    assert "out 1\n" in out
    # The tail must survive — this is the regression the FIFO+wait fix addresses.
    assert out.rstrip().endswith("TAIL_MARKER")
    assert out.count("out ") == 5000
    assert "boom" in err


def test_instant_failure_still_uploads_logs(tmp_path):
    """A job that dies before emitting output still uploads (empty) log files."""
    proc, capture_dir = _run_entry(tmp_path, ["bash", "-c", "exit 7"])
    assert proc.returncode == 7
    # Both files exist (pre-created) and were uploaded even with no output.
    assert (capture_dir / "train.out").is_file()
    assert (capture_dir / "train.err").is_file()
    assert (capture_dir / "train.out").read_text() == ""


@pytest.mark.parametrize("command_exit, expected", [(0, 1), (7, 7)])
def test_output_upload_failure(tmp_path, command_exit, expected):
    proc, capture = _run_entry(
        tmp_path,
        ["bash", "-c", f"echo done; exit {command_exit}"],
        GCPRUNNER_OUTPUT_DIR_GCS="gs://fake/output",
        RSYNC_EXIT="9",
    )
    assert proc.returncode == expected
    assert (capture / "train.out").read_text() == "done\n"


def test_staging_failure_does_not_run_command(tmp_path):
    marker = tmp_path / "ran"
    proc, capture = _run_entry(
        tmp_path,
        ["touch", str(marker)],
        GCPRUNNER_DATA_MOUNTS=f"stage_local\tgs://fake/input\t{tmp_path / 'input'}",
        RSYNC_EXIT="9",
    )
    assert proc.returncode == 9
    assert not marker.exists()
    assert (capture / "train.err").exists()


@pytest.mark.parametrize("failure", ["mktemp", "mkdir"])
def test_setup_failure_runs_cleanup(tmp_path, failure):
    marker = tmp_path / "ran"
    blocked = tmp_path / "not-a-directory"
    blocked.touch()
    overrides = (
        {"TMPDIR": str(blocked)}
        if failure == "mktemp"
        else {"GCPRUNNER_OUTPUT_DIR_LOCAL": str(blocked)}
    )
    proc, capture = _run_entry(tmp_path, ["touch", str(marker)], **overrides)
    assert proc.returncode != 0
    assert not marker.exists()
    assert "running cleanup" in proc.stderr
    assert "unbound variable" not in proc.stderr
    if failure == "mkdir":
        assert (capture / "train.out").exists()
        assert (capture / "train.err").exists()
