"""In-silico mutagenesis (ISM) visualization.

`plot_ism_logo` ports an established logo + heatmap pattern, adding an
explicit variant marker. `ism_track_switcher` is a notebook helper providing
dropdowns (grouped by `targets_df.group`) to switch tracks interactively.

`launch_ism_bg` shells out to `hound_ism_snp_folds` via subprocess and
returns the Popen + log path so the long-running job can cook in the
background while the user does other things.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ism_logo(
    scores: np.ndarray,
    seq_1hot: np.ndarray,
    *,
    variant_offset: int | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (16.0, 3.0),
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Render an ISM panel: sequence logo on top, ACGT heatmap below.

    Args:
        scores: shape (mut_len, 4) — per-position per-nucleotide effect.
        seq_1hot: shape (mut_len, 4) — one-hot of the underlying sequence.
        variant_offset: index within mut_len where the variant lies. Draws
            a vertical line if provided. Defaults to None.
        ylim: optional (ymin, ymax) for the logo y-axis.
        title: optional plot title.
        figsize: only used when axes is None.
        axes: optional (logo_ax, heatmap_ax) to draw into; otherwise a new
            figure is created.

    Returns: (fig, (logo_ax, heatmap_ax)).
    """
    import logomaker
    import seaborn as sns

    scores = np.asarray(scores, dtype=float)
    seq_1hot = np.asarray(seq_1hot, dtype=float)
    if scores.shape != seq_1hot.shape:
        raise ValueError(f"scores {scores.shape} vs seq_1hot {seq_1hot.shape} mismatch")

    if axes is None:
        fig, (ax_logo, ax_heat) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
        )
    else:
        ax_logo, ax_heat = axes
        fig = ax_logo.figure

    seq_df = pd.DataFrame(seq_1hot * scores, columns=["A", "C", "G", "T"])
    logomaker.Logo(seq_df, ax=ax_logo)
    if ylim is not None:
        ax_logo.set_ylim(*ylim)
    ax_logo.set_xticks([])

    sns.heatmap(scores.T, center=0, cbar=False, ax=ax_heat)
    ax_heat.set_yticklabels(list("ACGT"), rotation=0)
    ax_heat.set_xlabel("position")

    if variant_offset is not None:
        # logomaker centers each letter on integer index `i`; seaborn's
        # heatmap draws cell `i` spanning [i, i+1] with center at i+0.5.
        # hound_ism_snp's mut window is centered on the variant such that
        # the variant lives one position left of `mut_len // 2`, so the
        # logo vline goes at variant_offset-1 and the heatmap at the cell
        # boundary just left of it (variant_offset-0.5).
        ax_logo.axvline(
            variant_offset - 1,
            color="k",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        ax_heat.axvline(
            variant_offset - 0.5,
            color="k",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )

    if title:
        ax_logo.set_title(title, fontsize=10)

    return fig, (ax_logo, ax_heat)


def ism_track_switcher(
    targets_df: pd.DataFrame,
    ref_seq: np.ndarray,
    alt_seq: np.ndarray,
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    variant_offset: int | None = None,
    group_col: str = "group",
    label_col: str = "identifier",
):
    """Interactive ipywidgets dropdowns: pick a stat, a group, then a track.

    Re-renders two `plot_ism_logo` panels (ref above, alt below) whenever
    any dropdown changes.

    Args:
        scores: maps a stat name (e.g. "logD2", "logSUM") to a
            ``(ref_scores, alt_scores)`` pair, each shaped
            ``(mut_len, 4, num_targets)`` and already ensemble-averaged.
            The stat dropdown is populated from these keys.
        ref_seq, alt_seq: ``(mut_len, 4)`` one-hot sequences.
        variant_offset: index within mut_len where the variant lies
            (typically ``mut_len // 2``); drawn as a vertical line.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    if not scores:
        raise ValueError("scores is empty")
    if group_col not in targets_df.columns:
        raise ValueError(f"targets_df has no {group_col!r} column")
    if label_col not in targets_df.columns:
        raise ValueError(f"targets_df has no {label_col!r} column")

    stat_names = list(scores)
    groups = sorted(targets_df[group_col].dropna().unique().tolist())

    stat_dd = widgets.Dropdown(options=stat_names, description="stat")
    group_dd = widgets.Dropdown(options=groups, description="group")
    track_dd = widgets.Dropdown(description="track")
    out = widgets.Output()

    def _update_tracks(*_):
        g = group_dd.value
        sub = targets_df[targets_df[group_col] == g]
        # value is the positional column into the score arrays' last axis
        opts = [
            (
                f"{row[label_col]}  —  {row.get('description', '')}",
                int(targets_df.index.get_loc(idx)),
            )
            for idx, row in sub.iterrows()
        ]
        track_dd.options = opts
        if opts:
            track_dd.value = opts[0][1]

    def _render(*_):
        col = track_dd.value
        if col is None:
            return
        stat = stat_dd.value
        ref_scores, alt_scores = scores[stat]
        row = targets_df.iloc[col]
        ident, desc = row[label_col], row.get("description", "")
        title_r = f"REF  {ident}  ({desc})  [{stat}]"
        title_a = f"ALT  {ident}  ({desc})  [{stat}]"

        rs = ref_scores[..., col]
        as_ = alt_scores[..., col]
        # Calibrate ylim to the body of the displayed logo (wild-type letter
        # per position = seq_1hot * scores), using the 99th percentile of
        # nonzero magnitudes. A handful of strong-motif outliers would
        # otherwise blow the axis up and flatten the rest to invisibility.
        ref_disp = (ref_seq * rs).ravel()
        alt_disp = (alt_seq * as_).ravel()
        nz = np.abs(np.concatenate([ref_disp, alt_disp]))
        nz = nz[nz > 0]
        if nz.size:
            ymax = float(np.quantile(nz, 0.99)) * 1.1
        else:
            ymax = 1e-6
        ymax = max(ymax, 1e-6)
        ylim = (-ymax, ymax)

        with out:
            clear_output(wait=True)
            fig, _ = plt.subplots(
                4,
                1,
                figsize=(16, 6),
                gridspec_kw={"height_ratios": [2, 1, 2, 1], "hspace": 0.1},
            )
            axes = fig.axes
            plot_ism_logo(
                rs,
                ref_seq,
                variant_offset=variant_offset,
                ylim=ylim,
                title=title_r,
                axes=(axes[0], axes[1]),
            )
            plot_ism_logo(
                as_,
                alt_seq,
                variant_offset=variant_offset,
                ylim=ylim,
                title=title_a,
                axes=(axes[2], axes[3]),
            )
            # display() (not plt.show()) so it renders from the observe
            # callback, where the inline backend's cell flush does not fire
            display(fig)
            plt.close(fig)

    stat_dd.observe(_render, names="value")
    group_dd.observe(_update_tracks, names="value")
    track_dd.observe(_render, names="value")
    _update_tracks()
    _render()

    display(widgets.HBox([stat_dd, group_dd, track_dd]), out)


def launch_ism_bg(
    *,
    params_file: str,
    models_dir: str,
    vcf_file: str,
    out_dir: str,
    targets_file: str,
    genome_fasta: str,
    genes_gtf: str | None = None,
    stats: Sequence[str] = ("cov/logSUM", "cov/logD2"),
    mut_len: int | None = None,
    rc: bool = True,
    parallel: int | None = None,
    queue: str | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[subprocess.Popen, str]:
    """Start `hound_ism_snp_folds` in the background (SLURM dispatch).

    Hydra blocks require GPU, so this always dispatches per-fold jobs to
    SLURM via the folds wrapper. The orchestrator process runs locally
    (the Popen we return) and blocks waiting for SLURM jobs to finish.

    `mut_len` is the centered bp window that gets mutated (hound_ism's
    `-l`); leave it None to use the script default.

    Returns the Popen handle and the path to the captured log file. Logs
    are written to `{out_dir}/ism.log`. Call `check_ism(proc)` to poll.
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ism.log")

    cmd = [
        "hound_ism_snp_folds",
        "-f",
        genome_fasta,
        "-t",
        targets_file,
        "--stats",
        ",".join(stats),
        "-o",
        out_dir,
    ]
    if genes_gtf:
        cmd += ["-g", genes_gtf]
    if mut_len is not None:
        cmd += ["-l", str(mut_len)]
    if rc:
        cmd.append("--rc")
    if parallel is not None:
        cmd += ["-p", str(parallel)]
    if queue is not None:
        cmd += ["-q", queue]
    cmd += list(extra_args)
    cmd += [params_file, models_dir, vcf_file]

    log_f = open(log_path, "w")
    log_f.write(f"# {' '.join(shlex.quote(c) for c in cmd)}\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
    )
    return proc, log_path


def check_ism(proc: subprocess.Popen) -> str:
    """Return a one-line status string for a background ISM job."""
    rc = proc.poll()
    if rc is None:
        return f"running (pid {proc.pid})"
    if rc == 0:
        return f"done (pid {proc.pid})"
    return f"failed rc={rc} (pid {proc.pid})"
