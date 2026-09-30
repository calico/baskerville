"""High-level SNP ref/alt coverage visualization."""

from __future__ import annotations

import json
import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..gene import Gene
from ._style import ALT_COLOR, REF_COLOR, bin_centers_bp, despine
from .annotation import draw_gene_track, mark_variant
from .coverage import plot_coverage_pair


def _trunk_stride_crop(model: dict) -> tuple[int, int]:
    """Derive (output_stride, output_crop_bp) from the trunk block defs.

    Pure-dict mirror of SeqNNMod.build_block's stride/crop bookkeeping
    (seqnn.py): stride accumulates pool_size/stride (each raised to the
    block's repeat) and is halved by Unet upsamples; crop accumulates
    crop_size scaled by the stride in effect after that block. The model's
    global_vars (act_func/norm_type/num_species) never affect geometry, so
    parsing the block dicts is exact and needs no torch.
    """
    stride = 1
    crop_bp = 0
    for blk in model["trunk"]:
        name = blk.get("name", "")
        rep = blk.get("repeat", 1)
        if "pool_size" in blk:
            stride *= blk["pool_size"] ** rep
        elif "stride" in blk:
            stride *= blk["stride"] ** rep
        elif "Unet" in name:
            stride //= 2**rep
        if "crop_size" in blk:
            crop_bp += stride * blk["crop_size"]
    return stride, crop_bp


def seq_window_for_variant(
    params,
    variant_pos_bp: int,
) -> dict:
    """Return geometry of the model-output window centered on a variant.

    `params` can be a path to a params.json or an already-loaded dict.
    `seq_length` is read from the 'model' subdict (or top level). The
    output stride and crop are taken from explicit `output_stride` /
    `output_crop_bp` keys when present, otherwise derived from the trunk
    block definitions so the result matches what `score_snps` used to
    place prediction bins.

    Returns a dict with: seq_start_bp, bin_size, n_bins.
    """
    if isinstance(params, str):
        with open(params) as f:
            params = json.load(f)

    model = params.get("model", params)

    def _get(key, default=None):
        if key in model:
            return model[key]
        if key in params:
            return params[key]
        return default

    seq_length = _get("seq_length")
    if seq_length is None:
        raise ValueError("seq_length not found in params")

    output_stride = _get("output_stride")
    output_crop_bp = _get("output_crop_bp")
    if output_stride is None or output_crop_bp is None:
        d_stride, d_crop = _trunk_stride_crop(model)
        if output_stride is None:
            output_stride = d_stride
        if output_crop_bp is None:
            output_crop_bp = d_crop

    n_bins = (seq_length - 2 * output_crop_bp) // output_stride
    seq_start_bp = variant_pos_bp - seq_length // 2 + output_crop_bp
    return {
        "seq_start_bp": int(seq_start_bp),
        "bin_size": int(output_stride),
        "n_bins": int(n_bins),
    }


def _track_label(targets_df: pd.DataFrame, idx: int, max_chars: int = 22) -> str:
    """Short y-axis label: prefer `description`, drop the ``ASSAY:`` prefix,
    truncate to the first `max_chars` characters."""
    row = targets_df.iloc[idx]
    text = None
    for col in ("description", "identifier", "name"):
        if col in targets_df.columns and pd.notna(row[col]):
            text = str(row[col])
            break
    if text is None:
        return f"track {idx}"
    # strip a leading "RNA:" / "ATAC:" / "DNASE:" ... assay prefix
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def plot_snp(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    *,
    variant_pos_bp: int,
    seq_start_bp: int,
    track_indices: Sequence[int],
    targets_df: pd.DataFrame,
    bin_size: int = 32,
    gene: Gene | None = None,
    isoforms: Sequence[Gene] | None = None,
    other_genes: Sequence[Gene] | None = None,
    window_bp: tuple[int, int] | None = None,
    normalize_counts: bool = False,
    log_scale: bool = False,
    same_scale: bool = True,
    show_diff: bool = False,
    label_max_chars: int = 22,
    figsize_per_track: tuple[float, float] = (12.0, 1.8),
    gene_panel_height: float = 1.2,
    colors: tuple[str, str] = (REF_COLOR, ALT_COLOR),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot ref vs alt coverage for one variant across selected tracks.

    Inputs are post-untransform predictions of shape `[n_targets, n_bins]`
    (e.g. from `dataset.untransform_preds(model_output.coverage.squeeze(0),
    targets_df)`).

    Returns `(fig, axes)` where `axes[-1]` is the gene-annotation panel
    (only present when `gene is not None`).
    """
    ref_preds = np.asarray(ref_preds)
    alt_preds = np.asarray(alt_preds)
    if ref_preds.shape != alt_preds.shape:
        raise ValueError(
            f"ref/alt shape mismatch: {ref_preds.shape} vs {alt_preds.shape}"
        )
    n_targets, n_bins = ref_preds.shape

    end_bp = seq_start_bp + n_bins * bin_size
    if window_bp is not None:
        xlim = (max(seq_start_bp, window_bp[0]), min(end_bp, window_bp[1]))
    else:
        xlim = (seq_start_bp, end_bp)

    n_tracks = len(track_indices)
    has_gene = gene is not None
    diff_ratio = 0.5  # diff panel height relative to a coverage panel

    # row layout: [cov, (diff)] per track, then optional gene panel
    height_ratios = []
    for _ in range(n_tracks):
        height_ratios.append(1.0)
        if show_diff:
            height_ratios.append(diff_ratio)
    if has_gene:
        height_ratios.append(gene_panel_height / figsize_per_track[1])
    n_rows = len(height_ratios)

    per_track_h = figsize_per_track[1] * (1.0 + (diff_ratio if show_diff else 0.0))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(
            figsize_per_track[0],
            per_track_h * n_tracks + (gene_panel_height if has_gene else 0),
        ),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.25},
        squeeze=False,
    )
    axes = axes[:, 0]

    # map each track to its coverage axis (and diff axis when show_diff)
    step = 2 if show_diff else 1
    cov_axes = [axes[i * step] for i in range(n_tracks)]
    diff_axes = [axes[i * step + 1] for i in range(n_tracks)] if show_diff else None

    # y-limits must reflect only the bins inside the visible x-window:
    # matplotlib would otherwise autoscale to the whole track, and a large
    # out-of-window peak (e.g. a distant gene body) flattens the in-window
    # signal to an invisible sliver.
    x_bp = bin_centers_bp(n_bins, bin_size, seq_start_bp)
    wmask = (x_bp >= xlim[0]) & (x_bp <= xlim[1])
    if not wmask.any():
        wmask = np.ones(n_bins, dtype=bool)

    def _disp_pair(t_idx):
        """ref/alt for a track with the same transforms plot_coverage_pair applies."""
        r = np.asarray(ref_preds[t_idx], dtype=float)
        a = np.asarray(alt_preds[t_idx], dtype=float)
        if normalize_counts:
            asum = a.sum()
            if asum > 0:
                a = a * (r.sum() / asum)
        if log_scale:
            r = np.log2(np.clip(r, 0, None) + 1.0)
            a = np.log2(np.clip(a, 0, None) + 1.0)
        return r, a

    def _win_max(t_idx):
        r, a = _disp_pair(t_idx)
        return float(max(r[wmask].max(), a[wmask].max()))

    def _win_absdiff(t_idx):
        r, a = _disp_pair(t_idx)
        d = np.abs(a - r)[wmask]
        return float(d.max()) if d.size else 0.0

    global_max = None
    global_diff = None
    if same_scale and n_tracks:
        global_max = max(_win_max(t) for t in track_indices) * 1.05
        if show_diff:
            global_diff = max(_win_absdiff(t) for t in track_indices) * 1.05

    for row, t_idx in enumerate(track_indices):
        ax = cov_axes[row]
        plot_coverage_pair(
            ax,
            ref_preds[t_idx],
            alt_preds[t_idx],
            bin_size=bin_size,
            start_bp=seq_start_bp,
            colors=colors,
            alpha=0.5,
            log_scale=log_scale,
            normalize_counts=normalize_counts,
        )
        ax.set_ylabel(
            _track_label(targets_df, t_idx, max_chars=label_max_chars), fontsize=8
        )
        ymax = global_max if global_max is not None else _win_max(t_idx) * 1.05
        if ymax and ymax > 0:
            ax.set_ylim(0, ymax)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=False)

        if show_diff:
            dax = diff_axes[row]
            r, a = _disp_pair(t_idx)
            d = a - r
            # alt>ref filled in the ALT color, alt<ref in the REF color
            dax.fill_between(
                x_bp, 0, d, where=d >= 0, color=colors[1], linewidth=0, alpha=0.7
            )
            dax.fill_between(
                x_bp, 0, d, where=d < 0, color=colors[0], linewidth=0, alpha=0.7
            )
            dax.axhline(0, color="k", linewidth=0.6)
            dmax = (
                global_diff if global_diff is not None else _win_absdiff(t_idx) * 1.05
            )
            if dmax and dmax > 0:
                dax.set_ylim(-dmax, dmax)
            dax.set_ylabel("Δ alt−ref", fontsize=7)
            despine(dax)

    # gene panel
    if has_gene:
        draw_gene_track(
            axes[-1],
            gene,
            isoforms=isoforms,
            other_genes=other_genes,
            xlim=xlim,
        )

    # variant marker on every panel
    mark_variant(list(axes), variant_pos_bp)

    # prediction-span boundaries: when the visible window extends past the
    # edge of the model's prediction range, draw thin gray lines so it's
    # obvious where coverage data actually exists.
    pred_lo, pred_hi = int(seq_start_bp), int(end_bp)
    for boundary in (pred_lo, pred_hi):
        if xlim[0] <= boundary <= xlim[1]:
            for ax in axes:
                ax.axvline(
                    boundary,
                    color="0.5",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=4,
                )

    # shared x limits + label only on bottom
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(f"genomic position (bp)")
    axes[0].set_xlim(*xlim)

    if title:
        fig.suptitle(title, fontsize=10)

    # the slim gene panel + shared-x stack isn't tight_layout-friendly;
    # the result is fine, so just suppress the cosmetic warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible",
            category=UserWarning,
        )
        fig.tight_layout()
    return fig, axes


# default zoom levels for interactive_plot_snp: half-widths in bp around
# the variant. None = use the full prediction span (no window clipping).
_DEFAULT_WINDOW_OPTIONS: tuple[tuple[str, int | None], ...] = (
    ("2 kb", 1_000),
    ("5 kb", 2_500),
    ("20 kb", 10_000),
    ("100 kb", 50_000),
    ("500 kb", 250_000),
    ("full", None),
)


def interactive_plot_snp(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    *,
    variant_pos_bp: int,
    seq_start_bp: int,
    targets_df: pd.DataFrame,
    bin_size: int = 32,
    gene: Gene | None = None,
    isoforms: Sequence[Gene] | None = None,
    other_genes: Sequence[Gene] | None = None,
    gene_trees: dict | None = None,
    chrom: str | None = None,
    max_other_genes: int = 8,
    default_group: str | None = "gtex",
    window_options: Sequence[tuple[str, int | None]] = _DEFAULT_WINDOW_OPTIONS,
    default_window_label: str = "100 kb",
    group_col: str = "group",
    title: str | None = None,
    **plot_snp_kwargs,
):
    """Interactive ref/alt browser: group + zoom dropdowns + ipympl pan/zoom.

    Wraps :func:`plot_snp` with a group dropdown (built from
    ``targets_df[group_col]``) and a window-size dropdown. The figure itself
    is an ipympl canvas, so the matplotlib toolbar (drag-pan, scroll-zoom,
    home) is wired up natively.

    If ``gene_trees`` (from ``Transcriptome.gene_trees()``) and ``chrom``
    are provided, the gene annotation panel is recomputed per zoom level:
    the nearest gene to the variant becomes the main row, and other genes
    overlapping the visible window are drawn beneath (capped at
    ``max_other_genes``). This is how zooming out reveals neighbors like
    STXBP5 that aren't visible in the closer view.

    The caller is responsible for activating the ipympl backend once in the
    notebook with ``%matplotlib widget`` before invoking this function.

    Returns a :class:`ipywidgets.VBox` ready to ``display(...)``.
    """
    from ..gene import find_overlapping_genes
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    if group_col not in targets_df.columns:
        raise ValueError(f"targets_df missing {group_col!r} column")

    group_counts = targets_df[group_col].value_counts()
    group_opts: list[tuple[str, str]] = [(f"ALL ({len(targets_df)})", "__ALL__")]
    group_opts += [(f"{g} ({n})", g) for g, n in group_counts.items()]
    if default_group is not None and default_group in set(targets_df[group_col]):
        initial_group = default_group
    else:
        initial_group = group_opts[0][1]

    window_opts = list(window_options)
    label_to_value = {lbl: val for lbl, val in window_opts}
    if default_window_label not in label_to_value:
        default_window_label = window_opts[0][0]

    group_dd = widgets.Dropdown(
        options=group_opts, value=initial_group, description="group"
    )
    window_dd = widgets.Dropdown(
        options=window_opts,
        value=label_to_value[default_window_label],
        description="zoom",
    )
    out = widgets.Output()
    state: dict = {"fig": None}

    n_targets, n_bins = np.asarray(ref_preds).shape
    span0 = int(seq_start_bp)
    span1 = span0 + n_bins * bin_size

    def _render(*_):
        g = group_dd.value
        if g == "__ALL__":
            idx = list(range(n_targets))
        else:
            mask = targets_df[group_col].to_numpy() == g
            idx = np.flatnonzero(mask).tolist()
        if not idx:
            with out:
                clear_output(wait=True)
                print(f"(no tracks in group {g!r})")
            return

        half = window_dd.value
        if half is None:
            window_bp = (span0, span1)
        else:
            window_bp = (
                max(span0, variant_pos_bp - half),
                min(span1, variant_pos_bp + half),
            )

        # gene panel: when gene_trees+chrom are given, refresh per-window so
        # zoom-out reveals neighboring genes. Pick the gene closest to the
        # variant as the main row; the rest become "other_genes".
        cur_gene = gene
        cur_others = list(other_genes) if other_genes else []
        if gene_trees is not None and chrom is not None:
            hits = find_overlapping_genes(gene_trees, chrom, window_bp[0], window_bp[1])
            if hits:

                def _d(g):
                    s, e = g.span()
                    return (
                        0
                        if s <= variant_pos_bp <= e
                        else min(abs(s - variant_pos_bp), abs(e - variant_pos_bp))
                    )

                hits = sorted(hits, key=_d)
                cur_gene = hits[0]
                cur_others = hits[1 : 1 + max_other_genes]

        # release the previous ipympl canvas so figures don't leak
        if state["fig"] is not None:
            plt.close(state["fig"])

        # plt.ioff() so the new figure doesn't auto-show before our display()
        with plt.ioff():
            fig, _ = plot_snp(
                ref_preds,
                alt_preds,
                variant_pos_bp=variant_pos_bp,
                seq_start_bp=seq_start_bp,
                track_indices=idx,
                targets_df=targets_df,
                bin_size=bin_size,
                gene=cur_gene,
                isoforms=isoforms,
                other_genes=cur_others,
                window_bp=window_bp,
                title=title,
                **plot_snp_kwargs,
            )
        # let the ipympl canvas stretch to fill the cell width
        fig.canvas.layout.width = "100%"
        fig.canvas.header_visible = False
        state["fig"] = fig
        with out:
            clear_output(wait=True)
            display(fig.canvas)

    group_dd.observe(_render, names="value")
    window_dd.observe(_render, names="value")
    _render()

    return widgets.VBox([widgets.HBox([group_dd, window_dd]), out])
