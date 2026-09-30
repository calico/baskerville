"""Plotly-based interactive ref/alt coverage browser.

A full-width, smoothly-zoomable alternative to the matplotlib/ipympl
``interactive_plot_snp``: a Plotly ``FigureWidget`` driven by group + zoom
dropdowns, with a per-zoom gene-annotation panel. The figure autosizes to the
cell width (the main thing the matplotlib canvas couldn't do).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..gene import Gene, find_overlapping_genes
from ._style import (
    ALT_COLOR,
    GENE_COLOR,
    ISOFORM_COLOR,
    REF_COLOR,
    VARIANT_COLOR,
    bin_centers_bp,
)
from .snp import _track_label

# zoom half-widths in bp around the variant; None = full prediction span.
# Adds 50 kb on top of the matplotlib browser's levels (request).
_DEFAULT_WINDOW_OPTIONS: tuple[tuple[str, int | None], ...] = (
    ("2 kb", 1_000),
    ("5 kb", 2_500),
    ("20 kb", 10_000),
    ("50 kb", 25_000),
    ("100 kb", 50_000),
    ("500 kb", 250_000),
    ("full", None),
)


def _rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)' for Plotly fill colors."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plotly_browser_snp(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    *,
    variant_pos_bp: int,
    seq_start_bp: int,
    targets_df: pd.DataFrame,
    bin_size: int = 32,
    gene: Gene | None = None,
    other_genes: Sequence[Gene] | None = None,
    gene_trees: dict | None = None,
    chrom: str | None = None,
    max_other_genes: int = 8,
    default_group: str | None = "gtex",
    window_options: Sequence[tuple[str, int | None]] = _DEFAULT_WINDOW_OPTIONS,
    default_window_label: str = "100 kb",
    group_col: str = "group",
    same_scale: bool = False,
    show_diff: bool = True,
    log_scale: bool = False,
    normalize_counts: bool = False,
    label_max_chars: int = 22,
    row_height_px: int = 150,
    title: str | None = None,
    colors: tuple[str, str] = (REF_COLOR, ALT_COLOR),
):
    """Interactive Plotly ref/alt browser with group + zoom dropdowns.

    Mirrors :func:`baskerville.viz.interactive_plot_snp` but renders a
    Plotly ``FigureWidget`` that fills the cell width and supports native
    zoom/pan/hover. When ``gene_trees`` + ``chrom`` are supplied, the gene
    panel refreshes per zoom (nearest gene to the variant on top, neighbors
    beneath, capped at ``max_other_genes``).

    Returns an ``ipywidgets.VBox`` ready to ``display(...)``. Requires
    ``plotly`` and ``ipywidgets`` in the kernel.
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if group_col not in targets_df.columns:
        raise ValueError(f"targets_df missing {group_col!r} column")

    ref_preds = np.asarray(ref_preds)
    alt_preds = np.asarray(alt_preds)
    if ref_preds.shape != alt_preds.shape:
        raise ValueError(
            f"ref/alt shape mismatch: {ref_preds.shape} vs {alt_preds.shape}"
        )
    n_targets, n_bins = ref_preds.shape
    span0 = int(seq_start_bp)
    span1 = span0 + n_bins * bin_size
    x_bp = bin_centers_bp(n_bins, bin_size, span0)

    # group dropdown options
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

    def _disp_pair(t_idx):
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

    def _render(*_):
        g = group_dd.value
        if g == "__ALL__":
            idx = list(range(n_targets))
        else:
            idx = np.flatnonzero(targets_df[group_col].to_numpy() == g).tolist()

        with out:
            from IPython.display import clear_output

            clear_output(wait=True)
            if not idx:
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
            wmask = (x_bp >= window_bp[0]) & (x_bp <= window_bp[1])
            if not wmask.any():
                wmask = np.ones(n_bins, dtype=bool)

            # gene panel: nearest gene to variant on top, neighbors beneath
            cur_gene = gene
            cur_others = list(other_genes) if other_genes else []
            if gene_trees is not None and chrom is not None:
                hits = find_overlapping_genes(
                    gene_trees, chrom, window_bp[0], window_bp[1]
                )
                if hits:

                    def _d(gg):
                        s, e = gg.span()
                        return (
                            0
                            if s <= variant_pos_bp <= e
                            else min(abs(s - variant_pos_bp), abs(e - variant_pos_bp))
                        )

                    hits = sorted(hits, key=_d)
                    cur_gene = hits[0]
                    cur_others = hits[1 : 1 + max_other_genes]

            has_gene = cur_gene is not None

            # rows: per track [cov, (diff)], then optional gene row
            row_specs = []  # (kind, t_idx)
            for t in idx:
                row_specs.append(("cov", t))
                if show_diff:
                    row_specs.append(("diff", t))
            if has_gene:
                row_specs.append(("gene", None))
            n_rows = len(row_specs)

            row_heights = []
            for kind, _t in row_specs:
                row_heights.append(
                    0.4 if kind == "diff" else (0.5 if kind == "gene" else 1.0)
                )

            fig = make_subplots(
                rows=n_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=min(0.012, 0.2 / max(n_rows - 1, 1)),
                row_heights=row_heights,
            )

            # global y scaling
            gmax = None
            gdiff = None
            if same_scale:
                gmax = (
                    max(
                        float(
                            max(
                                _disp_pair(t)[0][wmask].max(),
                                _disp_pair(t)[1][wmask].max(),
                            )
                        )
                        for t in idx
                    )
                    * 1.05
                )
                if show_diff:
                    gdiff = (
                        max(
                            float(
                                np.abs(_disp_pair(t)[1] - _disp_pair(t)[0])[wmask].max()
                            )
                            for t in idx
                        )
                        * 1.05
                    )

            first_cov = True
            for r, (kind, t) in enumerate(row_specs, start=1):
                if kind == "cov":
                    rr, aa = _disp_pair(t)
                    fig.add_trace(
                        go.Scatter(
                            x=x_bp,
                            y=rr,
                            name="ref",
                            legendgroup="ref",
                            showlegend=first_cov,
                            mode="lines",
                            line=dict(color=colors[0], width=1),
                            fill="tozeroy",
                            fillcolor=_rgba(colors[0], 0.4),
                            hovertemplate="%{x:,.0f}<br>ref %{y:.3g}<extra></extra>",
                        ),
                        row=r,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_bp,
                            y=aa,
                            name="alt",
                            legendgroup="alt",
                            showlegend=first_cov,
                            mode="lines",
                            line=dict(color=colors[1], width=1),
                            fill="tozeroy",
                            fillcolor=_rgba(colors[1], 0.4),
                            hovertemplate="%{x:,.0f}<br>alt %{y:.3g}<extra></extra>",
                        ),
                        row=r,
                        col=1,
                    )
                    first_cov = False
                    ymax = (
                        gmax
                        if gmax is not None
                        else float(max(rr[wmask].max(), aa[wmask].max())) * 1.05
                    )
                    if ymax and ymax > 0:
                        fig.update_yaxes(range=[0, ymax], row=r, col=1)
                    fig.update_yaxes(
                        title_text=_track_label(
                            targets_df, t, max_chars=label_max_chars
                        ),
                        title_font=dict(size=9),
                        row=r,
                        col=1,
                    )
                elif kind == "diff":
                    rr, aa = _disp_pair(t)
                    d = aa - rr
                    fig.add_trace(
                        go.Scatter(
                            x=x_bp,
                            y=np.where(d >= 0, d, 0.0),
                            mode="lines",
                            line=dict(width=0),
                            fill="tozeroy",
                            fillcolor=_rgba(colors[1], 0.6),
                            showlegend=False,
                            hovertemplate="%{x:,.0f}<br>Δ %{y:.3g}<extra></extra>",
                        ),
                        row=r,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_bp,
                            y=np.where(d < 0, d, 0.0),
                            mode="lines",
                            line=dict(width=0),
                            fill="tozeroy",
                            fillcolor=_rgba(colors[0], 0.6),
                            showlegend=False,
                            hovertemplate="%{x:,.0f}<br>Δ %{y:.3g}<extra></extra>",
                        ),
                        row=r,
                        col=1,
                    )
                    dmax = (
                        gdiff
                        if gdiff is not None
                        else float(np.abs(d)[wmask].max()) * 1.05
                    )
                    if dmax and dmax > 0:
                        fig.update_yaxes(range=[-dmax, dmax], row=r, col=1)
                    fig.update_yaxes(
                        title_text="Δ", title_font=dict(size=8), row=r, col=1
                    )
                else:  # gene row
                    _add_gene_traces(
                        fig,
                        go,
                        r,
                        cur_gene,
                        cur_others,
                        max_other_genes=max_other_genes,
                        xlim=window_bp,
                    )

            # variant marker across all rows
            fig.add_vline(
                x=variant_pos_bp,
                line=dict(color=VARIANT_COLOR, width=1, dash="dash"),
                opacity=0.7,
            )
            # shared x range = zoom window; label only on the bottom row
            fig.update_xaxes(range=list(window_bp))
            fig.update_xaxes(title_text="genomic position (bp)", row=n_rows, col=1)

            fig.update_layout(
                autosize=True,
                height=int(sum(row_heights) * row_height_px / 1.0) + 80,
                margin=dict(l=70, r=20, t=40 if title else 16, b=40),
                hovermode="x unified",
                title=title,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0
                ),
                template="simple_white",
            )
            # plain Figure (no anywidget dep): autosize fills the cell width,
            # and native zoom/pan/hover work in the notebook output.
            fig.show(config={"responsive": True, "scrollZoom": True})

    group_dd.observe(_render, names="value")
    window_dd.observe(_render, names="value")
    _render()

    box = widgets.VBox([widgets.HBox([group_dd, window_dd]), out])
    box.layout.width = "100%"
    return box


def _add_gene_traces(fig, go, row, gene, other_genes, *, max_other_genes, xlim):
    """Draw gene span lines + exon segments into the gene subplot row."""
    rows = []
    if gene is not None:
        rows.append((gene, GENE_COLOR, gene.name or "gene"))
    for og in list(other_genes or [])[:max_other_genes]:
        rows.append((og, ISOFORM_COLOR, og.name or ""))

    for i, (g, color, label) in enumerate(rows):
        y = -i
        start, end = g.span()
        # span line
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[y, y],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        # exons (thick segments)
        for exon in g.get_exons():
            fig.add_trace(
                go.Scatter(
                    x=[exon.begin, exon.end],
                    y=[y, y],
                    mode="lines",
                    line=dict(color=color, width=8),
                    showlegend=False,
                    hovertemplate=f"{label}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        # name annotation, clamped into view
        anchor_x = min(end, xlim[1])
        if label:
            fig.add_annotation(
                x=anchor_x,
                y=y,
                text=f" {label}",
                showarrow=False,
                xanchor="left" if end <= xlim[1] else "right",
                font=dict(size=9, color=color),
                row=row,
                col=1,
            )

    fig.update_yaxes(
        showticklabels=False, range=[-(len(rows)) - 0.5, 0.5], row=row, col=1
    )
