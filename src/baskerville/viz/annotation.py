"""Gene-annotation and marker primitives for variant visualization."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..gene import Gene
from ._style import GENE_COLOR, HIGHLIGHT_COLOR, ISOFORM_COLOR, VARIANT_COLOR


def _gene_row(
    ax: plt.Axes,
    gene: Gene,
    *,
    y_center: float,
    height: float,
    color: str,
    label: str | None,
    show_strand_arrows: bool,
    xlim: tuple[int, int] | None = None,
) -> None:
    """Draw one gene's span line + exon rectangles centered at y_center."""
    start, end = gene.span()
    half = height / 2.0

    ax.hlines(y_center, start, end, color=color, linewidth=0.8, zorder=2)

    for exon in gene.get_exons():
        ax.add_patch(
            mpatches.Rectangle(
                (exon.begin, y_center - half),
                exon.end - exon.begin,
                height,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )

    if show_strand_arrows and gene.strand in ("+", "-"):
        # arrows in introns
        marker = ">" if gene.strand == "+" else "<"
        exons = gene.get_exons()
        intron_spans = []
        for a, b in zip(exons[:-1], exons[1:]):
            if b.begin > a.end:
                intron_spans.append((a.end, b.begin))
        for a, b in intron_spans:
            mid = 0.5 * (a + b)
            ax.plot(
                [mid], [y_center], marker=marker, color=color, markersize=4, zorder=4
            )

    if label:
        # default: anchor label to the gene's end with a leading space. If
        # the gene's end is past the visible right edge, pin the label to
        # the right edge and right-align so it stays in view.
        anchor_x = end
        ha = "left"
        text = f" {label}"
        if xlim is not None:
            x0, x1 = xlim
            if end > x1:
                anchor_x = x1
                ha = "right"
                text = f"{label} "
            elif start < x0 and end < x0:
                # gene is entirely off-screen left; nothing to label
                return
        ax.text(
            anchor_x,
            y_center,
            text,
            va="center",
            ha=ha,
            fontsize=7,
            color=color,
            zorder=5,
        )


def draw_gene_track(
    ax: plt.Axes,
    gene: Gene,
    *,
    isoforms: Sequence[Gene] | None = None,
    other_genes: Sequence[Gene] | None = None,
    max_isoforms: int = 5,
    label: str | None = None,
    color: str = GENE_COLOR,
    isoform_color: str = ISOFORM_COLOR,
    other_color: str = ISOFORM_COLOR,
    xlim: tuple[int, int] | None = None,
) -> None:
    """Render a gene-annotation panel into `ax`.

    Layout: main gene on top, optional isoforms below, optional neighboring
    genes at the bottom. Each row has the same visual height. The y-axis is
    hidden; `ax` should be a slim subplot beneath the coverage panels.
    """
    isoforms = list(isoforms or [])[:max_isoforms]
    other_genes = list(other_genes or [])

    rows = []
    rows.append(("main", gene, color, label or (gene.name or "gene")))
    for i, iso in enumerate(isoforms):
        rows.append(("iso", iso, isoform_color, iso.name or f"iso{i}"))
    for og in other_genes:
        rows.append(("other", og, other_color, og.name or ""))

    height = 0.6
    spacing = 1.0
    for i, (_, g, c, lbl) in enumerate(rows):
        y = -(i * spacing)
        _gene_row(
            ax,
            g,
            y_center=y,
            height=height,
            color=c,
            label=lbl,
            show_strand_arrows=True,
            xlim=xlim,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(-(len(rows) - 1) * spacing - 1.0, 1.0)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def mark_variant(
    axes: plt.Axes | Iterable[plt.Axes],
    pos_bp: int,
    *,
    color: str = VARIANT_COLOR,
    linestyle: str = "--",
    linewidth: float = 0.8,
    alpha: float = 0.7,
) -> None:
    """Vertical line at `pos_bp` across one or more axes."""
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for ax in axes:
        ax.axvline(
            pos_bp,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )


def highlight_region(
    axes: plt.Axes | Iterable[plt.Axes],
    start_bp: int,
    end_bp: int,
    *,
    color: str = HIGHLIGHT_COLOR,
    alpha: float = 0.4,
) -> None:
    """Shaded vertical span across one or more axes."""
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for ax in axes:
        ax.axvspan(start_bp, end_bp, color=color, alpha=alpha, linewidth=0, zorder=0)
