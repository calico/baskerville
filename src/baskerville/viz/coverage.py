"""Coverage-track plotting primitives."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ._style import ALT_COLOR, REF_COLOR, bin_centers_bp, despine, log2_1p


def plot_coverage_pair(
    ax: plt.Axes,
    ref: np.ndarray,
    alt: np.ndarray | None = None,
    *,
    bin_size: int = 32,
    start_bp: int = 0,
    colors: tuple[str, str] = (REF_COLOR, ALT_COLOR),
    labels: tuple[str, str] = ("REF", "ALT"),
    alpha: float = 0.5,
    style: Literal["fill", "bar"] = "fill",
    log_scale: bool = False,
    normalize_counts: bool = False,
) -> None:
    """Plot reference (and optional alternative) coverage on `ax`.

    Arrays are expected to be post-untransform (see
    `baskerville.dataset.untransform_preds`). Shapes are `[n_bins]`.
    """
    ref = np.asarray(ref, dtype=float)
    if alt is not None:
        alt = np.asarray(alt, dtype=float)
        if alt.shape != ref.shape:
            raise ValueError(f"ref/alt shape mismatch: {ref.shape} vs {alt.shape}")

    if normalize_counts and alt is not None:
        ref_sum = float(ref.sum())
        alt_sum = float(alt.sum())
        if alt_sum > 0:
            alt = alt * (ref_sum / alt_sum)

    if log_scale:
        ref = log2_1p(ref)
        if alt is not None:
            alt = log2_1p(alt)

    x = bin_centers_bp(ref.shape[0], bin_size, start_bp)

    def _draw(values, color, label):
        if style == "fill":
            ax.fill_between(
                x, 0, values, color=color, alpha=alpha, label=label, linewidth=0
            )
        elif style == "bar":
            ax.bar(
                x,
                values,
                width=bin_size,
                color=color,
                alpha=alpha,
                label=label,
                linewidth=0,
            )
        else:
            raise ValueError(f"unknown style {style!r}")

    _draw(ref, colors[0], labels[0])
    if alt is not None:
        _draw(alt, colors[1], labels[1])

    ax.set_xlim(x[0] - bin_size / 2, x[-1] + bin_size / 2)
    ax.set_ylim(bottom=0)
    despine(ax)
