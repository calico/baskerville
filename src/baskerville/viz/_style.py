"""Shared style defaults for baskerville.viz."""

from __future__ import annotations

import numpy as np

REF_COLOR = "#1f77b4"
ALT_COLOR = "#d62728"
GENE_COLOR = "#222222"
ISOFORM_COLOR = "#555555"
HIGHLIGHT_COLOR = "#ffeb99"
VARIANT_COLOR = "#000000"


def bin_centers_bp(n_bins: int, bin_size: int, start_bp: int) -> np.ndarray:
    """Return the genomic-bp center of each output bin."""
    return start_bp + (np.arange(n_bins) + 0.5) * bin_size


def log2_1p(x: np.ndarray) -> np.ndarray:
    return np.log2(np.clip(x, 0, None) + 1.0)


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
