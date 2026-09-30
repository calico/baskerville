"""Static matplotlib visualization for baskerville.

Composable primitives for variant analysis: plot reference vs alternative
coverage predictions in the context of gene annotation.

Common entry point:
    from baskerville.viz import plot_snp

Lower-level primitives:
    from baskerville.viz import (
        plot_coverage_pair, draw_gene_track, mark_variant, highlight_region,
    )
"""

from .annotation import draw_gene_track, highlight_region, mark_variant
from .coverage import plot_coverage_pair
from .ism import check_ism, ism_track_switcher, launch_ism_bg, plot_ism_logo
from .snp import interactive_plot_snp, plot_snp, seq_window_for_variant
from .snp_plotly import plotly_browser_snp

__all__ = [
    "plot_snp",
    "interactive_plot_snp",
    "plotly_browser_snp",
    "plot_coverage_pair",
    "draw_gene_track",
    "mark_variant",
    "highlight_region",
    "seq_window_for_variant",
    "plot_ism_logo",
    "ism_track_switcher",
    "launch_ism_bg",
    "check_ism",
]
