"""
Tests for the strand-selection plumbing that hound_grad's gradient ensembling
depends on, independent of any trained model: under reverse-complement the
gene's signal must be read from the opposite-strand tracks. Numerical fwd/rev
*equivariance* is deliberately not tested here — that is a property of a trained
model, not of this plumbing. The reverse-complement of the gradients themselves
is handled by dna.hot1_rc (see test_dna.py).
"""

import pandas as pd
import pytest

from baskerville.dataset import annotate_strand
from baskerville.scripts.hound_grad import select_strand_indices


def _stranded_targets():
    """Targets table with a +/- stranded pair (rows 0,1) and an unstranded track (row 2)."""
    df = pd.DataFrame(
        {
            "identifier": ["CAGE+", "CAGE-", "DNASE"],
            # row 0 pairs with row 1 and vice versa; row 2 pairs with itself
            "strand_pair": [1, 0, 2],
        },
        index=[0, 1, 2],
    )
    return annotate_strand(df)


def test_annotate_strand_labels():
    df = _stranded_targets()
    assert list(df.strand) == ["+", "-", "."]


@pytest.mark.parametrize(
    "gene_strand, rev_comp, expected",
    [
        # forward orientation: read the gene's own strand (+ unstranded)
        ("+", False, [0, 2]),
        ("-", False, [1, 2]),
        # reverse-complement: the gene's signal moves to the opposite strand
        ("+", True, [1, 2]),
        ("-", True, [0, 2]),
    ],
)
def test_select_strand_indices(gene_strand, rev_comp, expected):
    df = _stranded_targets()
    got = select_strand_indices(df, gene_strand, rev_comp)
    assert list(got) == expected


def test_rc_selects_opposite_strand_and_is_nonempty():
    """Regression for the original bug: a + gene under RC selected no targets
    because the '-' rows had been dropped, zeroing the gradient."""
    df = _stranded_targets()
    fwd = set(select_strand_indices(df, "+", rev_comp=False))
    rev = set(select_strand_indices(df, "+", rev_comp=True))

    assert 1 in rev, "RC pass must include the '-' strand track"
    assert 0 not in rev, "RC pass must drop the '+' strand track"
    assert len(rev) > 0
    # the stranded partner is swapped between orientations; '.' stays in both
    assert (fwd ^ rev) == {0, 1}
    assert 2 in fwd and 2 in rev
