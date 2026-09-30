import numpy as np
import pytest

from baskerville.gene import Gene, interval_output_slice


@pytest.mark.parametrize(
    "start,end,nearest,any_overlap",
    [
        (1000, 1064, (0, 2), (0, 2)),
        (1017, 1039, (1, 1), (0, 2)),
        (1016, 1048, (0, 2), (0, 2)),  # half-bin ties round to even
        (1048, 1080, (2, 2), (1, 3)),
        (0, 990, (0, 0), (0, 0)),
        (1200, 1250, (4, 4), (4, 4)),
        (990, 1010, (0, 0), (0, 1)),
        (1110, 1200, (3, 4), (3, 4)),
        (990, 1200, (0, 4), (0, 4)),
    ],
)
def test_interval_output_slice(start, end, nearest, any_overlap):
    assert interval_output_slice(start, end, 1000, 128, 32) == nearest
    assert (
        interval_output_slice(start, end, 1000, 128, 32, majority_overlap=False)
        == any_overlap
    )


@pytest.mark.parametrize(
    "span,majority_overlap,expected",
    [
        (False, False, [0, 1, 3]),
        (False, True, [1, 3]),
        (True, False, [0, 1, 2, 3]),
        (True, True, [0, 1, 2, 3]),
    ],
)
def test_gene_output_slice(span, majority_overlap, expected):
    gene = Gene("chr1", "+", {})
    for start, end in [(1001, 1015), (1020, 1040), (1030, 1048), (1104, 1120)]:
        gene.add_exon(start, end)
    np.testing.assert_array_equal(
        gene.output_slice(1000, 128, 32, span=span, majority_overlap=majority_overlap),
        expected,
    )
    assert gene.output_slice(1200, 128, 32, span=span).size == 0
