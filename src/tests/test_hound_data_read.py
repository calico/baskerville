"""CovFace reads a coarse-resolution .hw as its base-pair expansion, zero past the end."""

import numpy as np
import pytest

hdwig = pytest.importorskip("hdwig")

from baskerville.scripts.hound_data_read import CovFace


def test_resolution_matches_bp_expansion(tmp_path):
    rng = np.random.default_rng(0)
    length, res = 10_000, 20
    values = (rng.random(length // res) * 50).astype("float16")
    hdwig.write(tmp_path / "r20.hw", {"chrT": values}, resolution=res)
    hdwig.write(tmp_path / "r1.hw", {"chrT": np.repeat(values, res)})
    coarse, fine = CovFace(str(tmp_path / "r20.hw")), CovFace(str(tmp_path / "r1.hw"))

    for _ in range(1000):
        start = int(rng.integers(0, length))
        end = start + int(rng.integers(1, 700))
        x, y = coarse.read("chrT", start, end), fine.read("chrT", start, end)
        assert len(x) == end - start
        np.testing.assert_array_equal(x.astype("float32"), y.astype("float32"))
