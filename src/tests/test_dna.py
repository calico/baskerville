"""
Tests for DNA one-hot utilities in baskerville.dna.
"""

import random

import numpy as np
import pytest

from baskerville.dna import dna_1hot, hot1_rc


def test_hot1_rc_known_value():
    """Reverse the sequence axis and swap complementary bases (A<->T, C<->G),
    column order A, C, G, T."""
    # one nonzero base per position, down the diagonal: A=1, C=2, G=3, T=4
    seq = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # A
            [0.0, 2.0, 0.0, 0.0],  # C
            [0.0, 0.0, 3.0, 0.0],  # G
            [0.0, 0.0, 0.0, 4.0],  # T
        ]
    )
    # reversed position order is T,G,C,A; complementing each (T->A, G->C, C->G,
    # A->T) puts every value back on the diagonal in order 4, 3, 2, 1
    expected = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],  # last pos was T(4) -> complements to A
            [0.0, 3.0, 0.0, 0.0],  # was G(3) -> C
            [0.0, 0.0, 2.0, 0.0],  # was C(2) -> G
            [0.0, 0.0, 0.0, 1.0],  # was A(1) -> T
        ]
    )
    np.testing.assert_array_equal(hot1_rc(seq), expected)


def test_hot1_rc_is_involution():
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((11, 4))
    np.testing.assert_allclose(hot1_rc(hot1_rc(seq)), seq)


def test_hot1_rc_does_not_mutate_input():
    seq = np.arange(12, dtype=float).reshape(3, 4)
    before = seq.copy()
    hot1_rc(seq)
    np.testing.assert_array_equal(seq, before)


def test_hot1_rc_batched_matches_singleton():
    """The batched path (ndim==3) should reverse-complement each sequence
    independently, matching the 2D singleton path."""
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((3, 7, 4))
    out = hot1_rc(batch)
    assert out.shape == batch.shape
    for i in range(batch.shape[0]):
        np.testing.assert_array_equal(out[i], hot1_rc(batch[i]))


def _dna_1hot_per_base(seq, seq_len=None, n_uniform=False, n_sample=False):
    """dna_1hot's original per-base loop, the reference for the vectorized one."""
    if seq_len is None:
        seq_len, seq_start = len(seq), 0
    elif seq_len <= len(seq):
        seq_trim = (len(seq) - seq_len) // 2
        seq, seq_start = seq[seq_trim : seq_trim + seq_len], 0
    else:
        seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()
    seq_code = np.zeros((seq_len, 4), dtype="float16" if n_uniform else "bool")
    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt in "ACGT":
                seq_code[i, "ACGT".index(nt)] = 1
            elif n_uniform:
                seq_code[i, :] = 0.25
            elif n_sample:
                seq_code[i, random.randint(0, 3)] = 1
    return seq_code


def test_dna_1hot_known_value():
    expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    np.testing.assert_array_equal(dna_1hot("ACgtN"), np.array(expected, dtype=bool))


@pytest.mark.parametrize("seq_len", [None, 37, 64, 101])
@pytest.mark.parametrize("mode", ["plain", "n_uniform", "n_sample"])
def test_dna_1hot_matches_per_base_loop(seq_len, mode):
    """Trimming, padding, N handling and n_sample's random draws all match."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGTNacgtn"), 64))
    kwargs = {"n_uniform": mode == "n_uniform", "n_sample": mode == "n_sample"}
    random.seed(1)
    expected = _dna_1hot_per_base(seq, seq_len, **kwargs)
    random.seed(1)
    got = dna_1hot(seq, seq_len, **kwargs)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
