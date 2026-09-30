"""
Tests for DNA one-hot utilities in baskerville.dna.
"""

import numpy as np

from baskerville.dna import hot1_rc


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
