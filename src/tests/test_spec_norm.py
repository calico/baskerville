"""Tests for streaming cross-track specificity normalization (spec_norm).

These are CPU-only: they compare the streaming implementation against the
original ``qnorm``-based reshape path on synthetic arrays. Strand pairs are
collapsed to one column upstream (``seqnn.eval(combine_pairs=...)``), so
spec_norm here only ever handles individual columns.
"""

import numpy as np
import pandas as pd
import pytest
from qnorm import quantile_normalize
from scipy.stats import pearsonr
import zarr

from baskerville import spec_norm as sn
from baskerville.dataset import strand_pair_indices


# --------------------------------------------------------------------------- #
# reference implementation (mirrors the original qnorm-based hound_eval_spec)
# --------------------------------------------------------------------------- #
def _reference_spec(eval_preds, eval_targets, column_indices, var_pct=1.0):
    """Original path: build the (rows, cols) matrix, qnorm, mean-subtract, corr.

    Each column is one track's flattened vector, matching the streaming column
    definition up to a row permutation (which the metric is invariant to).
    """
    cols_p = [
        eval_preds[:, c, :].astype(np.float32).reshape(-1) for c in column_indices
    ]
    cols_t = [
        eval_targets[:, c, :].astype(np.float32).reshape(-1) for c in column_indices
    ]
    ep = np.stack(cols_p, axis=1)
    et = np.stack(cols_t, axis=1)

    epn = quantile_normalize(ep, axis=1)
    etn = quantile_normalize(et, axis=1)

    if var_pct < 1:
        v = etn.var(axis=1)
        thr = np.percentile(v, 100 * (1 - var_pct))
        mask = v >= thr
        epn, etn = epn[mask], etn[mask]

    epn -= epn.mean(axis=-1, keepdims=True)
    etn -= etn.mean(axis=-1, keepdims=True)
    return np.array(
        [pearsonr(epn[:, j], etn[:, j])[0] for j in range(len(column_indices))]
    )


def _synthetic(num_seqs, num_targets, bins, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    preds = rng.normal(1.0, 1.0, (num_seqs, num_targets, bins))
    targets = preds + rng.normal(0.0, 0.5, (num_seqs, num_targets, bins))
    if ties:
        # inject a large block of zeros / repeated values (like coverage data)
        preds = np.maximum(preds - 1.0, 0.0)
        targets = np.maximum(targets - 1.0, 0.0)
        preds = np.round(preds * 2) / 2
        targets = np.round(targets * 2) / 2
    return preds.astype(np.float16), targets.astype(np.float16)


# --------------------------------------------------------------------------- #
# apply_quantile_column
# --------------------------------------------------------------------------- #
def test_apply_quantile_column_matches_qnorm_ties():
    rng = np.random.default_rng(3)
    rows, cols = 400, 5
    # ties-heavy integer matrix + a few jittered values
    M = rng.integers(0, 3, size=(rows, cols)).astype(np.float32)
    jitter = (rng.random((rows, cols)) > 0.8) * rng.normal(0, 0.01, (rows, cols))
    M += jitter.astype(np.float32)

    ref_qn = quantile_normalize(M.copy(), axis=1)
    reference = np.sort(M, axis=0).mean(axis=1)
    mine = np.stack(
        [sn.apply_quantile_column(M[:, j], reference) for j in range(cols)], axis=1
    )
    np.testing.assert_allclose(mine, ref_qn, atol=1e-4)


# --------------------------------------------------------------------------- #
# group_specificity_pearson vs the original path
# --------------------------------------------------------------------------- #
def test_group_specificity_unstranded():
    preds, targets = _synthetic(8, 6, 40, seed=10)
    cols = list(range(6))
    mine = sn.group_specificity_pearson(preds, targets, cols, band_size=4)
    ref = _reference_spec(preds, targets, cols)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_group_specificity_ties_heavy():
    preds, targets = _synthetic(10, 8, 50, seed=11, ties=True)
    cols = list(range(8))
    mine = sn.group_specificity_pearson(preds, targets, cols, band_size=3)
    ref = _reference_spec(preds, targets, cols)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_group_specificity_var_pct():
    preds, targets = _synthetic(8, 6, 40, seed=12)
    cols = list(range(6))
    mine = sn.group_specificity_pearson(preds, targets, cols, band_size=2, var_pct=0.5)
    ref = _reference_spec(preds, targets, cols, var_pct=0.5)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_group_specificity_subset_columns():
    # a group need not be every track: pass an out-of-order index subset
    preds, targets = _synthetic(8, 8, 40, seed=19)
    cols = [5, 1, 6, 2]
    mine = sn.group_specificity_pearson(preds, targets, cols, band_size=2)
    ref = _reference_spec(preds, targets, cols)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_group_specificity_zarr_matches_numpy():
    preds, targets = _synthetic(8, 6, 40, seed=13)
    # target-axis chunked store (chunk=3 tracks), as seqnn.eval(target_chunk=...)
    # would write it — the band reader touches only its own target-chunks
    zp = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=preds.shape,
        chunks=(2, 3, 40),
        dtype="float16",
    )
    zt = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=targets.shape,
        chunks=(2, 3, 40),
        dtype="float16",
    )
    zp[:] = preds
    zt[:] = targets
    cols = list(range(6))
    r_np = sn.group_specificity_pearson(preds, targets, cols, band_size=3)
    r_zr = sn.group_specificity_pearson(zp, zt, cols, band_size=3)
    np.testing.assert_allclose(r_np, r_zr, atol=1e-6)


def test_group_specificity_ncpus_matches_serial():
    preds, targets = _synthetic(8, 6, 40, seed=14)
    cols = list(range(6))
    r1 = sn.group_specificity_pearson(preds, targets, cols, band_size=2, ncpus=1)
    r4 = sn.group_specificity_pearson(preds, targets, cols, band_size=2, ncpus=4)
    np.testing.assert_allclose(r1, r4, atol=1e-6)


def test_constant_track_matches_reference():
    # A constant raw track becomes constant after quantile norm, but the
    # per-position cross-track mean subtraction still makes it vary, so it is not
    # degenerate. The key property is that streaming matches the qnorm path.
    preds, targets = _synthetic(6, 4, 20, seed=16)
    targets = targets.astype(np.float32)
    targets[:, 0, :] = 5.0
    targets = targets.astype(np.float16)
    cols = list(range(4))
    mine = sn.group_specificity_pearson(preds, targets, cols, band_size=4)
    ref = _reference_spec(preds, targets, cols)
    np.testing.assert_allclose(mine, ref, atol=1e-4)


def test_non_finite_input_raises():
    preds, targets = _synthetic(6, 4, 20, seed=18)
    bad = preds.astype(np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        sn.group_specificity_pearson(bad.astype(np.float16), targets, list(range(4)))


def test_single_column_group():
    preds, targets = _synthetic(6, 1, 20, seed=17)
    # one column: after cross-track mean subtraction everything is 0 -> NaN
    mine = sn.group_specificity_pearson(preds, targets, [0], band_size=1)
    assert mine.shape == (1,)
    assert np.isnan(mine[0])


def test_collapsed_group_loop():
    """Mirror hound_eval_spec's per-group loop over a pre-summed (collapsed) store."""
    num_seqs, bins = 8, 30
    # one column per experiment: RNA (4), DNASE (5), SMALL (2, below group_min).
    group = ["RNA"] * 4 + ["DNASE"] * 5 + ["SMALL"] * 2
    num_cols = len(group)
    preds, targets = _synthetic(num_seqs, num_cols, bins, seed=42)
    group_values = np.array(group)

    group_min = 4
    targets_spec = np.full(num_cols, np.nan)
    for tg in sorted(set(group)):
        col_idx = np.where(group_values == tg)[0]
        if len(col_idx) < group_min:
            continue
        r = sn.group_specificity_pearson(preds, targets, col_idx, band_size=2)
        ref = _reference_spec(preds, targets, col_idx)
        np.testing.assert_allclose(r, ref, atol=1e-4)
        targets_spec[col_idx] = r

    assert np.all(np.isnan(targets_spec[9:11]))  # SMALL skipped
    assert np.all(np.isfinite(targets_spec[:9]))


# --------------------------------------------------------------------------- #
# strand_pair_indices helper (used by hound_eval_spec to build combine_pairs)
# --------------------------------------------------------------------------- #
def test_strand_pair_indices_identity_without_column():
    df = pd.DataFrame({"identifier": ["x", "y"]}, index=[5, 6])
    np.testing.assert_array_equal(strand_pair_indices(df), [0, 1])


def test_strand_pair_indices_remaps_labels():
    df = pd.DataFrame(
        {"strand_pair": [2, 3, 0, 1]},  # labels happen to be 0..N-1 here
        index=[0, 1, 2, 3],
    )
    np.testing.assert_array_equal(strand_pair_indices(df), [2, 3, 0, 1])
