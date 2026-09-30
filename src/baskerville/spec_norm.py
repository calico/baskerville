"""Memory-efficient cross-track specificity normalization.

`hound_eval_spec` measures how well a model predicts the *relative* differences
between same-assay tracks at each genomic position (rather than just the shared
mean). Historically this quantile-normalized a dense ``(num_seqs*bins, num_tracks)``
matrix per target group with the ``qnorm`` package, whose internal int64 argsort
temporaries blow up memory for large track collections.

This module reproduces the exact quantile-normalization semantics but computes them
**one column at a time**, so peak memory is ``O(num_positions)`` and independent of
the number of tracks. A "column" is one stored track's flattened ``eval[:, ti, :]``
vector. Stranded experiments are already collapsed to a single total-signal column
upstream (``seqnn.eval(combine_pairs=...)`` sums each plus/minus pair before writing
the store), so this module only ever sees individual columns. Quantile normalization,
the per-position cross-track mean subtraction, the optional high-variance-position
filter, and the per-track Pearson correlation are all computed in a few streaming
passes with float64 accumulators.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import pearsonr

# Default columns (tracks) per band. Band memory is ~2 * band_size * R * 4
# bytes (preds + targets, float32). When the eval store is chunked along the
# target axis (hound_eval_spec passes target_chunk=band), a band read only
# touches its own chunks, so this doubles as the natural target-chunk size.
DEFAULT_BAND_SIZE = 16


def apply_quantile_column(col, reference):
    """Map a column onto a reference distribution, exactly like ``qnorm``.

    Each value is replaced by the reference value at its rank; tied values are
    all assigned the *average* of the reference over the tied block (matching
    ``qnorm.quantile_normalize``'s handling, which a rank-midpoint interpolation
    would not reproduce on the large block of zeros typical of coverage data).

    Args:
        col: 1D array of length R.
        reference: 1D ascending array of length R (the mean of the sorted
            columns across the group).

    Returns:
        float64 1D array of length R (the normalized column).
    """
    col = np.asarray(col)
    n = col.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    # ties are block-averaged below, so intra-tie order is irrelevant: the
    # default (faster) quicksort gives identical results to a stable sort.
    order = np.argsort(col)
    sorted_col = col[order]

    # group id per sorted position: increments at each new (untied) value
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(sorted_col[1:], sorted_col[:-1], out=change[1:])
    group_id = np.cumsum(change) - 1

    # average the reference within each tie block
    ref = np.asarray(reference, dtype=np.float64)
    block_sum = np.bincount(group_id, weights=ref)
    block_cnt = np.bincount(group_id)
    block_mean = block_sum / block_cnt

    out[order] = block_mean[group_id]
    return out


def _read_tracks(eval_arr, track_indices):
    """Read ``eval_arr[:, track_indices, :]`` (numpy or zarr) as finite float32.

    Non-finite values raise (QN has no NaN/inf handling).
    """
    if isinstance(eval_arr, np.ndarray):
        sub = eval_arr[:, track_indices, :]
    else:
        # zarr: orthogonal indexing along the target axis
        sub = eval_arr.oindex[:, list(track_indices), :]
    arr = np.asarray(sub, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("spec_norm requires finite preds/targets (found NaN/inf)")
    return arr


def _iter_bands(eval_preds, eval_targets, column_indices, band_size):
    """Yield, for each band of columns, a list of ``(col_p, col_t)`` pairs.

    A band reads ``band_size`` columns from ``eval_preds``/``eval_targets`` in a
    single (chunk-friendly) read and flattens each into its ``num_seqs*bins``
    vector. Re-invoking this generator re-reads the data (one read per pass).
    """
    num_cols = len(column_indices)
    for start in range(0, num_cols, band_size):
        idx = [int(c) for c in column_indices[start : start + band_size]]
        band_p = _read_tracks(eval_preds, idx)
        band_t = _read_tracks(eval_targets, idx)
        band = [
            # reshape already copies (non-contiguous view)
            (band_p[:, k, :].reshape(-1), band_t[:, k, :].reshape(-1))
            for k in range(len(idx))
        ]
        yield band


def _pmap(func, items, executor):
    """Map ``func`` over ``items``, on ``executor``'s threads if one is given.

    numpy sort/apply release the GIL, so threads give real parallelism; the
    caller reduces the results serially to avoid races on the accumulators.
    """
    if executor is not None and len(items) > 1:
        return list(executor.map(func, items))
    return [func(it) for it in items]


def _stream_columns(
    eval_preds, eval_targets, column_indices, band_size, per_column, executor
):
    """Yield ``per_column((col_p, col_t))`` for every column, in column order.

    Wraps the band read + optional threaded map so each streaming pass is just a
    per-column function plus a serial reduction of the yielded results.
    """
    for band in _iter_bands(eval_preds, eval_targets, column_indices, band_size):
        yield from _pmap(per_column, band, executor)


def _pearson(xp, xt):
    """Pearson r, returning NaN for degenerate (constant / <2 point) inputs."""
    if xp.size < 2:
        return np.nan
    sp = xp.std()
    st = xt.std()
    if sp == 0 or st == 0 or not np.isfinite(sp) or not np.isfinite(st):
        return np.nan
    return pearsonr(xp, xt)[0]


def group_specificity_pearson(
    eval_preds,
    eval_targets,
    column_indices,
    *,
    var_pct=1.0,
    band_size=DEFAULT_BAND_SIZE,
    ncpus=1,
):
    """Cross-track specificity Pearson r per column, computed by streaming.

    For a group of columns, quantile-normalize every column to a common marginal,
    subtract the per-position cross-track mean, optionally restrict to the most
    variable positions, and correlate normalized predictions vs targets per
    column. Memory is ``O(band_size * num_positions)`` regardless of the number
    of columns; the data is read one band at a time (three passes).

    For a zarr input, reads are only cheap if the store is chunked along the
    target axis (see ``seqnn.eval(target_chunk=...)``); otherwise every band read
    decompresses the whole store.

    Args:
        eval_preds: array ``(num_seqs, num_targets, bins)`` (numpy in RAM or a
            zarr array on disk), float16/float32.
        eval_targets: same shape/type as ``eval_preds``.
        column_indices: 1D sequence of track indices (into the store's target
            axis) forming the group; stranded pairs are collapsed upstream.
        var_pct: proportion of highest-variance positions to keep (1.0 = all).
        band_size: number of columns read per band (memory/I-O knob).
        ncpus: threads for the per-column sort/apply (default 1); one executor is
            shared across all bands and passes.

    Returns:
        float64 array of length ``len(column_indices)`` with the per-column
        Pearson r (NaN where degenerate), aligned with ``column_indices``.
    """
    column_indices = list(column_indices)
    num_cols = len(column_indices)
    if num_cols == 0:
        return np.zeros(0, dtype=np.float64)

    if band_size < 1:
        raise ValueError("band_size must be >= 1")
    if not (0 < var_pct <= 1):
        raise ValueError("var_pct must be in (0, 1]")

    num_seqs, _, bins = eval_preds.shape
    num_rows = num_seqs * bins

    executor = ThreadPoolExecutor(max_workers=ncpus) if (ncpus and ncpus > 1) else None
    try:

        def stream(per_column):
            return _stream_columns(
                eval_preds,
                eval_targets,
                column_indices,
                band_size,
                per_column,
                executor,
            )

        # Pass 1: quantile-normalization references (mean of sorted columns).
        ref_p = np.zeros(num_rows, dtype=np.float64)
        ref_t = np.zeros(num_rows, dtype=np.float64)
        for sorted_p, sorted_t in stream(
            lambda pair: (np.sort(pair[0]), np.sort(pair[1]))
        ):
            ref_p += sorted_p
            ref_t += sorted_t
        ref_p /= num_cols
        ref_t /= num_cols

        # Pass 2: cross-track (per-position) means + target variance for the mask.
        rowsum_p = np.zeros(num_rows, dtype=np.float64)
        rowsum_t = np.zeros(num_rows, dtype=np.float64)
        rowsumsq_t = np.zeros(num_rows, dtype=np.float64)

        def _normalize(pair):
            cp, ct = pair
            return apply_quantile_column(cp, ref_p), apply_quantile_column(ct, ref_t)

        for npv, ntv in stream(_normalize):
            rowsum_p += npv
            rowsum_t += ntv
            rowsumsq_t += ntv * ntv
        rowmean_p = rowsum_p / num_cols
        rowmean_t = rowsum_t / num_cols

        mask = None
        if var_pct < 1:
            var_t = rowsumsq_t / num_cols - rowmean_t**2
            np.maximum(var_t, 0, out=var_t)
            thresh = np.percentile(var_t, 100 * (1 - var_pct))
            mask = var_t >= thresh

        # Pass 3: per-column specificity Pearson r.
        def _corr(pair):
            cp, ct = pair
            xp = apply_quantile_column(cp, ref_p) - rowmean_p
            xt = apply_quantile_column(ct, ref_t) - rowmean_t
            if mask is not None:
                xp = xp[mask]
                xt = xt[mask]
            return _pearson(xp, xt)

        corr = np.fromiter(stream(_corr), dtype=np.float64, count=num_cols)
    finally:
        if executor is not None:
            executor.shutdown()
    return corr
