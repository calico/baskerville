"""Streaming per-track Pearson R and R² against a float64 numpy reference."""

import numpy as np
import pytest
import torch

from baskerville import metrics


def _reference(p, y):
    """Per-track Pearson R and R² of (P, T) arrays."""
    pc, yc = p - p.mean(0), y - y.mean(0)
    r = (pc * yc).sum(0) / np.sqrt((pc**2).sum(0) * (yc**2).sum(0))
    r2 = 1 - ((y - p) ** 2).sum(0) / (yc**2).sum(0)
    return r, r2


def _flat(a):
    """(..., T, L) -> (pairs, T)."""
    return np.moveaxis(a, -2, -1).reshape(-1, a.shape[-2])


def test_streaming_matches_reference(monkeypatch):
    monkeypatch.setattr(
        metrics.TrackMetrics, "_chunk_pairs", 2 * 64 * 2
    )  # 2 + 2 + 1 tracks
    rng = np.random.default_rng(0)
    y = rng.gamma(0.5, 2.0, size=(3, 2, 5, 64)).astype(np.float32)
    y[:, :, 3] = 1000 + 1e-3 * rng.standard_normal(y[:, :, 3].shape)  # small variance
    p = (y + rng.standard_normal(y.shape)).astype(np.float32)
    p[:, :, 3] = y[:, :, 3] + 1e-3 * rng.standard_normal(y[:, :, 3].shape)
    scores = metrics.TrackMetrics(5)
    for pb, yb in zip(p, y):
        scores.update(torch.from_numpy(pb), torch.from_numpy(yb))
    r, r2 = _reference(_flat(p).astype(np.float64), _flat(y).astype(np.float64))
    np.testing.assert_allclose(scores.compute()[0], r, rtol=1e-4)
    np.testing.assert_allclose(scores.compute()[1], r2, rtol=1e-4, atol=1e-5)


def test_constant_targets():
    y = torch.tensor([[[0.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    p = y.clone()
    p[:, 2] += 1
    scores = metrics.TrackMetrics(3)
    scores.update(p, y)
    r, r2 = scores.compute()
    torch.testing.assert_close(
        r, torch.tensor([1.0, torch.nan, torch.nan]), equal_nan=True
    )
    torch.testing.assert_close(r2, torch.tensor([1.0, 1.0, 0.0]))


def test_r2_is_scale_independent_and_propagates_nonfinite_error():
    y = torch.tensor([[[0.0, 1e-5], [0.0, 1e-3], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    p = torch.zeros_like(y)
    p[0, 2, 0] = 1e-5
    p[0, 3, 0] = torch.nan
    p[0, 4, 0] = torch.inf
    scores = metrics.TrackMetrics(5)
    scores.update(p, y)
    _, r2 = scores.compute()
    torch.testing.assert_close(
        r2, torch.tensor([-1.0, -1.0, 0.0, torch.nan, torch.nan]), equal_nan=True
    )


def test_constant_noninteger_target_over_many_positions():
    # Summation can round the mean away from the repeated float32 value.
    y = torch.full((1, 1, 2_493_127), 0.1)
    scores = metrics.TrackMetrics(1)
    scores.update(torch.zeros_like(y), y)
    assert scores.compute()[1].item() == 0.0


def test_masked_genes_match_reference(monkeypatch):
    monkeypatch.setattr(metrics.TrackMetrics, "_chunk_pairs", 2 * 10 * 2)
    rng = np.random.default_rng(1)
    p, y = rng.random((2, 3, 10)), rng.random((2, 3, 10))
    mask = rng.random((2, 10)) > 0.4
    mask[0] = False
    p_nan = p.copy()
    p_nan.transpose(0, 2, 1)[~mask] = np.nan  # masked positions never count
    scores = metrics.TrackMetrics(3)
    scores.update(torch.tensor(p_nan), torch.tensor(y), torch.tensor(mask).float())
    scores.update(
        torch.tensor(p), torch.tensor(y), torch.zeros(2, 10, dtype=torch.bool)
    )
    r, r2 = _reference(p.transpose(0, 2, 1)[mask], y.transpose(0, 2, 1)[mask])
    np.testing.assert_allclose(scores.compute()[0], r, rtol=1e-5)
    np.testing.assert_allclose(scores.compute()[1], r2, rtol=1e-5)


def test_empty_and_reset():
    scores = metrics.TrackMetrics(3)
    p = torch.rand(2, 3, 8)
    nan = torch.full((3,), torch.nan)
    for _ in range(2):
        for v in scores.compute():
            torch.testing.assert_close(v, nan, equal_nan=True)
        scores.update(p * torch.nan, p, torch.zeros(2, 8, dtype=torch.bool))
        for v in scores.compute():
            torch.testing.assert_close(v, nan, equal_nan=True)
        scores.update(p, p)
        torch.testing.assert_close(scores.compute()[1], torch.ones(3))
        scores.reset()


def test_invalid_shapes_leave_state_intact():
    scores = metrics.TrackMetrics(2)
    p = torch.rand(2, 2, 4)
    scores.update(p, p)
    expected = scores.compute()
    with pytest.raises(ValueError, match="same"):
        scores.update(p[:, :, :1], p)
    with pytest.raises(ValueError, match="same"):
        scores.update(p[:, :1], p[:, :1])
    with pytest.raises(ValueError, match="mask"):
        scores.update(p, p, torch.ones(4, 2, dtype=torch.bool))
    assert scores.count.item() == 8
    torch.testing.assert_close(scores.compute(), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_metrics_on_mps():
    p, y = (torch.rand(2, 3, 8) for _ in range(2))
    mask = torch.rand(2, 8) > 0.5
    for selected in (None, mask):
        cpu, mps = metrics.TrackMetrics(3), metrics.TrackMetrics(3)
        cpu.update(p, y, selected)
        mps.update(
            p.to("mps"), y.to("mps"), None if selected is None else selected.to("mps")
        )
        torch.testing.assert_close(mps.compute(), cpu.compute())
