import numpy as np
import pandas as pd
import pytest
import torch

from baskerville import dataset, metrics


def _yy(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    def rand():
        return torch.rand(shape, generator=g, dtype=torch.float64)

    y = rand() * (rand() < 0.7)
    yp = (rand() + 0.1).requires_grad_()
    return yp, y


def _split_reference(yp, y, w, wsum, c, t):
    """t·P(total) + Σ_k c_k Σ_nodes binomial split NLL, built level by level."""
    a, b = yp * w, y * w
    loss = 0
    for ck in c:
        al, ar = a[..., 0::2], a[..., 1::2]
        bl, br = b[..., 0::2], b[..., 1::2]
        ap = al + ar
        loss = loss + ck * (-bl * torch.log(al / ap) - br * torch.log(ar / ap)).sum(-1)
        a, b = ap, bl + br
    total = (a - b * torch.log(a)).sum(-1)
    return ((loss + t * total) / wsum).mean()


def test_tree_matches_split_reference():
    yp, y = _yy((2, 3, 8))
    t = 0.3
    loss_fn = metrics.PoissonTreeLoss(t, split_exp=0.5, split_knee=2, weight_range=10,
                                      weight_exp=8, epsilon=0)
    loss = loss_fn(yp, y)
    (grad,) = torch.autograd.grad(loss, yp)
    w = metrics.position_weights(8, 10, 8).double()
    c = metrics.split_profile(3, 0.5, 2)
    ref = _split_reference(yp, y, w, loss_fn.weights_sum.double(), c, t)
    (grad_ref,) = torch.autograd.grad(ref, yp)
    torch.testing.assert_close(loss, ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(grad, grad_ref, rtol=1e-10, atol=1e-12)


def test_tree_flat_matches_poisson_mn():
    yp, y = _yy((2, 3, 16))
    grads = []
    for loss_fn in [metrics.PoissonMultinomialLoss(0.25, 10, 8),
                    metrics.PoissonTreeLoss(0.25, weight_range=10, weight_exp=8)]:
        grads.append(torch.autograd.grad(loss_fn(yp, y), yp)[0])
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=1e-9)


def test_split_profile():
    c = np.array(metrics.split_profile(14, 0.25, 32))
    np.testing.assert_allclose(c[:5], 1)
    np.testing.assert_allclose(c[5:], 2.0 ** (-0.25 * np.arange(1, 10)))
    assert metrics.split_profile(2, 0.5, 1) == [2**-0.5, 0.5]


def test_tree_checks_lengths():
    yp, y = _yy((1, 1, 12))
    with pytest.raises(ValueError, match="power-of-two length"):
        metrics.PoissonTreeLoss(0.25)(yp, y)
    with pytest.raises(ValueError, match="split_knee"):
        metrics.PoissonTreeLoss(0.25, split_knee=24)


def _values():
    v = np.arange(dataset.NUM_HIST_BINS, dtype=np.uint16).view(np.float16)
    return v.astype(np.float64)


def _hists(seed=0, tracks=3, n=4000):
    """Equal-total fp16 histograms with ties, a zero mass, and a long tail."""
    rng = np.random.default_rng(seed)
    y = (rng.gamma(0.5, 3, (tracks, n)) * (rng.random((tracks, n)) < 0.6))
    bits = y.astype(np.float16).view(np.uint16)
    return np.stack([np.bincount(b, minlength=dataset.NUM_HIST_BINS) for b in bits])


def _apply(x, xk, qk):
    return metrics.qmap_apply(x, torch.from_numpy(xk), torch.from_numpy(qk))


def test_qmap_knots_match_tables():
    hist = _hists()
    tables = metrics.qmap_tables(hist).astype(np.float64)
    xk, qk = metrics.qmap_knots(hist, 64)
    values = _values()
    for t in range(len(hist)):
        nz = np.flatnonzero(hist[t])
        # exact at the knots
        real = xk[t] <= values[nz[-1]]
        x = torch.from_numpy(np.tile(xk[t][real], (len(hist), 1)))
        q = _apply(x, xk, qk)[t].numpy()
        np.testing.assert_allclose(q, qk[t][real], rtol=1e-12)
        # monotone between knots, so off the table by at most a segment's rise;
        # clamped beyond the observed values
        x = np.concatenate([[-1.0], values[nz], [values[nz[-1]] * 2]])
        q = _apply(torch.from_numpy(np.tile(x, (len(hist), 1))), xk, qk)[t].numpy()
        assert np.all(np.diff(q) >= 0)
        assert q[0] == tables[t, nz[0]] and q[-1] == tables[t, nz[-1]]


def test_qmap_constant_track():
    hist = np.zeros((2, dataset.NUM_HIST_BINS), dtype=np.int64)
    hist[:, 0] = 100  # every value is 0
    xk, qk = metrics.qmap_knots(hist, 16)
    x = torch.tensor([[0.0, 1.0, 1e3]] * 2, dtype=torch.float64, requires_grad=True)
    q = _apply(x, xk, qk)
    np.testing.assert_array_equal(q.detach().numpy(), 0)
    (grad,) = torch.autograd.grad(q.sum(), x)
    np.testing.assert_array_equal(grad.numpy(), 0)


def test_qmap_flat_stretch():
    xk = np.array([[0.0, 1.0, 2.0, 3.0]])
    qk = np.array([[0.0, 1.0, 1.0, 2.0]])
    x = torch.tensor([[1.5, 0.5, 2.5, -1.0, 4.0]], dtype=torch.float64, requires_grad=True)
    q = _apply(x, xk, qk)
    np.testing.assert_allclose(q.detach().numpy(), [[1.0, 0.5, 1.5, 0.0, 2.0]])
    (grad,) = torch.autograd.grad(q.sum(), x)
    np.testing.assert_allclose(grad.numpy(), [[0.0, 1.0, 1.0, 0.0, 0.0]])


@pytest.fixture
def targets_df():
    """One group of 3 unstranded tracks + 1 stranded pair; one track ungrouped."""
    rows = [
        ("a0", "RNA:a0", 0),
        ("a1+", "RNA:a1", 2),
        ("a1-", "RNA:a1", 1),
        ("a2", "RNA:a2", 3),
        ("a3", "RNA:a3", 4),
        ("b0", "CHIP:K4:b0", 5),
    ]
    return pd.DataFrame(rows, columns=["identifier", "description", "strand_pair"])


def _qmap_loss(targets_df, seed=0, form="mse"):
    rng = np.random.default_rng(seed)
    y = rng.gamma(0.5, 3, (2, 6, 32)) * (rng.random((2, 6, 32)) < 0.6)
    y = y.astype(np.float16)
    pair = targets_df.strand_pair.to_numpy()
    s = (y.astype(np.float32) + y[:, pair].astype(np.float32)).astype(np.float16)
    bits = s.view(np.uint16).transpose(1, 0, 2).reshape(6, -1)
    hist = np.stack([np.bincount(b, minlength=dataset.NUM_HIST_BINS) for b in bits])
    loss_fn = metrics.QmapCrossTrackLoss(targets_df, hist, 10, 8, group_min=4, form=form)
    return loss_fn, torch.from_numpy(y.astype(np.float64))


def test_qmap_loss_zero_at_target(targets_df):
    loss_fn, y = _qmap_loss(targets_df)
    assert loss_fn.rep.tolist() == [0, 1, 3, 4]  # group RNA only
    yp = y.clone().requires_grad_()
    loss = loss_fn(yp, y)
    (grad,) = torch.autograd.grad(loss, yp)
    assert loss.item() < 1e-12
    assert grad.abs().max().item() < 1e-6


def test_qmap_loss_finite_and_specific(targets_df):
    loss_fn, y = _qmap_loss(targets_df)
    yp = torch.zeros_like(y).requires_grad_()  # all zeros: ties at the lowest knot
    loss = loss_fn(yp, y)
    (grad,) = torch.autograd.grad(loss, yp)
    assert loss.item() > 0 and torch.isfinite(grad).all()
    # the ungrouped track gets no gradient
    assert grad[:, 5].abs().max().item() == 0
    # the ungrouped track's error costs nothing; a grouped track's does
    off = y.clone()
    off[:, 5] *= 3
    assert loss_fn(off, y).item() < 1e-12
    off[:, 0] *= 3
    assert loss_fn(off, y).item() > 0


@pytest.mark.parametrize("form", metrics.QmapCrossTrackLoss.FORMS)
def test_qmap_loss_groups_weigh_equally(form):
    """Group B is group A at 10x scale: the same error costs the same in either."""
    df = pd.DataFrame(
        [(f"{g}{i}", f"{d}:{g}{i}", k) for k, (g, d, i) in enumerate(
            [(g, d, i) for g, d in [("a", "RNA"), ("b", "CHIP:K4")] for i in range(3)])],
        columns=["identifier", "description", "strand_pair"],
    )
    rng = np.random.default_rng(0)
    ya = rng.gamma(0.5, 3, (2, 3, 32)) * (rng.random((2, 3, 32)) < 0.6)
    y = np.concatenate([ya, 10 * ya], axis=1).astype(np.float16)
    s = (2 * y.astype(np.float32)).astype(np.float16)  # unstranded: strand sum is 2y
    bits = s.view(np.uint16).transpose(1, 0, 2).reshape(6, -1)
    hist = np.stack([np.bincount(b, minlength=dataset.NUM_HIST_BINS) for b in bits])
    loss_fn = metrics.QmapCrossTrackLoss(df, hist, 10, 8, group_min=3, form=form)
    y = torch.from_numpy(y.astype(np.float64))
    losses = []
    for t in (0, 3):  # first track of group a, of group b
        yp = y.clone()
        yp[:, t] *= 1.5
        losses.append(loss_fn(yp, y).item())
    assert losses[0] != 0
    np.testing.assert_allclose(losses[0], losses[1], rtol=0.02)


def test_per_dataset_weights():
    from baskerville.trainer import per_dataset

    assert per_dataset(0.5, 2, "w") == [0.5, 0.5]
    assert per_dataset([0.1, 0.9], 2, "w") == [0.1, 0.9]
    with pytest.raises(ValueError, match="one value per dataset"):
        per_dataset([0.1], 2, "w")


FORMS = metrics.QmapCrossTrackLoss.FORMS


@pytest.mark.parametrize("form", FORMS)
def test_qmap_forms_finite(targets_df, form):
    loss_fn, y = _qmap_loss(targets_df, form=form)
    yp = torch.zeros_like(y).requires_grad_()  # constant preds: a = 0
    (grad,) = torch.autograd.grad(loss_fn(yp, y), yp)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert grad[:, 5].abs().max().item() == 0


@pytest.mark.parametrize("form", ["proj", "mn"])
def test_qmap_forms_zero_at_target(targets_df, form):
    loss_fn, y = _qmap_loss(targets_df, form=form)
    yp = y.clone().requires_grad_()
    loss = loss_fn(yp, y)
    (grad,) = torch.autograd.grad(loss, yp)
    assert abs(loss.item()) < 1e-6 and grad.abs().max().item() < 1e-6  # fp32


def _mapped(targets_df, form):
    """A form's loss, with synthetic mapped values (Tq = 4, B = 2, L = 16)."""
    loss_fn = _qmap_loss(targets_df, form=form)[0].double()
    loss_fn.position_weights = metrics.position_weights(16, 10, 8, "cpu").double()
    rng = np.random.default_rng(0)
    zy, zp = (torch.from_numpy(rng.gamma(0.5, 3, (4, 2, 16)) + 0.1) for _ in range(2))
    return loss_fn, zp, zy


def test_qmap_dot_gradient(targets_df):
    """dot's gradient is -2b, whatever the prediction."""
    loss_fn, zp, zy = _mapped(targets_df, "dot")
    zp.requires_grad_()
    (g,) = torch.autograd.grad(loss_fn.per_track(zp, zy).sum(), zp)
    w = loss_fn.position_weights
    b = zy - zy.mean(0, keepdim=True)
    torch.testing.assert_close(g, -2 * w * b / (2 * w.sum()))


def test_qmap_proj_direction_only(targets_df):
    """proj: any positive multiple of b, plus a shift shared by the group, costs 0;
    mse charges for the scale."""
    loss_fn, zp, zy = _mapped(targets_df, "proj")
    b = zy - zy.mean(0, keepdim=True)
    zero = torch.zeros(4, dtype=b.dtype)
    torch.testing.assert_close(loss_fn.per_track(2.5 * b + 7, zy), zero, atol=1e-12, rtol=0)
    assert (loss_fn.per_track(-b, zy) > 0).all()
    mse, _, _ = _mapped(targets_df, "mse")
    assert (mse.per_track(2.5 * b + 7, zy) > 0).all()


def test_qmap_mn_bin_scale(targets_df):
    """mn sees only shares across tracks: a factor common to a bin costs nothing."""
    loss_fn, zp, zy = _mapped(targets_df, "mn")
    loss_fn.eps.zero_()
    f = torch.from_numpy(np.random.default_rng(1).uniform(0.5, 4, (1, 2, 16)))
    zero = torch.tensor(0.0, dtype=zy.dtype)
    torch.testing.assert_close(loss_fn.per_track(zy * f, zy).sum(), zero, atol=1e-12, rtol=0)
    assert loss_fn.per_track(zp, zy).sum() > 0


def test_qmap_form_checked(targets_df):
    with pytest.raises(ValueError, match="qmap form"):
        _qmap_loss(targets_df, form="cos")
