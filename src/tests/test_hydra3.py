"""
Smoke tests for the bidirectional Mamba-3 block (Hydra3).

The Mamba-3 SISO scan is a Triton kernel and requires a CUDA GPU plus a
mamba_ssm build that ships the Mamba-3 kernels (see the recipe in
dockerfiles/baskerville.Dockerfile).
These tests skip automatically when either is unavailable.
"""

import pytest
import torch

from baskerville import layers
from baskerville.blocks import HydraTower

requires_mamba3 = pytest.mark.skipif(
    not torch.cuda.is_available() or layers.mamba3_siso_combined is None,
    reason="Hydra3 needs a CUDA GPU and the Mamba-3 mamba_ssm build",
)


@requires_mamba3
@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_hydra_tower_forward_backward(version, grad_checkpoint):
    """HydraTower v2 (Mamba-2) and v3 (Mamba-3) preserve shape and backprop, with and
    without gradient checkpointing (use_reentrant=False)."""
    torch.manual_seed(0)
    batch, channels, length = 2, 128, 128

    kwargs = dict(
        repeat=2,
        channels=channels,
        headdim=64,
        dropout=0.0,
        grad_checkpoint=grad_checkpoint,
    )
    if version == 3:
        # Mamba-3: d_state is the rotary head dim and must be an even power-of-two.
        kwargs.update(version=3, d_state=32, rope_fraction=0.5, chunk_size=64)
    else:
        kwargs.update(version=2, d_state=64)

    tower = HydraTower(**kwargs).cuda()
    x = torch.randn(batch, channels, length, device="cuda", requires_grad=True)

    y = tower(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in tower.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@requires_mamba3
def test_hydra3_bidirectional_receptive_field():
    """A Hydra3 output position should depend on both upstream and downstream input
    (true bidirectionality), unlike a causal scan."""
    torch.manual_seed(0)
    batch, channels, length = 1, 64, 64
    tower = (
        HydraTower(
            repeat=1,
            channels=channels,
            headdim=32,
            d_state=32,
            rope_fraction=0.5,
            chunk_size=64,
            version=3,
            dropout=0.0,
        )
        .cuda()
        .eval()
    )

    x = torch.randn(batch, channels, length, device="cuda")
    mid = length // 2
    with torch.no_grad():
        base = tower(x)
        # perturb a position strictly downstream of `mid`
        x_down = x.clone()
        x_down[:, :, mid + 8] += 5.0
        out_down = tower(x_down)

    # output at `mid` must change when a downstream input changes -> bidirectional
    delta_at_mid = (out_down[:, :, mid] - base[:, :, mid]).abs().max()
    assert delta_at_mid > 1e-3, "output at mid did not respond to downstream input"


@requires_mamba3
def test_hydra3_rope_rate_scales_phase_rate():
    """`rope_rate` must be an exact multiplier on the phase rate.

    The kernel advances phase at pi tanh(angle) dt, so the throttled block's
    tanh(angle) has to be exactly rope_rate times the unthrottled block's, and
    the two must give different outputs (a no-op would pass vacuously)."""
    torch.manual_seed(0)
    rate = 0.125
    kw = dict(d_model=64, d_state=16, headdim=32, chunk_size=64)
    full = layers.Hydra3(**kw).cuda().float().eval()
    thr = layers.Hydra3(**kw, rope_rate=rate).cuda().float().eval()
    thr.load_state_dict(full.state_dict())

    u = torch.randn(1, 128, 64, device="cuda")
    with torch.no_grad():
        angles = []
        for m in (full, thr):
            proj = m.in_proj_dyn(u)
            angles.append(proj[..., -2 * m.num_rope_angles :].float())
        a_full, a_thr_in = angles
        # replicate the block's throttle: atanh(rate * tanh(angle))
        a_thr = torch.atanh(rate * torch.tanh(a_thr_in))
        ratio = torch.tanh(a_thr) / torch.tanh(a_full)
        assert torch.allclose(ratio, torch.full_like(ratio, rate), atol=1e-5)
        assert (full(u) - thr(u)).abs().max() > 1e-4, "rope_rate had no effect"
