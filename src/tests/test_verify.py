"""Published-model regression guard + core verification machinery.

``test_published_forward`` is the every-PR guard: for each published config it
runs a full deterministic forward and compares a compact fingerprint against the
committed reference. It auto-detects downloaded weights (using them when
present, otherwise a deterministic synthetic-weights forward), so the published
architecture is exercised without the multi-hundred-MB checkpoints. It skips
only CUDA-only (Hydra) architectures on hosts without a GPU.
"""

import numpy as np
import pytest
import torch

from baskerville import verification as V


@pytest.mark.slow
@pytest.mark.parametrize("family", V.FAMILIES)
@pytest.mark.parametrize("species", V.SPECIES)
def test_published_forward(family, species):
    if not V.params_path(family, species).exists():
        pytest.fail(
            f"missing pinned reference for {family}/{species}; "
            f"run hound_verify --generate"
        )
    if V.needs_cuda(V.params_path(family, species)) and not torch.cuda.is_available():
        pytest.skip(f"{family} requires CUDA (Triton scan kernels)")
    fold, res = V.verify(family, species)
    assert res.shape_ok, (
        f"{family}/{species}: output shape {res.cur_shape} != reference {res.ref_shape}"
    )
    assert res.passed, (
        f"{family}/{species} (fold={fold}): r={res.r:.6f} "
        f"rmse_rel={res.rmse_rel:.3e} -- a code change likely regressed this "
        f"published architecture's forward numerics."
    )


def test_synthetic_weights_are_reproducible():
    """Layer A references are only valid if synthetic weights are identical
    across machines/torch versions -- this is the guarantee that lets the
    guard run in CI without trained checkpoints."""

    def tiny():
        return torch.nn.Sequential(
            torch.nn.Conv1d(4, 8, 3, padding=1),
            torch.nn.BatchNorm1d(8),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 16, 5),
        )

    a, b = tiny(), tiny()
    V.fill_synthetic_weights(a)
    V.fill_synthetic_weights(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k])
        assert torch.isfinite(sa[k]).all(), k


def test_compare_catches_regressions_r_alone_would_miss(tmp_path):
    """The reference round-trips, identical output passes, and -- crucially --
    a uniform affine drift is caught by the rmse gate even though Pearson r
    (scale/shift invariant) stays above its threshold."""
    out = np.random.default_rng(0).standard_normal((64, 96)).astype(np.float32)
    path = tmp_path / "ref.npz"
    V.save_fingerprint(
        path, V.fingerprint(out), V.make_meta("borzoi", "human", None, "synthetic", out)
    )
    ref = V.load_fingerprint(path)

    identical = V.compare(ref, out)
    assert identical.passed and identical.r >= 0.999999
    assert identical.rmse_rel == 0.0

    drifted = V.compare(ref, out * 1.01)
    assert drifted.r >= V.R_THRESHOLD, "sanity: r alone does not catch this"
    assert not drifted.passed, "rmse gate must catch the affine drift"

    mismatched = V.compare(ref, out[:, :-1])
    assert not mismatched.passed and not mismatched.shape_ok
