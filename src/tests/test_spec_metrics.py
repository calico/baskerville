import glob
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from baskerville import dataset, metrics
from baskerville.trainer import Trainer, parse_stop_stat


@pytest.fixture
def targets_df():
    """Group A: 3 unstranded + 2 stranded pairs; group B: 2 unstranded."""
    rows = [
        ("a0", "RNA:a0", 0),
        ("a1+", "RNA:a1", 2),
        ("a1-", "RNA:a1", 1),
        ("a2", "RNA:a2", 3),
        ("a3+", "RNA:a3", 5),
        ("a3-", "RNA:a3", 4),
        ("a4", "RNA:a4", 6),
        ("b0", "CHIP:K4:b0", 7),
        ("b1", "CHIP:K4:b1", 8),
    ]
    return pd.DataFrame(rows, columns=["identifier", "description", "strand_pair"])


def _hist_reference(y, pair):
    """Per-track fp16 bit-pattern counts of y_t + y_pair(t); y (N, T, L)."""
    s = (y.astype(np.float32) + y[:, pair].astype(np.float32)).astype(np.float16)
    bits = s.view(np.uint16).transpose(1, 0, 2).reshape(y.shape[1], -1)
    return np.stack([np.bincount(b, minlength=dataset.NUM_HIST_BINS) for b in bits])


PAIR = np.array([0, 2, 1, 3, 5, 4, 6, 7, 8])


def _hist(seed=9):
    """Histograms of a random dataset for the 9-track fixture."""
    y = np.random.default_rng(seed).gamma(2, size=(2, 9, 64)).astype("float16")
    return _hist_reference(y, PAIR)


def test_groups_and_collapse(targets_df):
    groups = dataset.target_groups(targets_df)
    assert list(groups) == ["RNA"] * 7 + ["CHIP/K4"] * 2
    rep, pair = dataset.strand_collapse(targets_df)
    assert list(rep) == [0, 1, 3, 4, 6, 7, 8]
    assert list(pair) == [0, 2, 3, 5, 6, 7, 8]


def _qnorm_reference(cols):
    """Sort-based quantile normalization with tie blocks averaged; cols (T, n)."""
    ref = np.sort(cols.astype(np.float64), axis=1).mean(axis=0)
    out = np.empty(cols.shape)
    for t, c in enumerate(cols):
        order = np.argsort(c, kind="stable")
        _, start, counts = np.unique(c[order], return_index=True, return_counts=True)
        out[t, order] = np.repeat(np.add.reduceat(ref, start) / counts, counts)
    return out


def test_qmap_tables_match_qnorm():
    rng = np.random.default_rng(3)
    y = rng.gamma(0.3, 4, size=(5, 4, 50)).astype("float16")
    y[y < 0.2] = 0  # a large tie block at zero
    cols = y.transpose(1, 0, 2).reshape(4, -1)
    tables = metrics.qmap_tables(_hist_reference(y, np.arange(4)))
    # identity pair doubles values; doubling is exact in fp16
    cols2 = (2 * cols.astype(np.float32)).astype(np.float16)
    mapped = np.take_along_axis(tables, cols2.view(np.uint16).astype(int), axis=1)
    np.testing.assert_allclose(mapped, _qnorm_reference(cols2), rtol=1e-5)
    # absent values interpolate between neighbors and clamp outside
    assert np.all(np.diff(tables, axis=1) >= 0)


def _spec_numpy(preds, targets, rep, pair, tables):
    """Direct reference: pair-sum, table-map, regress out the group mean, Pearson."""

    def mapped(x):
        s = (x[:, rep].astype(np.float32) + x[:, pair]).astype(np.float16)
        bits = s.view(np.uint16).transpose(1, 0, 2).reshape(len(rep), -1)
        return np.take_along_axis(tables, bits.astype(int), axis=1).astype(np.float64)

    def resid(z):
        m = z.mean(axis=0)
        return np.array([zi - np.polyval(np.polyfit(m, zi, 1), m) for zi in z])

    rp, rt = resid(mapped(preds)), resid(mapped(targets))
    return np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rp, rt)])


def test_spec_matches_numpy(targets_df):
    rng = np.random.default_rng(0)
    preds = rng.gamma(2, size=(6, 9, 40)).astype("float32")
    targets = (preds + rng.gamma(1, size=preds.shape)).astype("float16")
    hist = _hist_reference(targets, PAIR)

    spec = metrics.SpecPearsonCorrCoef(targets_df, hist, group_min=3)
    assert list(spec.groups) == ["RNA"]
    for bi in [slice(0, 4), slice(4, 6)]:
        spec.update(torch.tensor(preds[bi]), torch.tensor(targets[bi]))
    spec_t = spec.compute().numpy()

    rep, pair = spec.groups["RNA"]
    tables = metrics.qmap_tables(hist[rep])
    expected = _spec_numpy(preds, targets, rep, pair, tables)
    np.testing.assert_allclose(spec_t[rep], expected, rtol=1e-4, atol=1e-5)
    assert np.isnan(np.delete(spec_t, rep)).all()


def test_spec_nearly_identical_tracks():
    rng = np.random.default_rng(1)
    num_tracks = 20
    shape = (1, num_tracks, 10000)
    shared = rng.uniform(100, 200, (1, 1, shape[-1]))
    preds = (shared + rng.normal(0, 0.001, shape)).astype(np.float32)
    targets = (shared + rng.normal(0, 0.001, shape)).astype(np.float16)
    df = pd.DataFrame({"description": ["RNA:a"] * num_tracks})
    hist = _hist_reference(targets, np.arange(num_tracks))
    spec = metrics.SpecPearsonCorrCoef(df, hist)
    for start in range(0, shape[-1], 2000):
        spec.update(
            torch.from_numpy(preds[:, :, start : start + 2000]),
            torch.from_numpy(targets[:, :, start : start + 2000]),
        )

    rep, pair = spec.groups["RNA"]
    expected = _spec_numpy(preds, targets, rep, pair, spec.table["RNA"].numpy())
    np.testing.assert_allclose(spec.compute().numpy(), expected, atol=1e-5)


def test_spec_constant_group_mean():
    targets = np.tile([0, 1], (1, 20, 20)).astype(np.float16)
    targets[:, 10:] = 1 - targets[:, 10:]
    df = pd.DataFrame({"description": ["RNA:a"] * 20})
    spec = metrics.SpecPearsonCorrCoef(df, _hist_reference(targets, np.arange(20)))
    y = torch.from_numpy(targets)
    spec.update(y, y)
    torch.testing.assert_close(spec.compute(), torch.ones(20))

    # Constant tracks still have undefined residual correlation.
    spec.reset()
    spec.update(torch.zeros_like(y), y)
    assert spec.compute().isnan().all()


def test_spec_gain_and_specific():
    """Depth gain on a shared signal scores ~0; track-specific signal scores."""
    num_tracks = 8
    targets_df = pd.DataFrame(
        {"description": [f"CHIP:K4:t{i}" for i in range(num_tracks)]}
    )
    rng = np.random.default_rng(4)
    shape = (16, num_tracks, 512)
    shared = rng.gamma(0.3, 3, size=(16, 1, 512))
    gain = rng.uniform(0.3, 3, size=(1, num_tracks, 1))
    noise = rng.gamma(1, 0.3, size=shape)
    jitter = 1 + 0.01 * rng.standard_normal(shape)
    specific = rng.gamma(0.5, 6, size=shape) * (rng.random(shape) < 0.05)

    def score(preds, targets):
        targets = targets.astype("float16")
        hist = _hist_reference(targets, np.arange(num_tracks))
        spec = metrics.SpecPearsonCorrCoef(targets_df, hist, group_min=2)
        spec.update(torch.tensor(preds, dtype=torch.float32), torch.tensor(targets))
        return spec.compute().numpy().mean()

    # depth: tracks are scaled copies, so the tables remove gain exactly
    assert abs(score(gain * shared * jitter, gain * (shared + noise))) < 0.05
    # unequal SNR: the target tables warp noise-free preds nonlinearly per
    # track, so some credit remains
    assert score(gain * shared, gain * shared + noise) < 0.3
    assert score(gain * shared + specific, gain * shared + noise + specific) > 0.9


def test_spec_nan_preds_and_fresh(targets_df):
    spec = metrics.SpecPearsonCorrCoef(targets_df, _hist(), group_min=2)
    y = torch.rand(2, 9, 16)
    preds = y.clone()
    preds[0, 0, 0] = float("nan")
    spec.update(preds, y)  # NaN maps like 0 instead of indexing out of bounds
    assert np.isfinite(spec.compute().numpy()[spec.groups["RNA"][0]]).all()

    other = spec.fresh()
    assert other.table["RNA"] is spec.table["RNA"]
    assert other.count["RNA"] == 0 and spec.count["RNA"] == 32


def test_dataset_metrics_groups(targets_df):
    dm = metrics.DatasetMetrics(True, False, 9, None, "cpu", targets_df, _hist(), 2)
    rng = np.random.default_rng(1)
    y = torch.tensor(rng.gamma(2, size=(4, 9, 32)), dtype=torch.float32)
    yh = SimpleNamespace(has_coverage=True, has_gene=False, coverage=y * 0.9 + 0.1)
    dm.update(yh, y, None, None, torch.tensor(1.0), 4)
    res = dm.compute()

    for key in ["spec", "r/RNA", "r2/RNA", "spec/RNA", "r/CHIP/K4", "spec/CHIP/K4"]:
        assert np.isfinite(res[key]), key
    assert res["spec"] == pytest.approx((res["spec/RNA"] + res["spec/CHIP/K4"]) / 2)
    assert "valid_spec" in dm.format_log("valid", res)

    # no histograms: groups still get r, but no spec
    dm = metrics.DatasetMetrics(True, False, 9, None, "cpu", targets_df, None, 2)
    dm.update(yh, y, None, None, torch.tensor(1.0), 4)
    res = dm.compute()
    assert "r/RNA" in res and "spec" not in res


def test_stop_stat():
    assert parse_stop_stat("weighted_r_r2") == {"r": 1, "r2": 0.25}
    assert parse_stop_stat({"spec": 1}) == {"spec": 1}
    with pytest.raises(ValueError):
        parse_stop_stat("spec")

    human = {"loss": 1.0, "r": 0.5, "r2": 0.2, "spec": 0.3, "spec/gtex": 0.4}
    mouse = {"loss": 2.0, "r": 0.4, "r2": 0.1, "spec": 0.2}
    gene_only = {"loss": 3.0, "r": None, "r2": None, "r_gene": 0.8, "r2_gene": 0.6}

    def stat(stop_stat, results):
        trainer = SimpleNamespace(
            stop_stat=parse_stop_stat(stop_stat),
            stop_stat_loss_fallback=isinstance(stop_stat, str),
        )
        return Trainer._compute_stop_stat(trainer, results)

    assert stat("weighted_r_r2", [human, mouse]) == pytest.approx(0.9 + 0.3 / 4)
    assert stat("loss", [human, mouse]) == pytest.approx(-3.0)
    assert stat({"spec": 1}, [human, mouse, gene_only]) == pytest.approx(0.5)
    assert stat({"spec/gtex": 2}, [human, mouse]) == pytest.approx(0.8)
    assert stat("weighted_r_r2", [gene_only]) == pytest.approx(-3.0)
    assert stat("r", [human, gene_only]) == pytest.approx(0.5 - 3.0)
    assert stat({"r_gene": 1}, [gene_only]) == pytest.approx(0.8)
    assert stat({"r_gene": 1}, [human, gene_only]) == pytest.approx(0.8)
    assert stat(
        {"r_gene": 1, "r2_gene": 0.25, "loss": -2}, [gene_only]
    ) == pytest.approx(0.8 + 0.6 / 4 - 6)
    with pytest.raises(ValueError):
        stat({"spec/nope": 1}, [human, mouse])
    with pytest.raises(ValueError):
        stat({"r": 1}, [gene_only])


@pytest.mark.parametrize(
    "key, group_min, means, early_stop, validation, supported",
    [
        ("spec", 20, True, 2, True, False),
        ("spec/nope", 2, True, 2, True, False),
        ("spec/RNA", 2, True, 2, True, True),
        ("spec/CHIP/K4", 2, True, 2, True, True),
        ("spec/CHIP/K4", 3, True, 2, True, False),
        ("spec", 2, False, 2, True, False),
        ("spec/RNA", 2, True, 1, True, False),
        ("spec/RNA", 2, True, 0, False, True),
    ],
)
def test_spec_stop_validation(
    targets_df, key, group_min, means, early_stop, validation, supported
):
    train_data = [
        SimpleNamespace(
            has_coverage=True,
            has_genes=False,
            num_targets=len(df),
            targets_df=df,
            target_hist=_hist() if means else None,
        )
        for df in [targets_df.assign(group="other"), targets_df]
    ]
    trainer = SimpleNamespace(
        is_mlm=False,
        model=SimpleNamespace(heads_cov=True, heads_gene=None),
        model_heads=None,
        num_datasets=2,
        train_data=train_data,
        eval_data=train_data if validation else [],
        early_stop_datasets=early_stop,
        device="cpu",
        spec_group_min=group_min,
        stop_stat={key: 1},
    )
    if supported:
        Trainer._init_metrics(trainer)
    else:
        with pytest.raises(ValueError, match="absent from all datasets"):
            Trainer._init_metrics(trainer)


def test_target_hist_and_map_fp16_boundaries(monkeypatch):
    y = np.tile(np.array([-0.0, 1, 40000, 65504], dtype=np.float16), (1, 3, 1))
    monkeypatch.setattr(dataset.zarr, "open", lambda *a, **kw: {"target": y})
    hist = dataset._target_block_hist(("unused", 0, 1, np.array([1, 0, 2])))
    expected = np.zeros_like(hist)
    expected[:, 0] = 1
    expected[:, 0x4000] = 1  # 2.0
    expected[:, 0x7BFF] = 2  # 65504.0
    np.testing.assert_array_equal(hist, expected)

    df = pd.DataFrame(
        {
            "identifier": ["a+", "a-", "b"],
            "description": ["RNA:a", "RNA:a", "RNA:b"],
            "strand_pair": [1, 0, 2],
        }
    )
    spec = metrics.SpecPearsonCorrCoef(df, hist, group_min=2)
    mapped = spec._map(torch.from_numpy(y), "RNA")
    torch.testing.assert_close(
        mapped, torch.tensor([[0.0, 2, 65504, 65504]]).expand(2, -1)
    )


@pytest.mark.parametrize("value", [-1, np.inf, -np.inf, np.nan])
def test_target_hist_rejects_invalid_before_clamping(monkeypatch, value):
    y = np.array([[[value]]], dtype=np.float16)
    monkeypatch.setattr(dataset.zarr, "open", lambda *a, **kw: {"target": y})
    with pytest.raises(ValueError, match="negative or non-finite"):
        dataset._target_block_hist(("unused", 0, 1, np.array([0])))


def test_write_target_hist(tmp_path):
    # tracks 0/1 are a stranded pair, track 2 unstranded
    pd.DataFrame({"identifier": ["a+", "a-", "b"], "strand_pair": [1, 0, 2]}).to_csv(
        tmp_path / "targets.txt", sep="\t"
    )
    rng = np.random.default_rng(2)
    values = []
    for fold, num_seqs in enumerate([5, 3]):
        x = rng.gamma(0.5, 3, size=(num_seqs, 3, 16)).astype("float16")
        x[x < 0.5] = 0
        root = zarr.open_group(str(tmp_path / f"examples/fold{fold}.zarr"), mode="w")
        root.create_array("target", shape=x.shape, dtype="float16")[:] = x
        values.append(x)

    hist = dataset.write_target_hist(str(tmp_path), processes=2, block_seqs=2)
    expected = _hist_reference(np.concatenate(values), np.array([1, 0, 2]))
    np.testing.assert_array_equal(hist, expected)
    assert hist.sum(axis=1).tolist() == [8 * 16] * 3
    for fold in range(2):
        stored = zarr.open(str(tmp_path / f"examples/fold{fold}.zarr"), mode="r")
        np.testing.assert_array_equal(stored["target_hist"][:], expected)


def test_write_target_hist_rejects_negative(tmp_path):
    pd.DataFrame({"identifier": ["a"]}).to_csv(tmp_path / "targets.txt", sep="\t")
    root = zarr.open_group(str(tmp_path / "examples/fold0.zarr"), mode="w")
    root.create_array("target", shape=(1, 1, 4), dtype="float16")[:] = -1
    with pytest.raises(ValueError, match="negative or non-finite"):
        dataset.write_target_hist(str(tmp_path), processes=1)


def test_hound_data_writes_hist(data_me_dir):
    targets_df = pd.read_csv(f"{data_me_dir}/targets.txt", sep="\t", index_col=0)
    pair = dataset.strand_pair_indices(targets_df)
    zarr_files = sorted(glob.glob(f"{data_me_dir}/examples/*.zarr"))
    roots = [zarr.open(zf, mode="r") for zf in zarr_files]
    expected = _hist_reference(np.concatenate([r["target"][:] for r in roots]), pair)
    for r in roots:
        np.testing.assert_array_equal(r["target_hist"][:], expected)
