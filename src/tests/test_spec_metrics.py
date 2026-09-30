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


MEANS = np.linspace(0.5, 2.0, 9)


def test_groups_and_collapse(targets_df):
    groups = dataset.target_groups(targets_df)
    assert list(groups) == ["RNA"] * 7 + ["CHIP/K4"] * 2
    rep, pair = dataset.strand_collapse(targets_df)
    assert list(rep) == [0, 1, 3, 4, 6, 7, 8]
    assert list(pair) == [0, 2, 3, 5, 6, 7, 8]


def _spec_numpy(preds, targets, rep, pair, means):
    """Direct reference: pair-sum, mean-normalize, center, per-track Pearson."""
    b = (means[rep] + means[pair])[None, :, None]

    def center(x):
        x = (x[:, rep] + x[:, pair]) / b
        return x - x.mean(axis=1, keepdims=True)

    p, t = center(preds), center(targets)
    p = p.transpose(0, 2, 1).reshape(-1, len(rep))
    t = t.transpose(0, 2, 1).reshape(-1, len(rep))
    return np.array([np.corrcoef(p[:, i], t[:, i])[0, 1] for i in range(len(rep))])


def test_spec_matches_numpy(targets_df):
    rng = np.random.default_rng(0)
    preds = rng.gamma(2, size=(6, 9, 40)).astype("float32")
    targets = (preds + rng.gamma(1, size=preds.shape)).astype("float32")

    spec = metrics.SpecPearsonCorrCoef(targets_df, MEANS, group_min=3)
    assert list(spec.groups) == ["RNA"]
    for bi in [slice(0, 4), slice(4, 6)]:
        spec.update(torch.tensor(preds[bi]), torch.tensor(targets[bi]))
    spec_t = spec.compute().numpy()

    rep, pair = spec.groups["RNA"]
    expected = _spec_numpy(preds, targets, rep, pair, MEANS)
    np.testing.assert_allclose(spec_t[rep], expected, rtol=1e-4, atol=1e-5)
    assert np.isnan(np.delete(spec_t, rep)).all()


def test_dataset_metrics_groups(targets_df):
    dm = metrics.DatasetMetrics(True, False, 9, None, "cpu", targets_df, MEANS, 2)
    rng = np.random.default_rng(1)
    y = torch.tensor(rng.gamma(2, size=(4, 9, 32)), dtype=torch.float32)
    yh = SimpleNamespace(has_coverage=True, has_gene=False, coverage=y * 0.9 + 0.1)
    dm.update(yh, y, None, None, torch.tensor(1.0), 4)
    res = dm.compute()

    for key in ["spec", "r/RNA", "r2/RNA", "spec/RNA", "r/CHIP/K4", "spec/CHIP/K4"]:
        assert np.isfinite(res[key]), key
    assert res["spec"] == pytest.approx((res["spec/RNA"] + res["spec/CHIP/K4"]) / 2)
    assert "valid_spec" in dm.format_log("valid", res)

    # no means: groups still get r, but no spec
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
            target_means=MEANS if means else None,
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


def test_write_target_means(tmp_path):
    rng = np.random.default_rng(2)
    values = []
    for fold, num_seqs in enumerate([5, 3]):
        x = rng.gamma(2, size=(num_seqs, 9, 16)).astype("float16")
        root = zarr.open_group(str(tmp_path / f"examples/fold{fold}.zarr"), mode="w")
        root.create_array("target", shape=x.shape, dtype="float16")[:] = x
        values.append(x)

    means = dataset.write_target_means(str(tmp_path), processes=2, block_seqs=2)
    expected = np.concatenate(values).astype("float64").mean(axis=(0, 2))
    np.testing.assert_allclose(means, expected, rtol=1e-12)
    for fold in range(2):
        target = zarr.open(str(tmp_path / f"examples/fold{fold}.zarr"), mode="r")[
            "target"
        ]
        np.testing.assert_allclose(target.attrs["mean"], expected, rtol=1e-12)


def test_hound_data_writes_means(data_me_dir):
    zarr_files = sorted(glob.glob(f"{data_me_dir}/examples/*.zarr"))
    targets = [zarr.open(zf, mode="r")["target"] for zf in zarr_files]
    expected = np.concatenate([t[:].astype("float64") for t in targets])
    expected = expected.mean(axis=(0, 2))
    for t in targets:
        np.testing.assert_allclose(t.attrs["mean"], expected, rtol=1e-12)
