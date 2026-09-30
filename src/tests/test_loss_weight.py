import numpy as np
import pandas as pd
import pytest
import torch

from baskerville.metrics import (
    GeneMSELoss,
    GenePoissonLoss,
    PoissonLoss,
    PoissonMultinomialLoss,
)
from baskerville.trainer import parse_loss, parse_target_weights


@pytest.fixture
def targets_df():
    return pd.DataFrame(
        {
            "identifier": ["a", "b", "c", "d"],
            "assay": ["chip", "chip", "dnase", "rna"],
            "group": ["H3K4me3", "H3K9me3", "dnase", "gtex"],
        }
    )


def test_no_rules_no_column(targets_df):
    assert np.array_equal(parse_target_weights(targets_df, []), np.ones(4))


def test_column_alone(targets_df):
    targets_df["weight"] = [0.5, 1.0, 2.0, 1.0]
    assert np.array_equal(parse_target_weights(targets_df, []), [0.5, 1.0, 2.0, 1.0])


@pytest.mark.parametrize("source", ["column", "rule"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_weights_rejected(targets_df, source, value):
    rules = []
    if source == "column":
        targets_df["weight"] = [value, 1.0, 1.0, 1.0]
    else:
        rules = [{"assay": "chip", "weight": value}]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        parse_target_weights(targets_df, rules)


@pytest.mark.parametrize("source", ["column", "rule"])
def test_all_zero_weights_rejected(targets_df, source):
    rules = []
    if source == "column":
        targets_df["weight"] = 0.0
    else:
        rules = [{"assay": assay, "weight": 0.0} for assay in targets_df.assay.unique()]
    with pytest.raises(ValueError, match="positive total"):
        parse_target_weights(targets_df, rules)


def test_zero_weight_can_disable_individual_tracks(targets_df):
    weights = parse_target_weights(targets_df, [{"assay": "chip", "weight": 0.0}])
    assert np.array_equal(weights, [0.0, 0.0, 1.0, 1.0])


def test_rule_multiplies_column(targets_df):
    targets_df["weight"] = [0.5, 1.0, 2.0, 1.0]
    weights = parse_target_weights(targets_df, [{"assay": "chip", "weight": 0.2}])
    assert np.allclose(weights, [0.1, 0.2, 2.0, 1.0])


def test_first_rule_wins(targets_df):
    rules = [{"group": "H3K9me3", "weight": 0.1}, {"assay": "chip", "weight": 0.2}]
    assert np.allclose(parse_target_weights(targets_df, rules), [0.2, 0.1, 1.0, 1.0])


def test_multi_column_selector(targets_df):
    rules = [{"assay": "chip", "group": "H3K4me3", "weight": 0.3}]
    assert np.allclose(parse_target_weights(targets_df, rules), [0.3, 1.0, 1.0, 1.0])


def test_unknown_column_raises(targets_df):
    with pytest.raises(ValueError, match="not in targets"):
        parse_target_weights(targets_df, [{"nonesuch": "chip", "weight": 0.2}])


def test_selectorless_rule_raises(targets_df):
    with pytest.raises(ValueError, match="selector"):
        parse_target_weights(targets_df, [{"weight": 0.2}])


def test_unmatched_rule_warns(targets_df, capsys):
    parse_target_weights(targets_df, [{"assay": "cage", "weight": 0.2}])
    assert "matched no targets" in capsys.readouterr().out


def test_shadowed_rule_does_not_warn(capsys):
    """A general rule fully claimed by a specific one is the documented idiom."""
    targets_df = pd.DataFrame({"assay": ["chip"] * 2, "group": ["H3K9me3"] * 2})
    rules = [{"group": "H3K9me3", "weight": 0.1}, {"assay": "chip", "weight": 0.2}]
    assert np.allclose(parse_target_weights(targets_df, rules), [0.1, 0.1])
    assert capsys.readouterr().out == ""


@pytest.fixture
def loss_inputs():
    torch.manual_seed(0)
    y_true = torch.rand(2, 4, 32) * 10
    y_pred = torch.rand(2, 4, 32) * 10
    return y_pred, y_true


def test_unit_weights_match_unweighted(loss_inputs):
    y_pred, y_true = loss_inputs
    loss_fn = PoissonMultinomialLoss(total_weight=0.25)
    plain = loss_fn(y_pred.clone(), y_true.clone())
    weighted = loss_fn(y_pred.clone(), y_true.clone(), torch.ones(1, 4))
    assert torch.allclose(plain, weighted)


def test_weighted_mean_of_per_track_losses(loss_inputs):
    """The weighted reduction is the weight-average of the per-track losses."""
    y_pred, y_true = loss_inputs
    weights = torch.tensor([[0.2, 1.0, 1.0, 3.0]])

    per_track = PoissonMultinomialLoss(total_weight=0.25, reduction="none")(
        y_pred.clone(), y_true.clone()
    )
    expect = (per_track * weights).sum() / (weights.sum() * per_track.shape[0])

    loss_fn = PoissonMultinomialLoss(total_weight=0.25)
    assert torch.allclose(loss_fn(y_pred.clone(), y_true.clone(), weights), expect)


def test_scaling_all_weights_is_a_noop(loss_inputs):
    """A common factor cancels in the weighted mean, so only ratios matter."""
    y_pred, y_true = loss_inputs
    loss_fn = PoissonMultinomialLoss(total_weight=0.25)
    weights = torch.tensor([[0.2, 1.0, 1.0, 3.0]])
    one = loss_fn(y_pred.clone(), y_true.clone(), weights)
    ten = loss_fn(y_pred.clone(), y_true.clone(), weights * 10)
    assert torch.allclose(one, ten)


@pytest.mark.parametrize("label", ["poisson", "poisson_mn"])
def test_every_loss_takes_weights(loss_inputs, label):
    y_pred, y_true = loss_inputs
    loss_fn = parse_loss(label, total_weight=0.25)
    plain = loss_fn(y_pred.clone(), y_true.clone(), None)
    weighted = loss_fn(y_pred.clone(), y_true.clone(), torch.ones(1, 4))
    assert torch.allclose(plain, weighted)


def test_poisson_matches_torch(loss_inputs):
    """Unweighted PoissonLoss reproduces the torch loss it replaces."""
    y_pred, y_true = loss_inputs
    torch_fn = torch.nn.PoissonNLLLoss(log_input=False)
    assert torch.allclose(PoissonLoss()(y_pred, y_true), torch_fn(y_pred, y_true))


@pytest.mark.parametrize("weighted", [False, True])
def test_poisson_autocast_handles_zero_predictions(weighted):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_pred = torch.tensor(
        [[[0.0, 1e-7, 1.0]]], device=device, dtype=torch.float16, requires_grad=True
    )
    y_true = torch.zeros_like(y_pred, dtype=torch.float32)
    weights = torch.tensor([[0.5]], device=device) if weighted else None
    with torch.autocast(device_type=device, dtype=torch.float16):
        expected = torch.nn.PoissonNLLLoss(log_input=False)(y_pred, y_true)
        actual = PoissonLoss()(y_pred, y_true, weights)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(y_pred.grad).all()


def test_poisson_weights_broadcast_over_length(loss_inputs):
    """(1, T) weights weight tracks, not positions, in a B x T x L loss."""
    y_pred, y_true = loss_inputs
    weights = torch.tensor([[0.2, 1.0, 1.0, 3.0]])

    per_element = PoissonLoss(reduction="none")(y_pred, y_true)
    per_track = per_element.mean(dim=-1)
    expect = (per_track * weights).sum() / (weights.sum() * per_track.shape[0])

    assert torch.allclose(PoissonLoss()(y_pred, y_true, weights), expect)


@pytest.fixture
def gene_inputs():
    """B x num_gene_targets x max_genes, with the last gene absent from batch 1."""
    torch.manual_seed(0)
    y_true = torch.rand(2, 4, 5) * 10
    y_pred = torch.rand(2, 4, 5) * 10
    gene_presence = torch.ones(2, 5)
    gene_presence[1, -1] = 0
    return y_pred, y_true, gene_presence


@pytest.mark.parametrize("loss_cls", [GenePoissonLoss, GeneMSELoss])
def test_gene_unit_weights_match_unweighted(gene_inputs, loss_cls):
    y_pred, y_true, gene_presence = gene_inputs
    loss_fn = loss_cls()
    plain = loss_fn(y_pred, y_true, gene_presence)
    weighted = loss_fn(y_pred, y_true, gene_presence, torch.ones(1, 4))
    assert torch.allclose(plain, weighted)


@pytest.mark.parametrize("loss_cls", [GenePoissonLoss, GeneMSELoss])
def test_gene_weighted_mean_of_per_track_losses(gene_inputs, loss_cls):
    """Gene weights average the masked per-track losses, as on the coverage head."""
    y_pred, y_true, gene_presence = gene_inputs
    weights = torch.tensor([[0.2, 1.0, 1.0, 3.0]])

    per_track = loss_cls(reduction="none")(y_pred, y_true, gene_presence)
    expect = (per_track * weights).sum() / (weights.sum() * per_track.shape[0])

    loss = loss_cls()(y_pred, y_true, gene_presence, weights)
    assert torch.allclose(loss, expect)
