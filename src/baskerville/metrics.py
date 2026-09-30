import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torchmetrics

from baskerville import dataset


class Loss:
    """Basic mean metric."""

    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, loss, n):
        """Update metric state for a batch."""
        self._sum += loss
        self._count += n

    def compute(self):
        """Compute Loss result from state."""
        return self._sum / self._count

    def reset(self):
        """Reset metric state."""
        self._sum = 0
        self._count = 0


class PearsonCorrCoef(torchmetrics.Metric):
    """
    Metric class to calculate the Pearson correlation coefficient for each task.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(self, num_outputs: int = 1, average: bool = True) -> None:
        super().__init__()
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        target = target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        self.pearson.update(preds, target)

    def compute(self) -> torch.Tensor:
        output = self.pearson.compute()
        if self.average:
            return output.nanmean()
        else:
            return output

    def reset(self):
        self.pearson.reset()


class R2Score(torchmetrics.Metric):
    """
    Metric class to calculate the R2 score for each task.

    Args:
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    n_tasks is detected automatically from the input.
    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(self, average: bool = True) -> None:
        super().__init__()
        self.r2score = torchmetrics.R2Score(multioutput="raw_values")
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        target = target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1)  # NxL, n_tasks
        self.r2score.update(preds, target)

    def compute(self) -> torch.Tensor:
        output = self.r2score.compute()
        if self.average:
            return output.nanmean()
        else:
            return output

    def reset(self):
        self.r2score.reset()


class SpecPearsonCorrCoef:
    """Per-track specificity Pearson within target groups.

    For each group (dataset.target_groups) with at least ``group_min`` tracks
    after strand collapse: sum strand pairs, divide preds and targets by the
    dataset's track means (SeqDataset.target_means), subtract the group mean
    at each position, and correlate each track's residuals.

    Args:
        targets_df: targets table aligned with the output tracks.
        target_means: per-track means of the stored targets, or None.
        group_min: minimum tracks per group.

    ``groups`` maps group name to (rep, pair) track positions. compute()
    returns a per-track tensor with values at rep positions, NaN elsewhere.
    Without target_means, ``enabled`` is False and nothing is scored.
    """

    def __init__(self, targets_df, target_means, group_min: int = 20):
        self.num_targets = len(targets_df)
        rep_pos, pair_pos = dataset.strand_collapse(targets_df)
        rep_groups = dataset.target_groups(targets_df)[rep_pos]
        self.groups = {}
        for g in sorted(set(rep_groups)):
            gi = rep_groups == g
            if gi.sum() >= group_min:
                self.groups[g] = (rep_pos[gi], pair_pos[gi])

        self.enabled = target_means is not None
        if not self.enabled:
            print("Warning: data lack target means; spec metric disabled.")
            return

        means = np.asarray(target_means)
        self.index, self.b, self.pearson = {}, {}, {}
        for g, (rep, pair) in self.groups.items():
            self.index[g] = (torch.tensor(rep), torch.tensor(pair))
            # unstranded tracks have pair == rep, so the doubling cancels
            b = np.maximum(means[rep] + means[pair], 1e-6)
            self.b[g] = torch.tensor(b, dtype=torch.float32).view(1, -1, 1)
            self.pearson[g] = PearsonCorrCoef(len(rep), average=False)

    def to(self, device):
        if self.enabled:
            for g in self.groups:
                self.index[g] = tuple(ix.to(device) for ix in self.index[g])
                self.b[g] = self.b[g].to(device)
                self.pearson[g].to(device)
        return self

    def _center(self, x, g):
        rep, pair = self.index[g]
        x = (x[:, rep].float() + x[:, pair].float()) / self.b[g]
        return x - x.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """preds, target: (N, T, L)."""
        if self.enabled:
            for g in self.groups:
                self.pearson[g].update(self._center(preds, g), self._center(target, g))

    def compute(self) -> torch.Tensor:
        spec = torch.full((self.num_targets,), float("nan"))
        if self.enabled:
            for g in self.groups:
                spec[self.index[g][0].cpu()] = self.pearson[g].compute().float().cpu()
        return spec

    def reset(self):
        if self.enabled:
            for p in self.pearson.values():
                p.reset()


class _MaskedGeneMetric(torchmetrics.Metric):
    """Base class for gene metrics with masking support.

    Subclasses set self.inner_metric in __init__.

    Input shapes:
        preds: (B, num_gene_targets, max_genes)
        target: (B, num_gene_targets, max_genes)
        mask: (B, max_genes) - boolean mask for valid genes
    """

    def __init__(self, average: bool = True) -> None:
        super().__init__()
        self.average = average

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        preds_flat = preds.permute(0, 2, 1).reshape(-1, preds.shape[1])
        target_flat = target.permute(0, 2, 1).reshape(-1, target.shape[1])
        mask_flat = mask.reshape(-1)

        preds_valid = preds_flat[mask_flat]
        target_valid = target_flat[mask_flat]

        if preds_valid.shape[0] > 0:
            self.inner_metric.update(preds_valid, target_valid)

    def compute(self) -> torch.Tensor:
        output = self.inner_metric.compute()
        return output.nanmean() if self.average else output

    def reset(self):
        self.inner_metric.reset()


class GenePearsonCorrCoef(_MaskedGeneMetric):
    """Pearson correlation for gene predictions with masking support."""

    def __init__(self, num_outputs: int = 1, average: bool = True) -> None:
        super().__init__(average=average)
        self.inner_metric = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)


class GeneR2Score(_MaskedGeneMetric):
    """R2 score for gene predictions with masking support."""

    def __init__(self, average: bool = True) -> None:
        super().__init__(average=average)
        self.inner_metric = torchmetrics.R2Score(multioutput="raw_values")


def weighted_reduce(loss, target_weights, reduction):
    """Reduce a per-track loss, optionally weighting the tracks.

    Args:
        loss: B x T x ... unreduced loss.
        target_weights: optional (1, T) per-track weights, broadcast over any
          trailing dimensions.
        reduction: "mean", "sum", or "none".

    The "mean" reduction is the weighted mean, so the loss stays a per-track
    average and gradient clipping keeps its meaning across weightings.
    """
    if target_weights is not None:
        shape = target_weights.shape + (1,) * (loss.dim() - target_weights.dim())
        target_weights = target_weights.reshape(shape)
        weighted = loss * target_weights

        if reduction == "mean":
            denom = target_weights.sum() * (loss.numel() // target_weights.numel())
            return weighted.sum() / denom
        loss = weighted

    if reduction == "mean":
        # mean(), not sum()/numel(): the two are not bit-identical in half
        # precision, and the unweighted path must reproduce the plain reduction
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


class PoissonLoss(_Loss):
    """Poisson negative log likelihood, with optional per-track loss weights.

    Matches torch.nn.PoissonNLLLoss(log_input=False) when unweighted.
    """

    def __init__(self, epsilon: float = 1e-8, reduction: str = "mean"):
        super(PoissonLoss, self).__init__(reduction=reduction)
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, target_weights=None):
        loss = nn.functional.poisson_nll_loss(
            y_pred, y_true, log_input=False, eps=self.epsilon, reduction="none"
        )
        return weighted_reduce(loss, target_weights, self.reduction)


class PoissonMultinomialLoss(_Loss):
    """Poisson decomposition with multinomial specificity term."""

    def __init__(
        self,
        total_weight: float = 1,
        weight_range: float = 1,
        weight_exp: int = 4,
        epsilon: float = 1e-9,
        rescale: bool = False,
        reduction: str = "mean",
    ):
        super(PoissonMultinomialLoss, self).__init__(reduction=reduction)
        self.total_weight = total_weight
        self.weight_range = weight_range
        self.weight_exp = weight_exp
        self.epsilon = epsilon
        self.rescale = rescale
        self.register_buffer("position_weights", None)
        self.register_buffer("weights_sum", None)

    def _poisson(self, yt, yp):
        """Poisson loss, without mean reduction."""
        return yp - yt * torch.log(yp + self.epsilon)

    def _setup_position_weights(self, seq_len, device):
        """Construct position weight vectors from sequence length."""
        if self.position_weights is None:
            pos_start = -(seq_len / 2 - 0.5)
            pos_end = seq_len / 2 + 0.5
            sigma = -pos_start / (np.log(self.weight_range)) ** (1 / self.weight_exp)

            positions = torch.arange(
                pos_start, pos_end, dtype=torch.float32, device=device
            )
            weights = torch.exp(-((positions / sigma) ** self.weight_exp))
            weights = (weights / weights.max()).view(1, 1, -1)
            self.register_buffer("position_weights", weights)
            self.register_buffer("weights_sum", weights.sum())

    def forward(self, y_pred, y_true, target_weights=None):
        """Args:
        y_pred, y_true: B x T x L predictions and targets.
        target_weights: optional (1, T) per-track loss weights.
        """
        seq_len = y_true.shape[2]
        self._setup_position_weights(seq_len, y_true.device)

        # apply position weights
        y_true *= self.position_weights
        y_pred *= self.position_weights

        # sum across lengths
        s_true = y_true.sum(dim=-1)  # B x T
        s_pred = y_pred.sum(dim=-1)  # B x T

        # total count poisson loss, mean across targets
        poisson_term = self._poisson(s_true, s_pred)  # B x T
        poisson_term /= self.weights_sum

        # add pred eps, and normalize to sum to one
        p_pred = y_pred + self.epsilon
        p_pred /= p_pred.sum(dim=-1, keepdim=True)

        # multinomial loss
        # pl_pred = torch.log(p_pred)  # B x T x L
        # multinomial_dot = -y_true * pl_pred  # B x T x L
        # multinomial_term = multinomial_dot.sum(dim=-1)  # B x T
        multinomial_term = -(y_true * torch.log(p_pred)).sum(dim=-1)
        multinomial_term /= self.weights_sum

        # normalize to scale of 1:1 term ratio
        loss_raw = self.total_weight * poisson_term + multinomial_term
        if self.rescale:
            loss = loss_raw * 2 / (1 + self.total_weight)
        else:
            loss = loss_raw

        return weighted_reduce(loss, target_weights, self.reduction)


class DatasetMetrics:
    """Encapsulates all metrics for a single dataset (train or valid).

    Bundles loss, coverage metrics (r, r2, spec), and gene metrics (r_gene,
    r2_gene) with methods for update, compute, reset, and log formatting.

    Args:
        has_coverage: Whether dataset has coverage tracks.
        has_gene: Whether dataset has gene expression data.
        num_targets: Number of coverage targets (required if has_coverage).
        num_gene_targets: Number of gene targets (required if has_gene).
        device: Device to place metrics on.
        targets_df: Coverage targets table; enables per-group metrics when it
            matches num_targets.
        target_means: Per-track means of the stored targets; enables spec.
        spec_group_min: Minimum tracks for a group to be scored.
    """

    def __init__(
        self,
        has_coverage: bool,
        has_gene: bool,
        num_targets: int | None,
        num_gene_targets: int | None,
        device: str,
        targets_df=None,
        target_means=None,
        spec_group_min: int = 20,
    ):
        self.has_coverage = has_coverage
        self.has_gene = has_gene

        # Always have loss
        self.loss = Loss()

        # Coverage metrics
        self.spec = None
        if has_coverage and num_targets is not None:
            self.r = PearsonCorrCoef(num_targets, average=False).to(device)
            self.r2 = R2Score(average=False).to(device)
            if targets_df is not None and len(targets_df) == num_targets:
                self.spec = SpecPearsonCorrCoef(
                    targets_df, target_means, spec_group_min
                ).to(device)
        else:
            self.r = None
            self.r2 = None

        # Gene metrics
        if has_gene and num_gene_targets is not None:
            self.r_gene = GenePearsonCorrCoef(num_gene_targets).to(device)
            self.r2_gene = GeneR2Score().to(device)
        else:
            self.r_gene = None
            self.r2_gene = None

    def update(
        self,
        yh,
        y: torch.Tensor | None,
        yg: torch.Tensor | None,
        gene_presence: torch.Tensor | None,
        loss_val: torch.Tensor,
        batch_size: int,
    ):
        """Update all metrics for a batch.

        Args:
            yh: Model output with coverage and gene predictions.
            y: Coverage targets (can be None for gene-only).
            yg: Gene expression targets (can be None).
            gene_presence: Mask for valid genes (can be None).
            loss_val: Loss tensor for this batch.
            batch_size: Number of samples in batch.
        """
        self.loss.update(loss_val.item(), batch_size)

        # Coverage metrics
        if y is not None and yh.has_coverage and self.r is not None:
            self.r.update(yh.coverage, y)
            self.r2.update(yh.coverage, y)
            if self.spec is not None:
                self.spec.update(yh.coverage, y)

        # Gene metrics
        if yg is not None and self.r_gene is not None and yh.has_gene:
            self.r_gene.update(yh.gene, yg, gene_presence)
            self.r2_gene.update(yh.gene, yg, gene_presence)

    def compute(self) -> dict:
        """Compute all metrics and return as a flat dict of floats.

        Keys loss, r, r2, r_gene, r2_gene are always present (None if not
        applicable). With per-group metrics: spec (mean over groups) and, per
        group g, r/g and r2/g (over all group tracks) and spec/g.
        """
        results = {"loss": self.loss.compute()}

        if self.r is not None:
            r = self.r.compute().float().cpu()
            r2 = self.r2.compute().float().cpu()
            results["r"] = r.nanmean().item()
            results["r2"] = r2.nanmean().item()
            if self.spec is not None:
                spec = self.spec.compute()
                for g, (rep, pair) in self.spec.groups.items():
                    tracks = np.union1d(rep, pair)
                    results[f"r/{g}"] = r[tracks].nanmean().item()
                    results[f"r2/{g}"] = r2[tracks].nanmean().item()
                    if self.spec.enabled:
                        results[f"spec/{g}"] = spec[rep].nanmean().item()
                spec_groups = [v for k, v in results.items() if k.startswith("spec/")]
                if spec_groups:
                    results["spec"] = float(np.mean(spec_groups))
        else:
            results["r"] = None
            results["r2"] = None

        if self.r_gene is not None:
            results["r_gene"] = self.r_gene.compute().item()
            results["r2_gene"] = self.r2_gene.compute().item()
        else:
            results["r_gene"] = None
            results["r2_gene"] = None

        return results

    def reset(self):
        """Reset all metrics."""
        self.loss.reset()
        if self.r is not None:
            self.r.reset()
            self.r2.reset()
        if self.spec is not None:
            self.spec.reset()
        if self.r_gene is not None:
            self.r_gene.reset()
            self.r2_gene.reset()

    def format_log(self, prefix: str, results: dict = None) -> str:
        """Format summary metrics for logging.

        Args:
            prefix: Prefix for metric names (e.g., 'train' or 'valid').
            results: Pre-computed results dict. If None, calls compute().

        Returns:
            Formatted string like "train_loss: 0.12345 - train_r: 0.8765 - ..."
        """
        if results is None:
            results = self.compute()
        parts = [f"{prefix}_loss: {results['loss']:.5f}"]

        if results["r"] is not None:
            parts.append(f"{prefix}_r: {results['r']:.4f}")
            parts.append(f"{prefix}_r2: {results['r2']:.4f}")
        if "spec" in results:
            parts.append(f"{prefix}_spec: {results['spec']:.4f}")

        if results["r_gene"] is not None:
            parts.append(f"{prefix}_r_gene: {results['r_gene']:.4f}")
            parts.append(f"{prefix}_r2_gene: {results['r2_gene']:.4f}")

        return " - ".join(parts)


class _MaskedGeneLoss(_Loss):
    """Base class for gene expression losses with masking and reduction.

    Subclasses implement _element_loss(y_pred, y_true) for per-element loss.

    Args:
        reduction: Reduction method ('mean', 'sum', or 'none').
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def _element_loss(self, y_pred, y_true):
        raise NotImplementedError

    def forward(self, y_pred, y_true, gene_presence=None, target_weights=None):
        """
        Args:
            y_pred: Predicted gene expression (B, num_gene_targets, max_genes)
            y_true: True gene expression (B, num_gene_targets, max_genes)
            gene_presence: Boolean mask for valid genes (B, max_genes)
            target_weights: optional (1, num_gene_targets) per-track loss weights.

        Returns:
            Loss value
        """
        loss = self._element_loss(y_pred, y_true)

        if gene_presence is not None:
            mask = gene_presence.unsqueeze(1)  # (B, 1, max_genes)
            loss = loss * mask
            counts = gene_presence.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            loss = loss.sum(dim=2) / counts  # (B, num_gene_targets)
        else:
            loss = loss.mean(dim=2)  # (B, num_gene_targets)

        return weighted_reduce(loss, target_weights, self.reduction)


class GenePoissonLoss(_MaskedGeneLoss):
    """Poisson loss for gene expression prediction."""

    def __init__(self, epsilon: float = 1e-9, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.epsilon = epsilon

    def _element_loss(self, y_pred, y_true):
        return y_pred - y_true * torch.log(y_pred + self.epsilon)


class GeneMSELoss(_MaskedGeneLoss):
    """MSE loss for gene expression prediction."""

    def _element_loss(self, y_pred, y_true):
        return (y_pred - y_true) ** 2
