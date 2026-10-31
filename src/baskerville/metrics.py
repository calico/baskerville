import copy

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


def qmap_tables(hists):
    """Quantile-normalization lookup tables from per-track value histograms.

    Exactly qnorm's map for the histogrammed data: the group reference is the
    average of the tracks' sorted columns, and a value occupying ranks [s, e)
    in its track maps to the mean reference over [s, e) (ties averaged). Values
    absent from a track interpolate between its present values and clamp
    outside them, so any fp16 prediction has an entry.

    Args:
        hists: (T, dataset.NUM_HIST_BINS) counts per fp16 bit pattern, with
            equal totals.

    Returns:
        np.ndarray (T, NUM_HIST_BINS) float32 tables, indexed by bit pattern.
    """
    hists = np.asarray(hists, dtype=np.int64)
    totals = hists.sum(axis=1)
    if totals.min() == 0 or totals.min() != totals.max():
        raise ValueError("target histograms must have equal nonzero totals")
    values = np.arange(dataset.NUM_HIST_BINS, dtype=np.uint16).view(np.float16)
    values = values.astype(np.float64)

    # per track: present values and the rank bounds of their tie blocks
    knots = []
    for h in hists:
        nz = np.flatnonzero(h)
        knots.append((nz, np.concatenate([[0], np.cumsum(h[nz])])))

    # the reference (mean sorted column) is a step function of rank: it starts
    # at the mean minimum and steps by (value change) / T at every track's
    # block bound; sweep all bounds in order for its running sum
    num_tracks, num_ranks = len(hists), totals[0]
    pos = np.concatenate([b[1:-1] for _, b in knots])
    step = np.concatenate([np.diff(values[nz]) for nz, _ in knots])
    order = np.argsort(pos, kind="stable")
    ref_pos = np.concatenate([[0], pos[order], [num_ranks]])
    ref_level = sum(values[nz[0]] for nz, _ in knots)
    ref_level = (
        ref_level + np.concatenate([[0.0], np.cumsum(step[order])])
    ) / num_tracks
    ref_sum = np.concatenate([[0.0], np.cumsum(ref_level * np.diff(ref_pos))])

    tables = np.empty(hists.shape, dtype=np.float32)
    for t, (nz, bounds) in enumerate(knots):
        block_sum = np.diff(np.interp(bounds, ref_pos, ref_sum))
        tables[t] = np.interp(values, values[nz], block_sum / np.diff(bounds))
    return tables


def qmap_knots(hists, num_knots: int = 256, tables=None):
    """Piecewise-linear knots approximating each track's qmap_tables map.

    x knots are present values at quantile levels 1 − 10^(−s), s evenly spaced
    over [0, 8], plus the last present value: dense in the upper tail, where
    evenly spaced quantiles would leave a track's peaks a few segments. q knots
    are the table values there, unaltered, so flat
    stretches of the reference stay flat. A constant track gets two knots with
    equal q. Tracks with fewer knots are padded past the last one with q held.

    Args:
        tables: qmap_tables(hists), if already computed.

    Returns:
        (xk, qk) np.ndarray (T, num_knots) float64.
    """
    hists = np.asarray(hists)
    tables = qmap_tables(hists) if tables is None else tables
    values = np.arange(dataset.NUM_HIST_BINS, dtype=np.uint16).view(np.float16)
    values = values.astype(np.float64)
    levels = np.append(1 - 10 ** -np.linspace(0, 8, num_knots - 1), 1)
    xk, qk = np.empty((len(hists), num_knots)), np.empty((len(hists), num_knots))
    for t, h in enumerate(hists):
        nz = np.flatnonzero(h)
        cdf = np.cumsum(h[nz]) / h[nz].sum()
        idx = np.searchsorted(cdf, levels)
        idx = nz[np.unique(np.minimum(idx, len(nz) - 1))]
        x, q = values[idx], tables[t, idx].astype(np.float64)
        if len(x) == 1:
            x, q = np.array([x[0], x[0] + 1]), np.array([q[0], q[0]])
        pad = np.arange(1, num_knots - len(x) + 1)
        xk[t] = np.concatenate([x, x[-1] + pad])
        qk[t] = np.concatenate([q, np.full(len(pad), q[-1])])
    return xk, qk


def qmap_apply(x, xk, qk):
    """Map x (T, N) through per-track knots (T, K); clamped outside the knots."""
    x = torch.clamp(x, xk[:, :1], xk[:, -1:]).contiguous()
    i1 = torch.searchsorted(xk, x).clamp(1, xk.shape[1] - 1)
    i0 = i1 - 1
    x0, x1 = xk.gather(1, i0), xk.gather(1, i1)
    q0, q1 = qk.gather(1, i0), qk.gather(1, i1)
    return q0 + (q1 - q0) / (x1 - x0) * (x - x0)


class QmapCrossTrackLoss(nn.Module):
    """Cross-track specificity loss in differentiable quantile-mapped space.

    For each group of SpecPearsonCorrCoef (strand sums, >= group_min tracks),
    map preds and targets through qmap_knots. With a, b the group-centered
    mapped pred and target (the group mean at each position subtracted), the
    per-track loss over the batch's positions (position-weighted) is, by form:

    - mse: mean of (a - b)^2.
    - dot: mean of (a - b)^2 - a^2, i.e. mse without the penalty on a's own
      magnitude; the gradient is -2b.
    - proj: (|b| - a.b / |a|) / sqrt(n), i.e. |b|(1 - cos(a, b)); a's direction
      only. |a| is floored at 0.1 |b|, below which proj acts like dot.
    - mn: the multinomial across the group's tracks at each position, on
      uncentered mapped values: sum_t (z_t + e) log(q_t / p_t), with q, p the
      target and pred shares of the group sum, e 1e-3 of the group's mean
      mapped value. Zero at pred = target; weighted by the group's target sum.

    Groups weigh equally: each group's loss is divided by its mapped variance
    (mse, dot), SD (proj) or mean (mn), from the table values under each
    track's histogram, averaged over the group. Inputs past a track's observed
    values clamp, as in qmap_tables, so they carry no gradient.

    Args:
        targets_df: targets table aligned with the output tracks.
        target_hist: per-track histograms of the stored targets.
        weight_range, weight_exp: position weights, as in PoissonMultinomialLoss.
        form: mse, dot, proj or mn.
    """

    FORMS = ("mse", "dot", "proj", "mn")

    def __init__(self, targets_df, target_hist, weight_range=1, weight_exp=4,
                 group_min: int = 20, num_knots: int = 256, form: str = "mse"):
        super().__init__()
        if form not in self.FORMS:
            raise ValueError(f"qmap form {form!r} is not one of {self.FORMS}")
        self.form = form
        self.weight_range, self.weight_exp = weight_range, weight_exp
        rep_pos, pair_pos = dataset.strand_collapse(targets_df)
        groups = dataset.target_groups(targets_df)[rep_pos]
        hist = np.asarray(target_hist)
        rep, pair, gid, xk, qk, var, avg = [], [], [], [], [], [], []
        self.group_names = names = []
        for g in sorted(set(groups)):
            gi = groups == g
            if gi.sum() >= group_min:
                h = hist[rep_pos[gi]].astype(np.float64)
                tables = qmap_tables(hist[rep_pos[gi]]).astype(np.float64)
                x, q = qmap_knots(hist[rep_pos[gi]], num_knots, tables)
                mean = (h * tables).sum(1, keepdims=True) / h.sum(1, keepdims=True)
                v = ((h * (tables - mean) ** 2).sum(1) / h.sum(1)).mean()
                rep.append(rep_pos[gi]), pair.append(pair_pos[gi]), names.append(g)
                gid.append(np.full(gi.sum(), len(xk))), xk.append(x), qk.append(q)
                var.append(np.full(gi.sum(), v))
                avg.append(np.full(gi.sum(), mean.mean()))
        if not rep:
            raise ValueError(f"no target group has >= {group_min} tracks")
        # per-track weight 1 / (groups · group size · group mapped scale)
        var, avg = np.concatenate(var), np.concatenate(avg)
        scale = {"mse": var, "dot": var, "proj": np.sqrt(var), "mn": avg}[form]
        size = np.bincount(np.concatenate(gid))
        weight = np.where(scale > 0, 1 / (len(size) * size[np.concatenate(gid)] * scale), 0)
        gid = torch.from_numpy(np.concatenate(gid))
        member = nn.functional.one_hot(gid).double()
        self.register_buffer("rep", torch.from_numpy(np.concatenate(rep)))
        self.register_buffer("pair", torch.from_numpy(np.concatenate(pair)))
        self.register_buffer("gid", gid)
        self.register_buffer("mean_op", (member / member.sum(0)).float())  # (Tq, G)
        self.register_buffer("xk", torch.from_numpy(np.concatenate(xk)).float())
        self.register_buffer("qk", torch.from_numpy(np.concatenate(qk)).float())
        self.register_buffer("track_weight", torch.from_numpy(weight).float())
        self.register_buffer("eps", torch.from_numpy(1e-3 * avg).float().view(-1, 1, 1))
        self.register_buffer("position_weights", None)

    def _mapped(self, x):
        """(B, T, L) -> (Tq, B, L) qmap values of strand sums."""
        s = (x[:, self.rep] + x[:, self.pair]).transpose(0, 1)
        return qmap_apply(s.flatten(1), self.xk, self.qk).view(s.shape)

    def _group_mean(self, z):
        """(Tq, B, L) -> (Tq, B, L) each track's group mean at each position."""
        return torch.einsum("tg,tbl->gbl", self.mean_op, z)[self.gid]

    def per_track(self, zp, zy):
        """(Tq,) per-track loss of mapped preds zp and targets zy, (Tq, B, L)."""
        w = self.position_weights.view(1, 1, -1)
        n = zp.shape[1] * w.sum()

        def wsum(x):
            return (w * x).sum((1, 2))

        if self.form == "mn":
            e = self.eps
            log_q = torch.log((zy + e) / (self._group_mean(zy) + e))
            log_p = torch.log((zp + e) / (self._group_mean(zp) + e))
            return wsum((zy + e) * (log_q - log_p)) / n
        a, b = zp - self._group_mean(zp), zy - self._group_mean(zy)
        if self.form == "mse":
            return wsum((a - b).pow(2)) / n
        if self.form == "dot":
            return wsum((a - b).pow(2) - a.pow(2)) / n
        bb = wsum(b.pow(2))
        aa = wsum(a.pow(2)).clamp_min((0.01 * bb).clamp_min(1e-20))
        return (bb.sqrt() - wsum(a * b) / aa.sqrt()) / n.sqrt()

    def forward(self, y_pred, y_true):
        if self.position_weights is None:
            self.register_buffer("position_weights", position_weights(
                y_true.shape[-1], self.weight_range, self.weight_exp, y_true.device
            ))
        # fp32 throughout: the group mean is a matmul, which autocast would run in bf16
        with torch.autocast(y_true.device.type, enabled=False):
            with torch.no_grad():
                zy = self._mapped(y_true.float())
            zp = self._mapped(y_pred.float())
            return (self.track_weight * self.per_track(zp, zy)).sum()


class SpecPearsonCorrCoef:
    """Per-track specificity Pearson within target groups.

    For each group (dataset.target_groups) with at least ``group_min`` tracks
    after strand collapse: sum strand pairs, map preds and targets through each
    track's quantile-normalization table (qmap_tables on the dataset's
    whole-genome target histograms, SeqDataset.target_hist), regress the group
    mean at each position out of each track (its own slope and intercept), and
    correlate each track's pred and target residuals. Streaming state is the
    first and second moments of (pred, target, group-mean pred, group-mean
    target): per track for terms involving the track, per group otherwise.

    Args:
        targets_df: targets table aligned with the output tracks.
        target_hist: per-track histograms of the stored targets, or None.
        group_min: minimum tracks per group.

    ``groups`` maps group name to (rep, pair) track positions. compute()
    returns a per-track tensor with values at rep positions, NaN elsewhere.
    Without target_hist, ``enabled`` is False and nothing is scored.
    """

    def __init__(self, targets_df, target_hist, group_min: int = 20):
        self.num_targets = len(targets_df)
        rep_pos, pair_pos = dataset.strand_collapse(targets_df)
        rep_groups = dataset.target_groups(targets_df)[rep_pos]
        self.groups = {}
        for g in sorted(set(rep_groups)):
            gi = rep_groups == g
            if gi.sum() >= group_min:
                self.groups[g] = (rep_pos[gi], pair_pos[gi])

        self.enabled = target_hist is not None
        if not self.enabled:
            print("Warning: data lack target histograms; spec metric disabled.")
            return

        hist = np.asarray(target_hist)
        self.index, self.table = {}, {}
        for g, (rep, pair) in self.groups.items():
            self.index[g] = (torch.tensor(rep), torch.tensor(pair))
            # histograms are of the strand sum, stored at both partners
            self.table[g] = torch.from_numpy(qmap_tables(hist[rep]))
        self.reset()

    def fresh(self):
        """A new metric with zeroed state, sharing this one's groups and tables."""
        new = copy.copy(self)
        if self.enabled:
            new.index, new.table = dict(self.index), dict(self.table)
            new.reset()
        return new

    def to(self, device):
        if self.enabled:
            for g in self.groups:
                self.index[g] = tuple(ix.to(device) for ix in self.index[g])
                self.table[g] = self.table[g].to(device)
                self.track_sums[g] = self.track_sums[g].to(device)
                self.mean_sums[g] = self.mean_sums[g].to(device)
        return self

    def _map(self, x, g):
        """(N, T, L) -> (tracks, N*L) table values of the group's strand sums."""
        rep, pair = self.index[g]
        s = torch.nan_to_num(x[:, rep].float() + x[:, pair].float(), nan=0.0)
        bits = (s.clamp(0, 65504) + 0.0).half().view(torch.int16).long()
        return torch.gather(self.table[g], 1, bits.transpose(0, 1).flatten(1))

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """preds, target: (N, T, L)."""
        if self.enabled:
            for g in self.groups:
                # Residual covariance subtracts large, nearly equal moments.
                zp = self._map(preds, g).double()  # (tracks, P)
                zy = self._map(target, g).double()
                mp, my = zp.mean(dim=0), zy.mean(dim=0)  # (P,)

                def total(x):
                    return x.sum(dim=-1, dtype=torch.float64)

                self.track_sums[g] += torch.stack(
                    [total(zp), total(zy), total(zp * zp), total(zy * zy)]
                    + [total(zp * zy), total(zp * mp), total(zp * my)]
                    + [total(zy * mp), total(zy * my)],
                    dim=-1,
                )
                self.mean_sums[g] += torch.stack(
                    [total(mp), total(my), total(mp * mp), total(my * my)]
                    + [total(mp * my)]
                )
                self.count[g] += zp.shape[1]

    def compute(self) -> torch.Tensor:
        spec = torch.full((self.num_targets,), float("nan"))
        if self.enabled:
            for g in self.groups:
                n = self.count[g]
                ts, ms = self.track_sums[g].cpu(), self.mean_sums[g].cpu()
                # assemble per-track sums over (pred, target, mean pred, mean target)
                s1 = torch.stack(
                    [ts[:, 0], ts[:, 1], ms[0].expand(len(ts)), ms[1].expand(len(ts))],
                    dim=-1,
                )
                s2 = torch.empty(len(ts), 4, 4, dtype=torch.float64)
                pairs = {
                    (0, 0): ts[:, 2],
                    (1, 1): ts[:, 3],
                    (0, 1): ts[:, 4],
                    (0, 2): ts[:, 5],
                    (0, 3): ts[:, 6],
                    (1, 2): ts[:, 7],
                    (1, 3): ts[:, 8],
                    (2, 2): ms[2],
                    (3, 3): ms[3],
                    (2, 3): ms[4],
                }
                for (a, b), v in pairs.items():
                    s2[:, a, b] = s2[:, b, a] = v
                cov = s2 / n - s1[:, :, None] * s1[:, None, :] / n**2
                # residuals of pred/target after regressing out their group mean
                u = torch.zeros(len(cov), 4, dtype=torch.float64)
                w = torch.zeros_like(u)
                u[:, 0] = w[:, 1] = 1
                # A constant group mean removes only the intercept.
                u[:, 2] = torch.where(cov[:, 2, 2] > 0, -cov[:, 0, 2] / cov[:, 2, 2], 0)
                w[:, 3] = torch.where(cov[:, 3, 3] > 0, -cov[:, 1, 3] / cov[:, 3, 3], 0)
                cuw = torch.einsum("ta,tab,tb->t", u, cov, w)
                cuu = torch.einsum("ta,tab,tb->t", u, cov, u)
                cww = torch.einsum("ta,tab,tb->t", w, cov, w)
                spec[self.index[g][0].cpu()] = (cuw / torch.sqrt(cuu * cww)).float()
        return spec

    def reset(self):
        if self.enabled:
            device = {g: t.device for g, t in self.table.items()}
            self.track_sums = {
                g: torch.zeros(len(r), 9, dtype=torch.float64, device=device[g])
                for g, (r, _) in self.groups.items()
            }
            self.mean_sums = {
                g: torch.zeros(5, dtype=torch.float64, device=device[g])
                for g in self.groups
            }
            self.count = {g: 0 for g in self.groups}


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


def position_weights(seq_len, weight_range, weight_exp, device=None):
    """(1, 1, L) flat-topped position weights, 1 at the center, 1/weight_range at the edges."""
    pos_start = -(seq_len / 2 - 0.5)
    pos_end = seq_len / 2 + 0.5
    sigma = -pos_start / (np.log(weight_range)) ** (1 / weight_exp)

    positions = torch.arange(pos_start, pos_end, dtype=torch.float32, device=device)
    weights = torch.exp(-((positions / sigma) ** weight_exp))
    return (weights / weights.max()).view(1, 1, -1)


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
            weights = position_weights(
                seq_len, self.weight_range, self.weight_exp, device
            )
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
        y_true = y_true * self.position_weights
        y_pred = y_pred * self.position_weights

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


def split_profile(num_levels: int, split_exp: float, split_knee: int) -> list:
    """Split weights c_1..c_K = min(1, (2^k / split_knee)^−split_exp).

    c_k weights the splits of 2^k-bin blocks into halves: 1 up to blocks of
    split_knee bins, then falling by 2^−split_exp per doubling of block size.
    """
    return [min(1.0, (2**k / split_knee) ** -split_exp) for k in range(1, num_levels + 1)]


class PoissonTreeLoss(PoissonMultinomialLoss):
    """Position-weighted Poisson over the binary tree of positional splits, weighted per level.

    loss = [t·P(window total) + Σ_k c_k·B_k] / Σw, where B_k is the binomial NLL of
    the level-k splits (2^k-bin blocks into halves) and c_k = split_profile(...).
    Computed as Σ_ℓ a_ℓ·P_ℓ, P_ℓ the Poisson NLL sum-pooled to blocks of 2^ℓ bins:
    a_0 = c_1, a_ℓ = c_{ℓ+1} − c_ℓ, a_K = t − c_K. split_exp 0 gives
    PoissonMultinomialLoss(total_weight) up to a constant. Needs a power-of-two length.
    """

    def __init__(
        self,
        total_weight: float = 1,
        split_exp: float = 0,
        split_knee: int = 1,
        weight_range: float = 1,
        weight_exp: int = 4,
        epsilon: float = 1e-9,
        reduction: str = "mean",
    ):
        super().__init__(total_weight, weight_range, weight_exp, epsilon, reduction=reduction)
        if split_knee < 1 or split_knee & (split_knee - 1):
            raise ValueError("split_knee must be a power of two (bins)")
        self.split_exp = split_exp
        self.split_knee = split_knee

    def forward(self, y_pred, y_true, target_weights=None):
        seq_len = y_true.shape[-1]
        num_levels = seq_len.bit_length() - 1
        if 2**num_levels != seq_len:
            raise ValueError("PoissonTreeLoss needs a power-of-two length")
        self._setup_position_weights(seq_len, y_true.device)
        y_true = y_true * self.position_weights
        y_pred = y_pred * self.position_weights

        c = split_profile(num_levels, self.split_exp, self.split_knee)
        coefs = [c[0]] + [b - a for a, b in zip(c, c[1:])] + [self.total_weight - c[-1]]
        loss = 0
        for level, coef in enumerate(coefs):
            if coef != 0:
                loss = loss + coef * self._poisson(y_true, y_pred).sum(dim=-1)
            if level < num_levels:
                y_true = y_true.unflatten(-1, (-1, 2)).sum(dim=-1)
                y_pred = y_pred.unflatten(-1, (-1, 2)).sum(dim=-1)
        return weighted_reduce(loss / self.weights_sum, target_weights, self.reduction)


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
        target_hist: Per-track histograms of the stored targets; enables spec.
        spec_group_min: Minimum tracks for a group to be scored.
        spec_like: Optional SpecPearsonCorrCoef whose groups and tables to share
            (replaces building them from targets_df/target_hist).
    """

    def __init__(
        self,
        has_coverage: bool,
        has_gene: bool,
        num_targets: int | None,
        num_gene_targets: int | None,
        device: str,
        targets_df=None,
        target_hist=None,
        spec_group_min: int = 20,
        spec_like=None,
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
            if spec_like is not None:
                self.spec = spec_like.fresh()
            elif targets_df is not None and len(targets_df) == num_targets:
                self.spec = SpecPearsonCorrCoef(
                    targets_df, target_hist, spec_group_min
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
