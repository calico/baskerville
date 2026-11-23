#!/usr/bin/env python
# Copyright 2021 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
from qnorm import quantile_normalize
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

from baskerville import dataset
from baskerville.gene import Transcriptome, interval_output_slice


def _gene_intervals(gtf, bed_path, span):
    """Write BED6 intervals excluding same-strand overlaps with other genes."""
    tx = Transcriptome(gtf)
    regions = tx.bedtool_span() if span else tx.bedtool_exon()
    excluded = {
        (hit[3], hit.start, hit.end)
        for hit in regions.intersect(regions, s=True, wo=True)
        if hit[3] != hit[9]
    }
    with bed_path.open("w") as bed:
        for region in regions:
            if (region.name, region.start, region.end) not in excluded:
                bed.write(str(region))
    if not bed_path.stat().st_size:
        raise ValueError(f"No eligible gene intervals in {gtf}")
    return pr.read_bed(str(bed_path))


class GeneCoverage:
    """Reduce coverage on GTF intervals as sequences arrive in dataset order."""

    def __init__(self, data, model, targets, args):
        self.out_dir = Path(args.out_dir)
        self.pseudo_qtl = args.pseudo_qtl
        self.untransform = model.untransform_fn
        gene_tracks = dataset.gene_targets(targets)
        if gene_tracks.empty:
            raise ValueError("No coverage tracks selected for gene evaluation")
        rep, pair = dataset.strand_collapse(gene_tracks)
        positions = targets.index.get_indexer(gene_tracks.index)
        self.tracks = {"+": positions[rep], "-": positions[pair]}
        self.transforms = {"+": gene_tracks.iloc[rep], "-": gene_tracks.iloc[pair]}
        self.metadata = gene_tracks.iloc[rep].copy()
        self.metadata["group"] = dataset.target_groups(self.metadata)

        seqs = pd.read_csv(
            Path(data.data_dir) / "sequences.bed",
            sep="\t",
            names=["Chromosome", "Start", "End", "split"],
        )
        # Rows in dataset order: shard by shard, each matching its BED split label.
        shards = [Path(path).stem for path in data.zarr_files]
        seqs = pd.concat([seqs[seqs.split == shard] for shard in shards])
        seqs = seqs.reset_index(drop=True)
        if len(seqs) != len(data):
            raise ValueError(
                "Split BED row count differs from the number of dataset examples"
            )
        self.length = data.target_length
        width = data.pool_width
        crop = (data.extra_crop_bp or 0) + model.output_crop_bp() - data.data_crop_bp
        seqs.Start += crop
        seqs.End -= crop
        if (
            crop != (data.extra_crop_bins or 0) * width
            or not (seqs.End - seqs.Start == self.length * width).all()
            or model.model.output_stride != width
        ):
            raise ValueError(
                "Model, cropped target bins and split BED coordinates must align"
            )
        seqs["Row"] = seqs.index
        genes = _gene_intervals(args.genes_gtf, self.out_dir / "genes.bed", args.span)
        intervals = genes.df
        lengths = (
            (intervals.End - intervals.Start)
            .groupby(intervals.Name, observed=True)
            .sum()
        )
        self.overlaps = defaultdict(list)
        sizes = defaultdict(int)
        selected = pr.PyRanges(seqs.iloc[:: args.seq_step])
        for hit in selected.join(genes, apply_strand_suffix=False).df.itertuples():
            start, stop = interval_output_slice(
                hit.Start_b,
                hit.End_b,
                hit.Start,
                hit.End - hit.Start,
                width,
                majority_overlap=False,
            )
            self.overlaps[hit.Row].append((hit.Name, hit.Strand, start, stop))
            sizes[hit.Name] += stop - start
        if not sizes:
            raise ValueError("No GTF intervals overlap the selected evaluation windows")
        self.ids = sorted(sizes)
        self.index = {gene: i for i, gene in enumerate(self.ids)}
        self.sizes = np.array([sizes[gene] for gene in self.ids])
        self.length_bins = (lengths[self.ids] / width).to_numpy()

        # Centered moments of (preds, targets, log2 preds, log2 targets) per gene/track.
        shape = (len(self.ids), len(self.metadata))
        self.counts = np.zeros(len(self.ids), dtype=int)
        self.means = np.zeros((4, *shape))
        self.squares = np.zeros((4, *shape))
        self.cross = np.zeros(shape)

    def update(self, row, preds, targets):
        """Consume full-resolution (batch, tracks, bins) tensors before subsampling."""
        if preds.shape != targets.shape or preds.shape[-1] != self.length:
            raise ValueError(
                "Coverage predictions and targets must match the expected bin grid"
            )
        for i in range(len(preds)):
            intervals = self.overlaps.get(row + i, ())
            for strand in ("+", "-"):
                parts = [p for p in intervals if p[1] == strand]
                if not parts:
                    continue
                # Transfer and untransform each participating track/bin once per strand.
                used = np.zeros(self.length, dtype=bool)
                for _, _, start, stop in parts:
                    used[start:stop] = True
                bins = np.flatnonzero(used)
                tracks = self.tracks[strand][:, None]

                def select(values):
                    selected = values[i, tracks, bins].float().cpu().numpy()
                    return self.untransform(selected, self.transforms[strand])

                p, y = select(preds), select(targets)
                offsets = np.r_[0, used.cumsum()]
                for gene, _, start, stop in parts:
                    sl = slice(offsets[start], offsets[stop])
                    self._add(gene, p[:, sl], y[:, sl])

    def _add(self, gene, preds, targets):
        g = self.index[gene]
        n = preds.shape[1]
        previous = self.counts[g]
        count = previous + n
        # Merge block moments into the running ones (Chan et al.); exact when previous=0.
        values = np.stack((preds, targets, np.log2(preds + 1), np.log2(targets + 1)))
        means = values.mean(axis=-1, dtype=np.float64)
        centered = values - means[..., None]
        delta = means - self.means[:, g]
        shrink = previous * n / count
        self.squares[:, g] += np.einsum("itb,itb->it", centered, centered)
        self.squares[:, g] += delta**2 * shrink
        self.cross[g] += np.einsum("tb,tb->t", centered[2], centered[3])
        self.cross[g] += delta[2] * delta[3] * shrink
        self.means[:, g] += delta * (n / count)
        self.counts[g] = count

    def summaries(self):
        """Gene expression, within-gene log Pearson R and target variance."""
        n = self.counts[:, None]
        length = self.length_bins[:, None]
        preds, targets = self.means[0] * length, self.means[1] * length
        variance = self.squares[1] / n
        denom = np.sqrt(self.squares[2] * self.squares[3])
        valid = (self.squares[0] / n > 1e-6) & (variance > 1e-6) & (denom > 0)
        within = np.full_like(denom, np.nan)
        np.divide(self.cross, denom, out=within, where=valid)
        return preds, targets, within, variance

    def finish(self):
        if not np.array_equal(self.counts, self.sizes):
            raise ValueError("Missing coverage for planned GTF intervals")
        _write_reports(
            self.out_dir, self.pseudo_qtl, self.metadata, self.ids, *self.summaries()
        )


def _accuracy(targets, preds):
    """Pearson correlation and explained variance per column."""
    if len(targets) < 2:
        r = np.full(targets.shape[1], np.nan)
    else:
        r = pearsonr(targets, preds, axis=0).statistic
    return r, explained_variance_score(targets, preds, multioutput="raw_values")


def _write_reports(
    out_dir, pseudo_qtl, metadata, gene_ids, preds, targets, within, variance
):
    if pseudo_qtl is not None:
        for ti in range(len(metadata)):
            nonzero = targets[:, ti] != 0
            if nonzero.any():
                for matrix in (preds, targets):
                    matrix[:, ti] += np.quantile(matrix[nonzero, ti], pseudo_qtl)
    for matrix in (preds, targets):
        np.log2(matrix + 1, out=matrix)

    tasks = metadata[["identifier", "group"]].copy()
    genes = pd.DataFrame({"gene_id": gene_ids})
    norm_preds = np.empty_like(preds)
    norm_targets = np.empty_like(targets)
    for group in sorted(metadata.group.unique()):
        mask = (metadata.group == group).to_numpy()
        y, p = targets[:, mask], preds[:, mask]
        tasks.loc[mask, "pearsonr"], tasks.loc[mask, "r2"] = _accuracy(y, p)
        y, p = quantile_normalize(y), quantile_normalize(p)
        norm_targets[:, mask], norm_preds[:, mask] = y, p

        sd = y.std(axis=1)
        valid = (sd >= 1e-5) & (p.std(axis=1) >= 1e-5)
        r, r2 = np.zeros(len(y)), np.zeros(len(y))
        if valid.any():
            r[valid], r2[valid] = _accuracy(y[valid].T, p[valid].T)
        genes[f"{group}_r"], genes[f"{group}_r2"], genes[f"{group}_sd"] = r, r2, sd
        print(f"{group} R: {r.mean():.4f}  R2: {r2.mean():.4f}")
        high_sd = sd > np.median(sd)
        if high_sd.any():
            print(
                f"{group} R (sd50): {r[high_sd].mean():.4f}  R2 (sd50): {r2[high_sd].mean():.4f}"
            )

        y -= y.mean(axis=1, keepdims=True)
        p -= p.mean(axis=1, keepdims=True)
        tasks.loc[mask, "pearsonr_norm"], tasks.loc[mask, "r2_norm"] = _accuracy(y, p)

    cutoff = np.percentile(variance, 80, axis=0)
    tasks["pearsonr_gene"] = [
        within[keep, ti].mean() if keep.any() else np.nan
        for ti, keep in enumerate((variance > cutoff).T)
    ]
    tasks["description"] = metadata.description
    for name, values in (
        ("covgene_targets", targets),
        ("covgene_preds", preds),
        ("covgene_targets_norm", norm_targets),
        ("covgene_preds_norm", norm_preds),
        ("covgene_within", within),
        ("covgene_var", variance),
    ):
        pd.DataFrame(values, index=gene_ids, columns=metadata.identifier).to_csv(
            out_dir / f"{name}.tsv.gz", sep="\t", float_format="%.3f"
        )
    tasks.to_csv(out_dir / "acc_covgene.txt", sep="\t")
    print(f"{len(genes)} genes")
    for label, column in (
        ("Overall PearsonR", "pearsonr"),
        ("Overall R2", "r2"),
        ("Normalized PearsonR", "pearsonr_norm"),
        ("Normalized R2", "r2_norm"),
        ("Within-gene PearsonR", "pearsonr_gene"),
    ):
        print(f"{label}: {tasks[column].mean():.4f}")
    # Fold launchers use the final report to recognize completed evaluations.
    pending = out_dir / "covgene_metrics.tsv.tmp"
    genes.to_csv(pending, sep="\t", float_format="%.4f")
    pending.replace(out_dir / "covgene_metrics.tsv")
