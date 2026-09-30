#!/usr/bin/env python
"""
hound_viz_mlm.py

Visualize MLM predictions for a genomic region.
Filters sequences to a single species (--species: BED name or integer index).

Usage:
  python hound_viz_mlm.py params.json model.pth data_dir \
      sequences_test.bed --region chrXI:100000-105000

  # Zoom into a sub-region:
  python hound_viz_mlm.py ... --zoom 1000-2000
"""

import argparse
import json
import os

import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baskerville import dataset
from baskerville import mlm
from baskerville import seqnn
from baskerville.hardware import check_mixed_precision

NUC = ["A", "C", "G", "T"]
NUC_COLORS = {"A": "#228B22", "C": "#0000CD", "G": "#FFB300", "T": "#DC143C"}


def plot_predictions(
    x_true, x_pred, start, end, title, out_path, genomic_start=None, sw=None
):
    """Three-panel plot: true seq, predicted probability heatmap, weighted CE loss."""
    L = end - start
    xt = x_true[:, start:end]
    xp = x_pred[:, start:end]
    sw_sub = sw[start:end] if sw is not None else np.ones(L)
    x_offset = genomic_start if genomic_start is not None else start

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(14, L * 0.02), 7),
        height_ratios=[0.5, 2, 1.5],
        gridspec_kw={"hspace": 0.15},
    )

    # top: true sequence as colored bar
    ax = axes[0]
    true_idx = np.argmax(xt, axis=0)
    colors = [NUC_COLORS[NUC[i]] for i in true_idx]
    ax.bar(np.arange(L) + x_offset, 1, width=1.0, color=colors, linewidth=0)
    ax.set_xlim(x_offset, x_offset + L)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("True", fontsize=9)

    # middle: predicted probability heatmap
    ax = axes[1]
    im = ax.imshow(
        xp,
        aspect="auto",
        interpolation="none",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=[x_offset, x_offset + L, 3.5, -0.5],
    )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(NUC)
    ax.set_ylabel("Pred prob", fontsize=9)
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01)

    # bottom: per-position weighted CE loss
    ax = axes[2]
    eps = 1e-8
    ce = -np.sum(xt * np.log(xp + eps), axis=0) * sw_sub
    ax.bar(
        np.arange(L) + x_offset,
        ce,
        width=1.0,
        color="steelblue",
        alpha=0.7,
        linewidth=0,
    )
    ax.axhline(
        y=np.log(4),
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=0.8,
        label="random",
    )
    ax.set_xlim(x_offset, x_offset + L)
    ax.set_ylabel("Weighted CE", fontsize=9)
    ax.set_xlabel("Genomic position", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_logo(x_pred, start, end, title, out_path, genomic_start=None):
    """Sequence logo from predicted probabilities (information content)."""
    L = end - start
    xp = x_pred[:, start:end]
    x_offset = genomic_start if genomic_start is not None else start

    # compute information content per position
    eps = 1e-8
    entropy = -np.sum(xp * np.log2(xp + eps), axis=0)
    ic = np.clip(2.0 - entropy, 0, 2)

    # build IC-weighted matrix (positions x nucleotides)
    ic_matrix = (xp * ic[np.newaxis, :]).T
    df = pd.DataFrame(ic_matrix, columns=NUC, index=np.arange(x_offset, x_offset + L))

    fig, ax = plt.subplots(figsize=(max(14, L * 0.04), 3))
    logomaker.Logo(
        df,
        ax=ax,
        color_scheme={
            "A": NUC_COLORS["A"],
            "C": NUC_COLORS["C"],
            "G": NUC_COLORS["G"],
            "T": NUC_COLORS["T"],
        },
    )
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("Bits", fontsize=9)
    ax.set_xlabel("Genomic position", fontsize=9)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MLM predictions for a genomic region."
    )
    parser.add_argument("-o", "--out_dir", default="viz_out")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--species",
        default="GCA_000146045_2",
        help="Species to visualize: BED 'species' name or integer "
        "index (default: GCA_000146045_2)",
    )
    parser.add_argument(
        "--region", required=True, help="Genomic region e.g. chrXI:100000-105000"
    )
    parser.add_argument(
        "--zoom", default=None, help="Zoom range within sequence, e.g. 1000-2000"
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        default=False,
        help="Average forward and RC predictions",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of random masking runs to average (default: 1)",
    )
    parser.add_argument(
        "--save-npz", dest="save_npz", action="store_true", default=False
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file")
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("seq_bed", help="BED file with sequence coordinates")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read params
    with open(args.params_file) as f:
        params = json.load(f)
    params_model = params["model"]
    params_train = params["train"]
    mask_rate = params_train.get("mask_rate", 0.15)
    seq_length = params_model["seq_length"]
    mask_size = max(1, int(mask_rate * seq_length))

    with open(f"{args.data_dir}/statistics.json") as f:
        data_stats = json.load(f)
    num_species = data_stats.get("num_species", 1)

    # loss scale parameters (must match training config)
    exon_loss_scale = params_train.get("exon_loss_scale", None)
    non_exon_loss_scale = params_train.get("non_exon_loss_scale", 1.0)
    repeat_loss_scale = params_train.get("repeat_loss_scale", None)
    non_repeat_loss_scale = params_train.get("non_repeat_loss_scale", 1.0)
    has_mask = exon_loss_scale is not None
    has_repeat_mask = repeat_loss_scale is not None

    # load dataset up front (also used to resolve --species given as an index)
    eval_data = dataset.SeqDatasetMLM(
        args.data_dir,
        split_label=args.split,
        mode="eval",
        has_mask=has_mask,
        has_repeat_mask=has_repeat_mask,
    )

    # --species may be a BED 'species' name or an integer species index
    try:
        species_index = int(args.species)
        species_name = None
    except ValueError:
        species_index = None
        species_name = args.species

    bed_df = pd.read_csv(
        args.seq_bed, sep="\t", names=["chrom", "start", "end", "name", "species"]
    )

    # resolve sequence index from region, restricted to the requested species
    chrom, coords = args.region.split(":")
    rstart, rend = [int(x) for x in coords.split("-")]
    region_hits = bed_df[
        (bed_df["chrom"] == chrom)
        & (bed_df["start"] <= rstart)
        & (bed_df["end"] >= rend)
    ]
    if species_name is not None:
        matches = region_hits[region_hits["species"] == species_name]
    else:
        # no name<->index map is persisted, so resolve the index by checking
        # each region candidate's one-hot species label
        keep = [
            i
            for i in region_hits.index
            if i < len(eval_data)
            and int(eval_data[i].species_label.argmax()) == species_index
        ]
        matches = region_hits.loc[keep]
    if len(matches) == 0:
        sel = species_name if species_name is not None else f"index {species_index}"
        print(f"No '{sel}' sequence contains {args.region}. Available on {chrom}:")
        for _, r in bed_df[bed_df["chrom"] == chrom].iterrows():
            print(f"  {r['chrom']}:{r['start']}-{r['end']} ({r['species']})")
        return
    seq_idx = matches.index[0]
    if seq_idx >= len(eval_data):
        print(f"Index {seq_idx} out of range (max {len(eval_data) - 1})")
        return
    bed_row = matches.iloc[0]
    zoom_start = rstart - int(bed_row["start"])
    zoom_end = rend - int(bed_row["start"])
    print(
        f"Region {args.region} -> seq {seq_idx} "
        f"({bed_row['chrom']}:{bed_row['start']}-{bed_row['end']}), "
        f"local range [{zoom_start}, {zoom_end})"
    )

    if args.zoom is not None:
        zs, ze = [int(x) for x in args.zoom.split("-")]
        zoom_start, zoom_end = zs, ze

    # auto-detect old UnetBorzoiBlock checkpoints and patch params to match
    _ckpt = torch.load(args.model_file, map_location="cpu", weights_only=False)
    mlm.patch_params_for_old_unet(params_model, _ckpt)

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file)
    mix_dtype = params_train.get("mix_dtype", "float32")
    if mix_dtype == "float16" and check_mixed_precision():
        seqnn_model.mix_dtype = torch.float16
    elif mix_dtype == "bfloat16" and check_mixed_precision():
        seqnn_model.mix_dtype = torch.bfloat16

    device = seqnn_model.device
    model = seqnn_model.model
    model.eval()

    # get sequence and masks
    example = eval_data[seq_idx]
    x = example.sequence.unsqueeze(0).to(device)
    label = example.species_label.unsqueeze(0).to(device)

    species_id = torch.argmax(label.squeeze()).item()
    print(f"Species ID: {species_id}")

    # build sample weights matching eval/training
    sw = np.ones(seq_length, dtype=np.float32)
    if has_mask and example.exon_mask is not None:
        exon_mask = example.exon_mask.numpy()
        sw *= exon_mask * exon_loss_scale + (1 - exon_mask) * non_exon_loss_scale
    if has_repeat_mask and example.repeat_mask is not None:
        repeat_mask = example.repeat_mask.numpy()
        sw *= (
            repeat_mask * repeat_loss_scale + (1 - repeat_mask) * non_repeat_loss_scale
        )

    species_id = int(label[0, 0].argmax())

    # predict (average over num_runs random masking orders)
    num_runs = args.num_runs
    rounds_per_run = int(np.ceil(seq_length / mask_size))
    print(
        f"Predicting sequence {seq_idx} ({mask_size} positions/round, "
        f"~{rounds_per_run} rounds/run, {num_runs} run(s))..."
    )
    with torch.no_grad():
        x_pred_accum = torch.zeros(
            (1, 4, seq_length), device=device, dtype=torch.float32
        )
        for ri in range(num_runs):
            x_pred_accum += mlm.predict_masked_sequence(
                model,
                x,
                seq_length,
                mask_size,
                device,
                seqnn_model.mix_dtype,
                di=species_id,
                rc=args.rc,
            )
            if num_runs > 1:
                print(f"  Run {ri + 1}/{num_runs} done")
        x_pred = x_pred_accum / num_runs

    x_true_np = x[0].cpu().numpy()
    x_pred_np = x_pred[0].cpu().numpy()

    # summary stats (weighted, matching eval/training)
    eps = 1e-8
    ce_raw = -np.sum(x_true_np * np.log(x_pred_np + eps), axis=0)
    ce_weighted = np.mean(ce_raw * sw)
    print(f"Weighted mean CE: {ce_weighted:.5f}, Perplexity: {np.exp(ce_weighted):.5f}")

    # save npz
    if args.save_npz:
        npz_path = os.path.join(args.out_dir, f"seq{seq_idx}_pred.npz")
        save_dict = {
            "x_true": x_true_np,
            "x_pred": x_pred_np,
            "seq_idx": seq_idx,
            "chrom": str(bed_row["chrom"]),
            "start": int(bed_row["start"]),
            "end": int(bed_row["end"]),
        }
        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved: {npz_path}")

    # genomic coords for zoom region
    seq_chrom = str(bed_row["chrom"])
    seq_start = int(bed_row["start"])
    zoom_gstart = seq_start + zoom_start
    zoom_gend = seq_start + zoom_end
    zoom_title = f"{seq_chrom}:{zoom_gstart}-{zoom_gend}"

    # plot weighted CE overview for full sequence
    ce_w = ce_raw * sw
    overview_path = os.path.join(args.out_dir, f"seq{seq_idx}_ce_overview.png")
    fig, ax = plt.subplots(figsize=(14, 3))
    win = min(100, seq_length // 10)
    ce_smooth = np.convolve(ce_w, np.ones(win) / win, mode="same") if win > 1 else ce_w
    gx = np.arange(seq_length) + seq_start
    ax.plot(gx, ce_smooth, color="steelblue", linewidth=0.5)
    ax.axhline(y=np.log(4), color="red", linestyle="--", alpha=0.5, label="random")
    ax.axvspan(zoom_gstart, zoom_gend, alpha=0.15, color="orange", label="region")
    ax.set_xlabel("Genomic position")
    ax.set_ylabel("Weighted CE (smoothed)")
    ax.set_title(f"CE landscape | {zoom_title}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {overview_path}")
    plt.close()

    # plot zoomed heatmap + logo
    heatmap_path = os.path.join(args.out_dir, f"seq{seq_idx}_heatmap.png")
    plot_predictions(
        x_true_np,
        x_pred_np,
        zoom_start,
        zoom_end,
        zoom_title,
        heatmap_path,
        genomic_start=zoom_gstart,
        sw=sw,
    )

    logo_path = os.path.join(args.out_dir, f"seq{seq_idx}_logo.png")
    plot_logo(
        x_pred_np,
        zoom_start,
        zoom_end,
        f"Predicted logo | {zoom_title}",
        logo_path,
        genomic_start=zoom_gstart,
    )


if __name__ == "__main__":
    main()
