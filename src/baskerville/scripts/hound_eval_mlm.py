#!/usr/bin/env python
# Copyright 2023 Calico Life Sciences LLC
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
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import zarr
from tqdm import tqdm

from baskerville import dataset
from baskerville import mlm
from baskerville import seqnn
from baskerville.hardware import check_mixed_precision
from baskerville.types import BatchData

"""
hound_eval_mlm.py

Evaluate the accuracy, cross-entropy loss and perplexity of a masked language
model on held-out sequences. If a GTF file is provided, also computes
region-specific metrics (repeats, exons, genes, intergenic).

This script consolidates functionality from baskerville-yeast's:
- hound_eval_mlm.py
- hound_eval_mlm_perplexity.py
- hound_eval_mlm_perplexity_region.py
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MLM model.")
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average forward and reverse-complement nucleotide predictions "
        "[Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--has_mask",
        default=False,
        action="store_true",
        help="Dataset has exon masks [Default: %(default)s]",
    )
    parser.add_argument(
        "--has_repeat_mask",
        default=False,
        action="store_true",
        help="Dataset has repeat masks [Default: %(default)s]",
    )
    # BED file support for coordinate mapping (required for region metrics and visualization)
    parser.add_argument(
        "--seq-bed",
        dest="seq_bed",
        default=None,
        help="BED file with sequence coordinates (required for --gtf region metrics)",
    )
    # GTF file for region-specific metrics
    parser.add_argument(
        "--gtf",
        default=None,
        help="GTF file for region-specific metrics (repeats, exons, genes, intergenic) [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #######################################################
    # inputs

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # get masking parameters
    mask_rate = params_train.get("mask_rate", 0.15)
    seq_length = params_model["seq_length"]
    mask_size = max(1, int(mask_rate * seq_length))

    # read data parameters
    data_stats_file = f"{args.data_dir}/statistics.json"
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)
    num_species = data_stats.get("num_species", 1)

    # Get chrom_sizes for region metrics (if available)
    chrom_sizes = data_stats.get("chrom_sizes", {})

    print(f"num_species: {num_species}")
    print(f"mask_rate: {mask_rate}")
    print(f"mask_size: {mask_size}")

    # get loss scaling parameters
    exon_loss_scale = params_train.get("exon_loss_scale", None)
    non_exon_loss_scale = params_train.get("non_exon_loss_scale", 1.0)
    repeat_loss_scale = params_train.get("repeat_loss_scale", None)
    non_repeat_loss_scale = params_train.get("non_repeat_loss_scale", 1.0)

    # read has_mask / has_repeat_mask from params if not set on CLI
    if not args.has_mask and params_train.get("has_mask", False):
        args.has_mask = True
    if not args.has_repeat_mask and params_train.get("has_repeat_mask", False):
        args.has_repeat_mask = True

    # Load BED file for coordinate mapping (optional, needed for region metrics)
    bed_df = None
    if args.seq_bed is not None:
        bed_columns = ["chrom", "start", "end", "name", "species"]
        bed_df = pd.read_csv(args.seq_bed, sep="\t", names=bed_columns)
        print(f"Loaded BED file with {len(bed_df)} sequences")
    elif args.gtf is not None:
        print("Warning: --gtf requires --seq-bed for region-specific metrics")

    # construct eval data
    eval_data = dataset.SeqDatasetMLM(
        args.data_dir,
        split_label=args.split,
        mode="eval",
        has_mask=args.has_mask,
        has_repeat_mask=args.has_repeat_mask,
    )

    num_seqs = len(eval_data)

    # create dataloader
    eval_dataload = torch.utils.data.DataLoader(
        eval_data,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        collate_fn=BatchData.collate,
    )

    # auto-detect old UnetBorzoiBlock checkpoints and patch params to match
    _ckpt = torch.load(args.model_file, map_location="cpu", weights_only=False)
    mlm.patch_params_for_old_unet(params_model, _ckpt)

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    print(f"Restoring model from {args.model_file}")
    seqnn_model.restore(args.model_file)

    # set mixed precision
    mix_dtype = params_train.get("mix_dtype", "float32")
    if mix_dtype != "float32":
        if check_mixed_precision():
            if mix_dtype == "float16":
                seqnn_model.mix_dtype = torch.float16
            elif mix_dtype == "bfloat16":
                seqnn_model.mix_dtype = torch.bfloat16
            else:
                print(f"Warning: Unrecognized mixed precision dtype {mix_dtype}")
        else:
            print("Warning: Mixed precision training not supported on this GPU.")

    device = seqnn_model.device

    # Parse GTF annotations if provided
    annotations = None
    if args.gtf is not None:
        try:
            annotations = parse_gtf_annotations(args.gtf, chrom_sizes)
            print(f"Loaded GTF annotations for {len(annotations)} chromosomes")
        except ImportError:
            print("Warning: pyranges not available, skipping region-specific metrics")
        except Exception as e:
            print(f"Warning: Could not parse GTF: {e}")

    #######################################################
    # evaluate

    # running accumulators (O(1) memory in sequence count)
    eval_loss_sum = 0.0
    n_eval = 0
    eval_loss_per_species = np.zeros(num_species, dtype="float64")
    evals_per_species = np.zeros(num_species, dtype="int64")

    do_region = annotations is not None and bed_df is not None
    if do_region:
        region_total_loss = {"repeat": 0.0, "exon": 0.0, "gene": 0.0, "intergenic": 0.0}
        region_token_sum = {"repeat": 0.0, "exon": 0.0, "gene": 0.0, "intergenic": 0.0}

    # optionally stream full predictions to a zarr group (mirrors seqnn.eval)
    save_root = None
    if args.save:
        zarr_path = f"{args.out_dir}/preds_{args.split}.zarr"
        os.makedirs(zarr_path, exist_ok=True)
        save_root = zarr.open_group(zarr_path, mode="w")
        comp = zarr.codecs.BloscCodec(cname="zstd", clevel=1)
        x_true_z = save_root.create_array(
            "x_true",
            shape=(num_seqs, 4, seq_length),
            chunks=(1, 4, seq_length),
            dtype="float16",
            compressors=comp,
        )
        x_pred_z = save_root.create_array(
            "x_pred",
            shape=(num_seqs, 4, seq_length),
            chunks=(1, 4, seq_length),
            dtype="float16",
            compressors=comp,
        )
        label_z = save_root.create_array(
            "label",
            shape=(num_seqs,),
            chunks=(min(num_seqs, 1024),),
            dtype="int32",
            compressors=comp,
        )
        weight_z = save_root.create_array(
            "weight_scale",
            shape=(num_seqs, seq_length),
            chunks=(1, seq_length),
            dtype="float32",
            compressors=comp,
        )

    seqnn_model.model.eval()

    print(f"Evaluating {num_seqs} sequences...")

    with torch.no_grad():
        for x_ix, batch in enumerate(
            tqdm(eval_dataload, desc="Evaluating", total=num_seqs)
        ):
            if x_ix % 64 == 0:
                print(f"Evaluating sequence pattern = {x_ix}", flush=True)

            x = batch.sequence.to(device)
            label = batch.species_label.to(device)
            exon_mask = batch.exon_mask
            repeat_mask = batch.repeat_mask

            if exon_mask is not None:
                exon_mask = exon_mask.to(device)
            if repeat_mask is not None:
                repeat_mask = repeat_mask.to(device)

            # optionally set position-specific loss weight scales from binary mask
            sw = None
            if exon_mask is not None and exon_loss_scale is not None:
                sw = exon_mask * exon_loss_scale + (1 - exon_mask) * non_exon_loss_scale
            if repeat_mask is not None and repeat_loss_scale is not None:
                repeat_sw = (
                    repeat_mask * repeat_loss_scale
                    + (1 - repeat_mask) * non_repeat_loss_scale
                )
                if sw is None:
                    sw = repeat_sw
                else:
                    sw = sw * repeat_sw

            # extract species index for trunk normalization
            species_id = int(label[0, 0].argmax())

            # predict all positions via iterative masking
            x_pred = mlm.predict_masked_sequence(
                seqnn_model.model,
                x,
                seq_length,
                mask_size,
                device,
                seqnn_model.mix_dtype,
                di=species_id,
                rc=args.rc,
            )

            # fold reductions in immediately (no per-sequence buffering)
            x_true_np = x.cpu().numpy()  # (1, 4, seq_length)
            x_pred_np = x_pred.cpu().numpy()  # (1, 4, seq_length)
            sw_np = (
                sw.cpu().numpy()[0]
                if sw is not None
                else np.ones(seq_length, dtype="float32")
            )  # (seq_length,)

            # per-position cross-entropy: transpose to (seq_length, 4)
            eps = 1e-8
            ce_per_pos = -np.sum(
                x_true_np[0].T * np.log(x_pred_np[0].T + eps), axis=-1
            )  # (seq_length,)

            # weighted per-example loss -> overall and per-species accumulators
            example_loss = float(np.mean(ce_per_pos * sw_np))
            eval_loss_sum += example_loss
            n_eval += 1
            eval_loss_per_species[species_id] += example_loss
            evals_per_species[species_id] += 1

            # region-specific accumulation (BED row order == sequence order)
            if do_region and x_ix < len(bed_df):
                bed_row = bed_df.iloc[x_ix]
                chrom = str(bed_row["chrom"])
                if chrom.startswith("chr"):
                    chrom = chrom[3:]
                if chrom in annotations:
                    masks = build_region_masks(
                        int(bed_row["start"]), seq_length, annotations[chrom]
                    )
                    for region, region_mask in zip(
                        ["repeat", "exon", "gene", "intergenic"], masks
                    ):
                        if np.any(region_mask):
                            region_total_loss[region] += np.sum(
                                ce_per_pos[region_mask] * sw_np[region_mask]
                            )
                            region_token_sum[region] += np.sum(sw_np[region_mask])

            # optionally stream full arrays to disk
            if save_root is not None:
                x_true_z[x_ix] = x_true_np[0].astype("float16")
                x_pred_z[x_ix] = x_pred_np[0].astype("float16")
                label_z[x_ix] = species_id
                weight_z[x_ix] = sw_np

    # finalize overall and per-species loss from running accumulators
    eval_loss = eval_loss_sum / max(n_eval, 1)

    mask = evals_per_species > 0
    eval_loss_per_species[mask] /= evals_per_species[mask].astype("float64")
    eval_loss_per_species[~mask] = 0.0

    # compute perplexity
    eval_perplexity = float(np.exp(eval_loss))
    eval_perplexity_per_species = np.exp(eval_loss_per_species)
    eval_perplexity_per_species[~mask] = 0.0

    # write species-level statistics
    acc_df = pd.DataFrame(
        {
            "species": np.arange(num_species, dtype="int32"),
            "loss": eval_loss_per_species,
            "perplexity": eval_perplexity_per_species,
            "n": evals_per_species,
        }
    )

    acc_df.to_csv(
        f"{args.out_dir}/acc_{args.split}.txt",
        sep="\t",
        index=False,
        float_format="%.5f",
    )

    print(f"Average Categorical Cross-Entropy loss = {eval_loss:.5f}")
    print(f"Overall Perplexity = {eval_perplexity:.5f}")

    # Region-specific metrics (accumulated during the eval loop above)
    if do_region:
        # Compute region averages and perplexities
        region_avg_loss = {}
        region_perplexity = {}
        for region in region_total_loss:
            if region_token_sum[region] > 0:
                avg_loss = region_total_loss[region] / region_token_sum[region]
                region_avg_loss[region] = avg_loss
                region_perplexity[region] = math.exp(avg_loss)
            else:
                region_avg_loss[region] = None
                region_perplexity[region] = None

        region_stats_df = pd.DataFrame(
            {
                "region": list(region_avg_loss.keys()),
                "avg_loss": list(region_avg_loss.values()),
                "perplexity": list(region_perplexity.values()),
                "weighted_tokens": list(region_token_sum.values()),
            }
        )
        region_stats_df.to_csv(
            f"{args.out_dir}/region_stats_{args.split}.txt",
            sep="\t",
            index=False,
            float_format="%.5f",
        )
        print("\nRegion-specific metrics:")
        print(region_stats_df)


def compute_complement(merged_df, chrom_sizes):
    """
    Compute complement intervals (e.g. intergenic) from merged gene intervals.
    Return a DataFrame with columns [Chromosome, Start, End].
    """
    complement_list = []
    for chrom, size in chrom_sizes.items():
        chrom_df = merged_df[merged_df.Chromosome == chrom].sort_values("Start")
        current_start = 0
        if chrom_df.shape[0] == 0:
            complement_list.append({"Chromosome": chrom, "Start": 0, "End": size})
        else:
            for _, row in chrom_df.iterrows():
                if row["Start"] > current_start:
                    complement_list.append(
                        {
                            "Chromosome": chrom,
                            "Start": current_start,
                            "End": row["Start"],
                        }
                    )
                current_start = max(current_start, row["End"])
            if current_start < size:
                complement_list.append(
                    {"Chromosome": chrom, "Start": current_start, "End": size}
                )
    return pd.DataFrame(complement_list)


def parse_gtf_annotations(gtf_file, chrom_sizes):
    """
    Parse a GTF file using pyranges and return a dictionary of annotations:
    { chrom: {"gene": [...], "exon": [...], "repeat": [...], "intergenic": [...]}, ... }
    """
    import pyranges as pr

    gtf_pr = pr.read_gtf(gtf_file)
    df = gtf_pr.as_df()

    # Extract gene and exon features
    gene_df = df[df.Feature == "gene"].copy()
    exon_df = df[df.Feature.isin(["exon", "CDS"])].copy()

    # Extract repeat features (if present)
    repeat_df = df[df.Feature.str.contains("repeat", case=False, na=False)].copy()

    # Build a PyRanges for the gene intervals to get intergenic
    gene_pr = pr.PyRanges(gene_df)
    merged_gene_df = gene_pr.merge().as_df()

    # Compute intergenic regions as complement of genes
    intergenic_df = compute_complement(merged_gene_df, chrom_sizes)

    # Convert each chromosome's annotation to a dictionary of intervals
    annotations = {}
    all_chroms = set(df.Chromosome.unique())
    if chrom_sizes:
        all_chroms.update(chrom_sizes.keys())

    for chrom in all_chroms:
        gene_intervals = (
            gene_df[gene_df.Chromosome == chrom][["Start", "End"]]
            .to_records(index=False)
            .tolist()
        )
        exon_intervals = (
            exon_df[exon_df.Chromosome == chrom][["Start", "End"]]
            .to_records(index=False)
            .tolist()
        )
        repeat_intervals = (
            repeat_df[repeat_df.Chromosome == chrom][["Start", "End"]]
            .to_records(index=False)
            .tolist()
            if len(repeat_df) > 0
            else []
        )
        intergenic_intervals = (
            intergenic_df[intergenic_df.Chromosome == chrom][["Start", "End"]]
            .to_records(index=False)
            .tolist()
            if len(intergenic_df) > 0
            else []
        )

        annotations[chrom] = {
            "gene": gene_intervals,
            "exon": exon_intervals,
            "repeat": repeat_intervals,
            "intergenic": intergenic_intervals,
        }

    return annotations


def build_region_masks(genomic_start, seq_length, ann):
    """
    Given the genomic start coordinate and sequence length, and annotation intervals
    (dict with keys: "repeat", "exon", "gene", "intergenic"),
    return boolean masks for each region (length seq_length).
    """
    mask_repeat = np.zeros(seq_length, dtype=bool)
    mask_exon = np.zeros(seq_length, dtype=bool)
    mask_gene = np.zeros(seq_length, dtype=bool)
    mask_intergenic = np.zeros(seq_length, dtype=bool)

    for region, mask in zip(
        ["repeat", "exon", "gene", "intergenic"],
        [mask_repeat, mask_exon, mask_gene, mask_intergenic],
    ):
        intervals = ann.get(region, [])
        for start, end in intervals:
            # Overlap between [genomic_start, genomic_start+seq_length) and [start, end)
            overlap_start = max(genomic_start, start)
            overlap_end = min(genomic_start + seq_length, end)
            if overlap_end > overlap_start:
                pos_start = overlap_start - genomic_start
                pos_end = overlap_end - genomic_start
                mask[pos_start:pos_end] = True

    return mask_repeat, mask_exon, mask_gene, mask_intergenic


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
