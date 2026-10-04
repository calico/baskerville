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
import os

import h5py
import numpy as np
import pandas as pd
import torch

from baskerville import bed
from baskerville import dataset
from baskerville import dna
from baskerville import ism
from baskerville import seqnn
from baskerville import snps
from baskerville.gene import Transcriptome, find_overlapping_genes
from baskerville.snps import build_gene_masks

"""
hound_ism_bed

Perform an in silico saturation mutagenesis of sequences in a BED file.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Perform in silico saturation mutagenesis of sequences in a BED file."
    )
    parser.add_argument("params_file", help="Model parameters JSON file")
    parser.add_argument("model_file", help="Model file")
    parser.add_argument("bed_file", help="BED file with sequences to analyze")

    parser.add_argument(
        "-d",
        "--mut_down",
        dest="mut_down",
        default=0,
        type=int,
        help="Nucleotides downstream of center sequence to mutate",
    )
    parser.add_argument(
        "-f",
        "--genome_fasta",
        dest="genome_fasta",
        required=True,
        help="Genome FASTA for sequences",
    )
    parser.add_argument(
        "-g",
        dest="genes_gtf",
        default=None,
        help="GTF for gene annotations. Enables covgene/ and gene/ scoring.",
    )
    parser.add_argument(
        "--head",
        dest="head",
        default=0,
        type=int,
        help="Model head with which to predict.",
    )
    parser.add_argument(
        "-l",
        "--mut_len",
        dest="mut_len",
        default=0,
        type=int,
        help="Length of center sequence to mutate",
    )
    parser.add_argument(
        "-m",
        "--mix_dtype",
        dest="mix_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Mixed precision dtype",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        dest="out_dir",
        default="ism_bed_out",
        help="Output directory",
    )
    parser.add_argument(
        "-p",
        "--processes",
        dest="processes",
        default=None,
        type=int,
        help="Number of processes, passed by multi script",
    )
    parser.add_argument(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions",
    )
    parser.add_argument(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts",
    )
    parser.add_argument(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save.",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        dest="targets_file",
        default=None,
        type=str,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--targets_gene_file",
        dest="targets_gene_file",
        default=None,
        type=str,
        help="File specifying gene head target indexes and labels",
    )
    parser.add_argument(
        "-u",
        "--mut_up",
        dest="mut_up",
        default=0,
        type=int,
        help="Nucleotides upstream of center sequence to mutate",
    )

    args = parser.parse_args()

    # parse options
    args.mix_dtype = snps.parse_mix_dtype(args.mix_dtype)

    args.shifts = [int(shift) for shift in args.shifts.split(",")]
    args.snp_stats = [snp_stat for snp_stat in args.snp_stats.split(",")]

    if args.mut_up > 0 or args.mut_down > 0:
        args.mut_len = args.mut_up + args.mut_down
    else:
        assert args.mut_len > 0
        args.mut_up = args.mut_len // 2
        args.mut_down = args.mut_len - args.mut_up

    os.makedirs(args.out_dir, exist_ok=True)

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # read targets
    if args.targets_file is None:
        parser.error("Must provide targets file to clarify stranded datasets")
    targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)

    # gene-level scoring
    score_genes = args.genes_gtf is not None

    # partition stats by prefix
    cov_stats, covgene_stats, gene_stats = snps.partition_snp_stats(args.snp_stats)

    # covgene/ only: predict gene tracks only
    if score_genes and covgene_stats and not cov_stats:
        targets_df = snps.gene_targets(targets_df)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        params_model["strand_pair"] = dataset.strand_pair_indices(targets_df)

        # construct strand sum transform
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None

    num_targets = targets_strand_df.shape[0]

    #################################################################
    # setup model

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model, output_slice=targets_df.index)
    seqnn_model.restore(args.model_file)
    seqnn_model.ensemble_rc = args.rc
    seqnn_model.ensemble_shifts = args.shifts
    seqnn_model.mix_dtype = args.mix_dtype
    seqnn_model.model.eval()

    output_length = seqnn_model.output_length()
    output_stride = seqnn_model.output_stride()
    output_crop = seqnn_model.output_crop_bp()

    # gene scoring setup
    gene_depth = (
        seqnn_model.output_depth(args.head, head_type="gene") if score_genes else 0
    )
    has_gene_head = gene_depth > 0

    # validate gene stats
    if gene_stats and not has_gene_head:
        print(
            f"WARNING: gene/ stats {gene_stats} requested but model has no gene head. Ignoring."
        )
        gene_stats = []

    # gene-track masks for covgene/ slicing
    if score_genes and covgene_stats:
        gene_mask = torch.tensor(
            snps.gene_track_mask(targets_df),
            dtype=torch.bool,
            device=seqnn_model.device,
        )
        if "strand_pair" in targets_df.columns:
            plus_mask = (
                torch.tensor(
                    (targets_df.strand != "-").values,
                    dtype=torch.bool,
                    device=seqnn_model.device,
                )
                & gene_mask
            )
            minus_mask = (
                torch.tensor(
                    (targets_df.strand != "+").values,
                    dtype=torch.bool,
                    device=seqnn_model.device,
                )
                & gene_mask
            )
        else:
            plus_mask = minus_mask = gene_mask
        gene_mask_strand = snps.gene_track_mask(targets_strand_df)
    else:
        plus_mask = minus_mask = None
        gene_mask_strand = None

    # write coverage targets table
    targets_out_df = targets_strand_df.reset_index(drop=True)
    targets_out_df.to_csv(f"{args.out_dir}/targets_cov.txt", sep="\t")

    # write gene targets table
    if gene_stats and args.targets_gene_file:
        targets_gene_df = pd.read_csv(args.targets_gene_file, sep="\t", index_col=0)
        targets_gene_df.to_csv(f"{args.out_dir}/targets_gene.txt", sep="\t")

    # write covgene targets table (gene-track subset of strand-collapsed targets)
    if covgene_stats and score_genes:
        targets_out_df[gene_mask_strand].to_csv(
            f"{args.out_dir}/targets_covgene.txt", sep="\t"
        )

    #################################################################
    # sequence dataset

    # read sequences from BED
    seqs_dna, seqs_coords = bed.make_bed_seqs(
        args.bed_file, args.genome_fasta, params_model["seq_length"], stranded=True
    )
    num_seqs = len(seqs_dna)

    # determine mutation region limits
    seq_mid = params_model["seq_length"] // 2
    mut_start = seq_mid - args.mut_up
    mut_end = mut_start + args.mut_len

    #################################################################
    # load genes

    if score_genes:
        transcriptome = Transcriptome(args.genes_gtf)
        gene_trees = transcriptome.gene_trees()

        # find overlapping genes for each sequence
        seq_genes = []
        all_gene_ids = set()
        for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
            pstart = seq_start + output_crop
            pend = seq_end - output_crop
            genes = find_overlapping_genes(gene_trees, seq_chr, pstart, pend)
            seq_genes.append(genes)
            for g in genes:
                all_gene_ids.add(g.kv["gene_id"])

        all_gene_ids = sorted(all_gene_ids)
        gene_to_index = {gid: idx for idx, gid in enumerate(all_gene_ids)}

        # build sequence-gene pair index
        pair_seq_idx = []
        pair_gene_idx = []
        for si, genes in enumerate(seq_genes):
            for gene in genes:
                pair_seq_idx.append(si)
                pair_gene_idx.append(gene_to_index[gene.kv["gene_id"]])
        num_pairs = len(pair_seq_idx)
    else:
        seq_genes = [[] for _ in range(num_seqs)]
        num_pairs = 0

    #################################################################
    # setup ISM analyzer

    ism_analyzer = ism.ISM(
        seqnn_model=seqnn_model,
        targets_df=targets_df,
        cov_stats=cov_stats,
        covgene_stats=covgene_stats,
        gene_stats=gene_stats,
        strand_transform=strand_transform,
        head=args.head,
    )

    #################################################################
    # setup output

    scores_h5_file = "%s/scores.h5" % args.out_dir
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, "w")
    scores_h5.create_dataset("progress_status", data="initialized".encode("utf-8"))
    scores_h5.create_dataset("seqs", dtype="bool", shape=(num_seqs, args.mut_len, 4))

    # cov stat datasets
    for cov_stat in cov_stats:
        scores_h5.create_dataset(
            cov_stat, dtype="float16", shape=(num_seqs, args.mut_len, 4, num_targets)
        )

    # store mutagenesis sequence coordinates
    scores_chr = []
    scores_start = []
    scores_end = []
    scores_strand = []
    for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
        scores_chr.append(seq_chr)
        scores_strand.append(seq_strand)
        if seq_strand == "+":
            score_start = seq_start + mut_start
            score_end = score_start + args.mut_len
        else:
            score_end = seq_end - mut_start
            score_start = score_end - args.mut_len
        scores_start.append(score_start)
        scores_end.append(score_end)

    scores_h5.create_dataset("chr", data=np.array(scores_chr, dtype="S"))
    scores_h5.create_dataset("start", data=np.array(scores_start))
    scores_h5.create_dataset("end", data=np.array(scores_end))
    scores_h5.create_dataset("strand", data=np.array(scores_strand, dtype="S"))

    # gene pair-indexed datasets
    if score_genes and num_pairs > 0:
        scores_h5.create_dataset("gene_ids", data=np.array(all_gene_ids, dtype="S"))
        scores_h5.create_dataset("seq_idx", data=np.array(pair_seq_idx))
        scores_h5.create_dataset("gene_idx", data=np.array(pair_gene_idx))

        for stat in covgene_stats:
            scores_h5.create_dataset(
                stat,
                dtype="float16",
                shape=(num_pairs, args.mut_len, 4, int(gene_mask_strand.sum())),
            )
        for stat in gene_stats:
            scores_h5.create_dataset(
                stat,
                dtype="float16",
                shape=(num_pairs, args.mut_len, 4, gene_depth),
            )

    #################################################################
    # predict scores, write output

    # pair write index
    pgi = 0

    with (
        torch.no_grad(),
        torch.autocast(device_type=seqnn_model.device, dtype=args.mix_dtype),
    ):
        for si, seq_dna in enumerate(seqs_dna):
            print("Predicting %d" % si, flush=True)

            # 1 hot code DNA
            seq_1hot = dna.dna_1hot(seq_dna)

            # save sequence
            scores_h5["seqs"][si] = seq_1hot[mut_start:mut_end].astype("bool")

            # build gene masks for this sequence
            genes = seq_genes[si]
            if genes and (covgene_stats or gene_stats):
                seq_chr, seq_start, seq_end, seq_strand = seqs_coords[si]
                pstart = seq_start + output_crop
                pend = seq_end - output_crop
                pred_length = pend - pstart
                gom, gp = build_gene_masks(
                    genes,
                    pstart,
                    pred_length,
                    output_stride,
                    output_length,
                    seqnn_model.device,
                )
                gene_strands = [g.strand for g in genes]
            else:
                gom = gp = None
                gene_strands = None

            # compute ISM scores (ISM expects [channels, length])
            if gom is not None:
                result = ism_analyzer.compute(
                    seq_1hot.T,
                    mut_start,
                    mut_end,
                    gene_out_mask=gom,
                    gene_presence=gp,
                    plus_mask=plus_mask,
                    minus_mask=minus_mask,
                    gene_strands=gene_strands,
                )

                # write cov scores
                for stat in cov_stats:
                    scores_h5[stat][si] = result.cov[stat]

                # write gene scores
                for gi, gene in enumerate(genes):
                    for stat in covgene_stats:
                        scores_h5[stat][pgi] = result.covgene[gi][stat]
                    for stat in gene_stats:
                        scores_h5[stat][pgi] = result.gene[gi][stat]
                    pgi += 1
            else:
                # no genes: compute cov-only
                result = ism_analyzer.compute(seq_1hot.T, mut_start, mut_end)
                for stat in cov_stats:
                    scores_h5[stat][si] = result.cov[stat]

                # advance pair index for genes with no overlap
                # (no pairs to write for this sequence)

    # mark completion and close output HDF5
    del scores_h5["progress_status"]
    scores_h5.create_dataset("progress_status", data="completed".encode("utf-8"))
    scores_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
