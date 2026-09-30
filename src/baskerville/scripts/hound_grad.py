#!/usr/bin/env python
import argparse
import gc
import json
import os

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch

from baskerville.dataset import annotate_strand
from baskerville.hardware import check_mixed_precision
from baskerville import dna as dna_io
from baskerville import gene as bgene
from baskerville.seqnn import SeqNN

"""
hound_grad

Calculate gradients for genes in a GTF file, averaged across a set of tracks.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Calculate gradients for genes in a GTF file, averaged across a set of tracks"
    )
    parser.add_argument(
        "--bigwig", default=False, action="store_true", help="Output bigwig files"
    )
    parser.add_argument(
        "-f",
        dest="genome_fasta",
        required=True,
        help="Genome FASTA for sequences (required)",
    )
    parser.add_argument(
        "--log",
        dest="log_transform",
        action="store_true",
        help="Apply log transformation to sum of coverage",
    )
    parser.add_argument("--head", type=int, default=0, help="Model head index")
    parser.add_argument("-o", "--out_dir", default="grad_out", help="Output directory")
    parser.add_argument(
        "--rc", action="store_true", help="Add reverse complement augmentation"
    )
    parser.add_argument(
        "-t", "--targets_file", required=True, help="Targets table (required)"
    )
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("model_file", help="Model file to use for predictions")
    parser.add_argument("gene_gtf", help="Gene GTF file")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    #################################################################
    # Read parameters and targets
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params.get("train", {})
    seq_len = params_model["seq_length"]

    targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)
    targets_df = annotate_strand(targets_df)

    #################################################################
    # Load model
    seqnn_model = SeqNN(params_model, output_slice=np.array(targets_df.index.values))
    seqnn_model.restore(
        args.model_file, strict=params_model.get("untransform") != "flashzoi"
    )
    seqnn_model.set_device()

    model_stride = seqnn_model.output_stride()
    model_crop = seqnn_model.output_crop_bp()
    target_length = seqnn_model.output_length()

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

    #################################################################
    # Read genes from GTF
    transcriptome = bgene.Transcriptome(args.gene_gtf)
    genome_open = pysam.Fastafile(args.genome_fasta)
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)

    min_start = -model_crop
    genes_chr, genes_start, genes_end, genes_strand = [], [], [], []
    for gene_id in gene_list:
        gene = transcriptome.genes[gene_id]
        genes_chr.append(gene.chrom)
        genes_strand.append(gene.strand)
        gene_midpoint = gene.midpoint()
        gene_start = max(min_start, gene_midpoint - seq_len // 2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)

    print("n genes = " + str(len(genes_chr)))
    scores_h5_file = os.path.join(args.out_dir, f"scores.h5")
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, "w")
    scores_h5.create_dataset("seqs", dtype="bool", shape=(num_genes, seq_len, 4))
    scores_h5.create_dataset("grads", dtype="float16", shape=(num_genes, seq_len, 4))
    scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
    scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
    scores_h5.create_dataset("start", data=np.array(genes_start))
    scores_h5.create_dataset("end", data=np.array(genes_end))
    scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))

    #################################################################
    # Compute gradients

    for gi, gene_id in enumerate(gene_list):
        gene = transcriptome.genes[gene_id]
        seq_1hot = make_seq_1hot(
            genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
        )
        scores_h5["seqs"][gi] = seq_1hot

        for rev_comp in [False, True] if args.rc else [False]:
            seq_1hot_torch = torch.from_numpy(seq_1hot).float().T.to(seqnn_model.device)
            if rev_comp:
                seq_1hot_torch = dna_io.torch_rc(seq_1hot_torch)

            # Determine output sequence coordinates
            seq_out_start = genes_start[gi] + model_crop
            seq_out_len = model_stride * target_length
            gene_slice = gene.output_slice(seq_out_start, seq_out_len, model_stride)
            if rev_comp:
                gene_slice = target_length - gene_slice - 1

            # Select strand-appropriate targets. Under RC we pick the
            # opposite-strand tracks (where the gene's signal lands) instead of
            # channel-swapping via strand_pair; gradients() reduces tasks to a
            # scalar before backprop, so there's nothing to realign. Adding a
            # strand_pair swap here would double-correct.
            gene_indices = select_strand_indices(targets_df, genes_strand[gi], rev_comp)
            grads = seqnn_model.gradients(
                seq_1hot_torch,
                hi=args.head,
                spatial_slice=gene_slice,
                task_slice=gene_indices,
                untransform_targets_df=targets_df,
                log_transform=args.log_transform,
            )
            grads = grads.detach().cpu().numpy().T
            if rev_comp:
                # map gradients on the RC input back to genomic coordinates
                grads = dna_io.hot1_rc(grads)
            scores_h5["grads"][gi] += grads
            gc.collect()

    # normalize gradients
    for gi, gene_id in enumerate(gene_list):
        scores_h5["grads"][gi] /= float(2 if args.rc else 1)
        scores_h5["grads"][gi] -= scores_h5["grads"][gi].mean(axis=1, keepdims=True)

    # write BigWigs
    if args.bigwig:
        write_bigwigs(
            scores_h5,
            genome_open,
            args.out_dir,
        )

    genome_open.close()
    scores_h5.close()


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    """Fetch a genomic sequence and convert it to a one-hot encoded format,
    padding with Ns if necessary.

    Args:
        genome_open: pysam.Fastafile object for the genome FASTA
        chrm: Chromosome name (string)
        start: Start coordinate (0-based, inclusive)
        end: End coordinate (0-based, exclusive)
        seq_len: Desired sequence length (int)
    """
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


def write_bigwigs(scores_h5, genome_open, out_dir):
    """Write two BigWig tracks per gene summarizing per-base gradients.

    For each gene window, emit:
      <gene>.grad_ref.bw : gradient corresponding to the reference (observed) nucleotide per position.
      <gene>.grad_var.bw : variance of gradients across the four nucleotides per position.

    Args:
        scores_h5: HDF5 with datasets 'seqs' (num_genes, L, 4), 'grads', 'chr', 'start', 'end', 'gene'.
        genome_open: pysam.Fastafile for chromosome lengths & sequences.
        out_dir: Output directory for BigWig files.
    """
    genes_chr = [chrom.decode("UTF-8") for chrom in scores_h5["chr"]]
    genes_start = scores_h5["start"][:]
    genes_ids = [gene.decode("UTF-8") for gene in scores_h5["gene"]]
    num_genes, seq_len, _ = scores_h5["seqs"].shape

    # chromosome lengths
    chrom_lengths = {
        chrm: genome_open.get_reference_length(chrm) for chrm in genome_open.references
    }

    def sanitize(name: str) -> str:
        return "".join(
            c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name
        )

    for gi in range(num_genes):
        chrm = genes_chr[gi]
        gene_id = genes_ids[gi]
        chrm_len = chrom_lengths[chrm]
        seq_start = int(genes_start[gi])

        seq_1hot = scores_h5["seqs"][gi]
        grads = scores_h5["grads"][gi]

        # compute nucleotide scores
        ref_scores = (seq_1hot * grads).sum(axis=1)
        var_scores = grads.astype(np.float32).var(axis=1)

        # determine valid genomic positions
        pos_idx = np.arange(seq_len)
        genomic_positions = pos_idx + seq_start
        valid = (genomic_positions >= 0) & (genomic_positions < chrm_len)
        genomic_positions = genomic_positions[valid]
        if genomic_positions.size == 0:
            raise ValueError(
                f"No valid genomic positions for gene {gene_id} on {chrm}:{seq_start}-{seq_start + seq_len}"
            )

        # write BigWigs
        header = [(chrm, chrm_len)]
        gene_safe = sanitize(gene_id)
        ref_path = os.path.join(out_dir, f"{gene_safe}_ref.bw")
        var_path = os.path.join(out_dir, f"{gene_safe}_var.bw")
        for f in (ref_path, var_path):
            if os.path.exists(f):
                os.remove(f)

        ref_bw = pyBigWig.open(ref_path, "w")
        var_bw = pyBigWig.open(var_path, "w")
        ref_bw.addHeader(header)
        var_bw.addHeader(header)

        starts = genomic_positions.tolist()
        ends = (genomic_positions + 1).tolist()
        ref_values = ref_scores[valid].astype(float).tolist()
        var_values = var_scores[valid].astype(float).tolist()

        ref_bw.addEntries([chrm] * len(starts), starts, ends=ends, values=ref_values)
        var_bw.addEntries([chrm] * len(starts), starts, ends=ends, values=var_values)

        ref_bw.close()
        var_bw.close()


def select_strand_indices(targets_df, gene_strand, rev_comp):
    """Positional indices of targets carrying a gene's signal in this orientation.

    Reverse-complementing the input flips the genome strand, so the gene's
    signal moves to the opposite-strand tracks; unstranded ('.') tracks are
    always included.
    """
    if gene_strand == "+":
        drop_strand = "+" if rev_comp else "-"
    else:
        drop_strand = "-" if rev_comp else "+"
    return np.where(targets_df.strand != drop_strand)[0]


if __name__ == "__main__":
    main()
