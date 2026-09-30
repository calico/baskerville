#!/usr/bin/env python
# Copyright 2017 Calico Life Sciences LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import argparse
import gzip
import json
import pdb
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
import h5py

from baskerville import data
from baskerville import dataset
from baskerville.gene import Transcriptome

try:
    from baskerville import utils
except ModuleNotFoundError:
    pass

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None

"""
hound_data

Compute model sequences from the genome, extracting DNA coverage values.
"""


################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Compute model sequences from the genome, extracting DNA coverage values."
    )
    parser.add_argument(
        "-b",
        "--blacklist_bed",
        help="Set blacklist nucleotides to a baseline value.",
    )
    parser.add_argument(
        "--break",
        dest="break_t",
        default=786432,
        type=int,
        help="Break in half contigs above length [Default: %(default)s]",
    )
    parser.add_argument(
        "-c",
        "--crop",
        dest="crop_bp",
        default=0,
        type=int,
        help="Crop bp off each end [Default: %(default)s]",
    )
    parser.add_argument(
        "-d",
        "--decimals",
        default=None,
        type=int,
        help="Round values to given decimals [Default: %(default)s]",
    )
    parser.add_argument(
        "--folds",
        default=None,
        type=int,
        help="Generate cross fold split [Default: %(default)s]",
    )
    parser.add_argument(
        "--fasta",
        dest="fasta_file",
        required=True,
        help="FASTA genome file",
    )
    parser.add_argument(
        "-g", "--gaps_file", help="Genome assembly gaps BED [Default: %(default)s]"
    )
    parser.add_argument(
        "--gene_cov_t",
        dest="gene_coverage_threshold",
        default=0.5,
        type=float,
        help="Minimum gene coverage fraction to include [Default: %(default)s]",
    )
    parser.add_argument(
        "--gtf",
        dest="gtf_file",
        help="GTF file for gene annotations [Default: %(default)s]",
    )
    parser.add_argument(
        "--gene_span",
        default=False,
        action="store_true",
        help="Use full gene span instead of exons only [Default: %(default)s]",
    )
    parser.add_argument(
        "--max_genes",
        default=16,
        type=int,
        help="Maximum genes per sequence; excess filtered by centrality×coverage [Default: %(default)s]",
    )
    parser.add_argument(
        "-i",
        "--interp_nan",
        default=False,
        action="store_true",
        help="Interpolate NaNs [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=131072,
        type=int,
        help="Sequence length [Default: %(default)s]",
    )
    parser.add_argument(
        "--limit",
        dest="limit_bed",
        help="Limit to segments that overlap regions in a BED file",
    )
    parser.add_argument(
        "--local",
        dest="run_local",
        default=False,
        action="store_true",
        help="Run jobs locally as opposed to on SLURM [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        type=int,
        help="Number parallel processes [Default: %(default)s]",
    )
    parser.add_argument(
        "--peaks",
        dest="peaks_only",
        default=False,
        action="store_true",
        help="Create contigs only from peaks [Default: %(default)s]",
    )
    parser.add_argument(
        "-r",
        "--seqs_per_write",
        default=128,
        type=int,
        help="Sequences per write job [Default: %(default)s]",
    )
    parser.add_argument(
        "--restart",
        default=False,
        action="store_true",
        help="Continue progress from midpoint. [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--sample_pct",
        default=1.0,
        type=float,
        help="Down-sample the segments",
    )
    parser.add_argument(
        "--seed",
        default=44,
        type=int,
        help="Random seed [Default: %(default)s]",
    )
    parser.add_argument(
        "--snap",
        default=1,
        type=int,
        help="Snap sequences to multiple of the given value [Default: %(default)s]",
    )
    parser.add_argument(
        "--st",
        "--split_test",
        dest="split_test",
        default=False,
        action="store_true",
        help="Exit after split. [Default: %(default)s]",
    )
    parser.add_argument(
        "--stride",
        "--stride_train",
        dest="stride_train",
        default=1.0,
        type=float,
        help="Stride to advance train sequences [Default: seq_length]",
    )
    parser.add_argument(
        "--stride_test",
        default=1.0,
        type=float,
        help="Stride to advance valid and test sequences [Default: seq_length]",
    )
    parser.add_argument(
        "-t",
        "--test_pct_or_chr",
        default="0.05",
        type=str,
        help="Proportion of the data for testing [Default: %(default)s]",
    )
    parser.add_argument(
        "--targets_cov",
        dest="targets_cov_file",
        help="Coverage targets table [Default: %(default)s]",
    )
    parser.add_argument(
        "--targets_gene",
        dest="targets_gene_file",
        help="Gene targets table [Default: %(default)s]",
    )
    parser.add_argument("-u", "--umap_bed", help="Unmappable regions in BED format")
    parser.add_argument(
        "--umap_t",
        default=0.5,
        type=float,
        help="Remove sequences with more than this unmappable bin %% [Default: %(default)s]",
    )
    parser.add_argument(
        "--umap_clip",
        default=1,
        type=float,
        help="Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %(default)s]",
    )
    parser.add_argument(
        "-w",
        "--pool_width",
        default=32,
        type=int,
        help="Sum pool width [Default: %(default)s]",
    )
    parser.add_argument(
        "-v",
        "--valid_pct_or_chr",
        default="0.05",
        type=str,
        help="Proportion of the data for validation [Default: %(default)s]",
    )
    parser.add_argument(
        "-z",
        "--zarr_chunks",
        default=1,
        type=int,
        help="Number of chunks per Zarr example [Default: %(default)s]",
    )
    parser.add_argument(
        "--write_mem",
        default=60000,
        type=int,
        help="Memory in MB per write job [Default: %(default)s]",
    )
    args = parser.parse_args()
    if slurmrunner is None and not args.run_local:
        print("slurmrunner not installed, running locally")
        args.run_local = True

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.break_t is not None and args.break_t < args.seq_length:
        print(
            "Maximum contig length --break cannot be less than sequence length.",
            file=sys.stderr,
        )
        exit(1)

    # transform proportion strides to base pairs
    if args.stride_train <= 1:
        print("stride_train %.f" % args.stride_train, end="")
        args.stride_train = args.stride_train * args.seq_length
        print(" converted to %f" % args.stride_train)
    args.stride_train = int(np.round(args.stride_train))
    if args.stride_test <= 1:
        if args.folds is None:
            print("stride_test %.f" % args.stride_test, end="")
            args.stride_test = args.stride_test * args.seq_length
            print(" converted to %f" % args.stride_test)
    args.stride_test = int(np.round(args.stride_test))

    # check snap
    if args.snap is not None:
        if np.mod(args.seq_length, args.snap) != 0:
            raise ValueError("seq_length must be a multiple of snap")
        if np.mod(args.stride_train, args.snap) != 0:
            raise ValueError("stride_train must be a multiple of snap")
        if np.mod(args.stride_test, args.snap) != 0:
            raise ValueError("stride_test must be a multiple of snap")

    # setup output directory
    if os.path.isdir(args.out_dir) and not args.restart:
        print(f"Remove output directory {args.out_dir} or use --restart option.")
        exit(1)
    elif not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # read coverage target datasets
    if args.targets_cov_file is not None:
        targets_df = pd.read_csv(args.targets_cov_file, index_col=0, sep="\t")
        num_targets = targets_df.shape[0]
        shutil.copy(args.targets_cov_file, f"{args.out_dir}/targets.txt")
    else:
        targets_df = pd.DataFrame()
        num_targets = 0

    # read gene target datasets
    if args.targets_gene_file is not None:
        targets_gene_df = pd.read_csv(args.targets_gene_file, index_col=0, sep="\t")
        num_targets_gene = targets_gene_df.shape[0]
        shutil.copy(args.targets_gene_file, f"{args.out_dir}/targets_gene.txt")

        if args.gtf_file is None:
            parser.error("Must provide --gtf file when using gene expression targets.")
    else:
        targets_gene_df = pd.DataFrame()
        num_targets_gene = 0

    # Validate that at least one data type is provided
    if num_targets == 0 and num_targets_gene == 0:
        parser.error(
            "Must provide at least one target dataset (coverage or gene expression)."
        )

    ################################################################
    # define genomic contigs
    ################################################################
    if not args.restart:
        chrom_contigs = data.load_chromosomes(args.fasta_file)

        # remove gaps
        if args.gaps_file:
            chrom_contigs = data.split_contigs(chrom_contigs, args.gaps_file)

        # ditch the chromosomes for contigs
        contigs = []
        for chrom in chrom_contigs:
            contigs += [
                data.Contig(0, chrom, ctg_start, ctg_end)
                for ctg_start, ctg_end in chrom_contigs[chrom]
            ]

        # limit to a BED file
        if args.limit_bed is not None:
            contigs = limit_contigs(contigs, args.limit_bed)

        # limit to peaks
        if args.peaks_only:
            peaks_bed = curate_peaks(
                targets_df, args.out_dir, args.pool_width, args.crop_bp
            )
            contigs = limit_contigs(contigs, peaks_bed)

        # filter for large enough
        seq_tlength = args.seq_length - 2 * args.crop_bp
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]

        # break up large contigs
        if args.break_t is not None:
            contigs = data.break_large_contigs(contigs, args.break_t)

        # print contigs to BED file
        # ctg_bed_file = '%s/contigs.bed' % args.out_dir
        # write_seqs_bed(ctg_bed_file, contigs)

    ################################################################
    # divide between train/valid/test
    ################################################################
    # label folds
    if args.folds is not None:
        fold_labels = [f"fold{fi}" for fi in range(args.folds)]
        num_folds = args.folds
    else:
        fold_labels = ["train", "valid", "test"]
        num_folds = 3

    if not args.restart:
        if args.folds is not None:
            # divide by fold pct
            fold_contigs = divide_contigs_folds(contigs, args.folds)

        else:
            try:
                # convert to float pct
                valid_pct = float(args.valid_pct_or_chr)
                test_pct = float(args.test_pct_or_chr)
                assert 0 <= valid_pct <= 1
                assert 0 <= test_pct <= 1

                # divide by pct
                fold_contigs = divide_contigs_pct(contigs, test_pct, valid_pct)

            except (ValueError, AssertionError):
                # divide by chr
                valid_chrs = args.valid_pct_or_chr.split(",")
                test_chrs = args.test_pct_or_chr.split(",")
                fold_contigs = divide_contigs_chr(contigs, test_chrs, valid_chrs)

        # rejoin broken contigs within set
        for fi in range(len(fold_contigs)):
            fold_contigs[fi] = data.rejoin_large_contigs(fold_contigs[fi])

        # write labeled contigs to BED file
        ctg_bed_file = f"{args.out_dir}/contigs.bed"
        ctg_bed_out = open(ctg_bed_file, "w")
        for fi in range(len(fold_contigs)):
            for ctg in fold_contigs[fi]:
                line = "%s\t%d\t%d\t%s" % (ctg.chr, ctg.start, ctg.end, fold_labels[fi])
                print(line, file=ctg_bed_out)
        ctg_bed_out.close()

    if args.split_test:
        exit()

    ################################################################
    # define model sequences
    ################################################################
    if not args.restart:
        fold_mseqs = []
        for fi in range(num_folds):
            if fold_labels[fi] in ["valid", "test"]:
                stride_fold = args.stride_test
            else:
                stride_fold = args.stride_train

            # stride sequences across contig
            fold_mseqs_fi = data.contig_sequences(
                fold_contigs[fi],
                seq_tlength,
                stride_fold,
                args.snap,
                fold_labels[fi],
            )
            fold_mseqs.append(fold_mseqs_fi)

            # shuffle
            random.shuffle(fold_mseqs[fi])

            # down-sample
            if args.sample_pct < 1.0:
                fold_mseqs[fi] = random.sample(
                    fold_mseqs[fi], int(args.sample_pct * len(fold_mseqs[fi]))
                )

        # merge into one list
        mseqs = [ms for fm in fold_mseqs for ms in fm]

    ################################################################
    # mappability
    ################################################################
    if not args.restart:
        if args.umap_bed is not None:
            if shutil.which("bedtools") is None:
                print("Install Bedtools to annotate unmappable sites", file=sys.stderr)
                exit(1)

            # annotate unmappable positions
            mseqs_unmap = data.annotate_unmap(
                mseqs, args.umap_bed, seq_tlength, args.pool_width
            )

            # filter unmappable
            mseqs_map_mask = mseqs_unmap.mean(axis=1, dtype="float64") < args.umap_t
            mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
            mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

            # write to file
            unmap_npy = f"{args.out_dir}/mseqs_unmap.npy"
            np.save(unmap_npy, mseqs_unmap)

        # write sequences to BED
        seqs_bed_file = f"{args.out_dir}/sequences.bed"
        data.write_seqs_bed(seqs_bed_file, mseqs, True)

    else:
        # read from directory
        seqs_bed_file = f"{args.out_dir}/sequences.bed"
        unmap_npy = f"{args.out_dir}/mseqs_unmap.npy"
        mseqs = []
        fold_mseqs = []
        for fi in range(num_folds):
            fold_mseqs.append([])

        # append extra list for 'free'-labeled sequences
        fold_mseqs.append([])

        for line in open(seqs_bed_file):
            a = line.split()
            msg = data.ModelSeq(0, a[0], int(a[1]), int(a[2]), a[3])
            mseqs.append(msg)
            if a[3] == "train":
                fi = 0
            elif a[3] == "valid":
                fi = 1
            elif a[3] == "test":
                fi = 2
            elif a[3] == "free":
                fi = -1
            else:
                fi = int(a[3].replace("fold", ""))
            fold_mseqs[fi].append(msg)

        # delete last list if no 'free'-labeled entries were found
        if len(fold_mseqs[-1]) == 0:
            del fold_mseqs[-1]
        else:
            # extend fold labels
            fold_labels += ["free"]

    ################################################################
    # read sequence coverage values (if coverage targets exist)
    ################################################################
    if num_targets > 0:
        seqs_cov_dir = f"{args.out_dir}/seqs_cov"
        os.makedirs(seqs_cov_dir, exist_ok=True)

        read_jobs = []
        for ti in range(num_targets):
            seqs_cov_stem = f"{seqs_cov_dir}/{ti}"
            seqs_cov_file = f"{seqs_cov_stem}.h5"

            exec_read_job = False
            if args.restart and os.path.isfile(seqs_cov_file):
                try:
                    # open file to check for corruption
                    seqs_cov_open = h5py.File(seqs_cov_file, "r")
                    seqs_cov_open.close()
                    print(f"Skipping existing {seqs_cov_file}", file=sys.stderr)
                except OSError:
                    exec_read_job = True
                    print(f"Re-starting corrupted {seqs_cov_file}", file=sys.stderr)
            else:
                exec_read_job = True

            if exec_read_job:
                cmd = "python -m baskerville.scripts.hound_data_read"
                if args.blacklist_bed:
                    cmd += f" -b {args.blacklist_bed}"
                cmd += f" -d {args.out_dir}"
                if args.interp_nan:
                    cmd += " -i"
                cmd += f" -t {ti}"
                cmd += f" -w {args.pool_width}"

                if args.run_local:
                    # breaks on some OS
                    # cmd += ' &> %s.err' % seqs_cov_stem
                    read_jobs.append(cmd)
                else:
                    j = slurmrunner.Job(
                        cmd,
                        name=f"read_t{ti}",
                        out_file=f"{seqs_cov_stem}.out",
                        err_file=f"{seqs_cov_stem}.err",
                        queue="standard",
                        mem=15000,
                        time="12:0:0",
                    )
                    read_jobs.append(j)

        if args.run_local:
            utils.exec_par(read_jobs, args.processes, verbose=True)
        else:
            slurmrunner.multi_run(
                read_jobs,
                args.processes,
                verbose=True,
                launch_sleep=0.5,
                update_sleep=3,
            )

    ################################################################
    # read gene expression values (if provided)
    ################################################################
    max_genes = 0
    if num_targets_gene > 0:
        print("Processing gene expression data...")

        # Parse GTF and map genes to sequences
        transcriptome = Transcriptome(args.gtf_file)

        gene_data_dir = f"{args.out_dir}/seqs_gene"
        os.makedirs(gene_data_dir, exist_ok=True)

        # Create gene mapping (with restart protection)
        gene_mapping_zarr = f"{gene_data_dir}/gene_mapping.zarr"
        if args.restart and os.path.isdir(gene_mapping_zarr):
            try:
                gene_mapping = zarr.open(gene_mapping_zarr, mode="r")
                max_genes = gene_mapping["gene_presence"].shape[1]
                print(f"Skipping existing {gene_mapping_zarr}", file=sys.stderr)
            except Exception:
                print(f"Re-starting corrupted {gene_mapping_zarr}", file=sys.stderr)
                max_genes = map_genes_to_sequences(
                    mseqs,
                    transcriptome,
                    gene_data_dir,
                    args.seq_length,
                    args.pool_width,
                    args.gene_coverage_threshold,
                    args.max_genes,
                    gene_span=args.gene_span,
                )
        else:
            max_genes = map_genes_to_sequences(
                mseqs,
                transcriptome,
                gene_data_dir,
                args.seq_length,
                args.pool_width,
                args.gene_coverage_threshold,
                args.max_genes,
                gene_span=args.gene_span,
            )

        # Filter sequences without genes in gene-only mode (before read jobs)
        if num_targets == 0:
            gene_mapping = zarr.open(gene_mapping_zarr, mode="r")
            gene_presence_arr = gene_mapping["gene_presence"][:]
            has_genes = gene_presence_arr.any(axis=1)

            original_count = len(mseqs)
            filtered_count = has_genes.sum()

            if filtered_count == 0:
                raise ValueError(
                    "No sequences contain genes meeting the coverage threshold. "
                    "Consider lowering --gene_cov_t or checking your GTF file."
                )

            if filtered_count < original_count:
                # Filter mseqs
                mseqs = [mseqs[i] for i in range(len(mseqs)) if has_genes[i]]

                # Rebuild fold_mseqs from filtered mseqs by label
                fold_mseqs = [[] for _ in range(num_folds)]
                for ms in mseqs:
                    for fi, label in enumerate(fold_labels):
                        if ms.label == label:
                            fold_mseqs[fi].append(ms)
                            break

                print(
                    f"Gene-only mode: filtered {original_count - filtered_count}/{original_count} "
                    f"sequences without genes"
                )

                # Rewrite sequences.bed with filtered sequences
                data.write_seqs_bed(seqs_bed_file, mseqs, True)

                # Rewrite gene mapping zarr with filtered sequences
                filtered_gene_ids = gene_mapping["gene_ids"][:][has_genes]
                filtered_gene_presence = gene_presence_arr[has_genes]
                filtered_gene_out_mask = gene_mapping["gene_out_mask"][:][has_genes]

                # Overwrite the zarr with filtered data
                zarr_root = zarr.open(gene_mapping_zarr, mode="w")
                zarr_root.create_array(
                    "gene_ids",
                    data=filtered_gene_ids,
                    chunks=(1, max_genes),
                )
                zarr_root.create_array(
                    "gene_presence",
                    data=filtered_gene_presence,
                    chunks=(1, max_genes),
                )
                zarr_root.create_array(
                    "gene_out_mask",
                    data=filtered_gene_out_mask,
                    chunks=(1, max_genes, filtered_gene_out_mask.shape[2]),
                )

        # Launch jobs to read gene expression for each target
        read_gene_jobs = []
        for ti in range(num_targets_gene):
            gene_stem = f"{gene_data_dir}/{ti}"
            seqs_gene_file = f"{gene_stem}.h5"

            exec_read_gene_job = False
            if args.restart and os.path.isfile(seqs_gene_file):
                try:
                    seqs_gene_open = h5py.File(seqs_gene_file, "r")
                    seqs_gene_open.close()
                    print(f"Skipping existing {seqs_gene_file}", file=sys.stderr)
                except OSError:
                    exec_read_gene_job = True
                    print(f"Re-starting corrupted {seqs_gene_file}", file=sys.stderr)
            else:
                exec_read_gene_job = True

            if exec_read_gene_job:
                cmd = f"python -m baskerville.scripts.hound_data_read_gene"
                cmd += f" -d {args.out_dir}"
                cmd += f" -t {ti}"

                if args.run_local:
                    read_gene_jobs.append(cmd)
                else:
                    j = slurmrunner.Job(
                        cmd,
                        name=f"read_gene_t{ti}",
                        out_file=f"{gene_stem}.out",
                        err_file=f"{gene_stem}.err",
                        queue="standard",
                        mem=8000,
                        time="4:0:0",
                    )
                    read_gene_jobs.append(j)

        if args.run_local:
            utils.exec_par(read_gene_jobs, args.processes, verbose=True)
        else:
            slurmrunner.multi_run(
                read_gene_jobs,
                args.processes,
                verbose=True,
                launch_sleep=0.5,
                update_sleep=3,
            )

        # Save genes.txt with gene information
        genes_data = []
        for gene_id, gene in transcriptome.genes.items():
            exon_coords = ";".join([f"{e.begin}-{e.end}" for e in gene.get_exons()])
            genes_data.append(
                {
                    "gene_id": gene_id,
                    "gene_name": gene.name if gene.name else gene_id,
                    "chr": gene.chrom,
                    "strand": gene.strand,
                    "exon_coords": exon_coords,
                }
            )
        genes_df = pd.DataFrame(genes_data)
        genes_df.to_csv(f"{args.out_dir}/genes.txt", sep="\t", index=False)

    ################################################################
    # write examples
    ################################################################
    # initialize examples dir
    examples_dir = f"{args.out_dir}/examples"
    os.makedirs(examples_dir, exist_ok=True)

    # zarr prep
    compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    target_length = args.seq_length - 2 * args.crop_bp
    target_length = target_length // args.pool_width
    chunk_length = target_length // args.zarr_chunks

    write_jobs = []

    for fold_set in fold_labels:
        fold_set_indexes = [i for i in range(len(mseqs)) if mseqs[i].label == fold_set]
        fold_set_start = fold_set_indexes[0]
        fold_set_end = fold_set_indexes[-1] + 1
        fold_seqs = fold_set_end - fold_set_start

        # create zarr file
        fold_zarr_file = f"{examples_dir}/{fold_set}.zarr"

        fold_zarr_root = zarr.open_group(fold_zarr_file, mode="a")
        if not os.path.isdir(f"{fold_zarr_file}/sequence"):
            fold_zarr_root.create_array(
                "sequence",
                shape=(fold_seqs, args.seq_length),
                chunks=(1, args.seq_length),
                dtype="uint8",
            )

        # Create coverage target array only if coverage targets exist
        if num_targets > 0 and not os.path.isdir(f"{fold_zarr_file}/target"):
            fold_zarr_root.create_array(
                "target",
                shape=(fold_seqs, num_targets, target_length),
                chunks=(1, num_targets, chunk_length),
                dtype="float16",
                compressors=compressors,
            )

        # Create gene zarr arrays if gene data is provided
        if num_targets_gene > 0:
            if not os.path.isdir(f"{fold_zarr_file}/gene_target"):
                fold_zarr_root.create_array(
                    "gene_target",
                    shape=(fold_seqs, num_targets_gene, max_genes),
                    chunks=(1, num_targets_gene, max_genes),
                    dtype="float16",
                )
            if not os.path.isdir(f"{fold_zarr_file}/gene_presence"):
                fold_zarr_root.create_array(
                    "gene_presence",
                    shape=(fold_seqs, max_genes),
                    chunks=(1, max_genes),
                    dtype="bool",
                )
            if not os.path.isdir(f"{fold_zarr_file}/gene_out_mask"):
                fold_zarr_root.create_array(
                    "gene_out_mask",
                    shape=(fold_seqs, max_genes, target_length),
                    chunks=(1, max_genes, target_length),
                    dtype="bool",
                    compressors=compressors,
                )
            if not os.path.isdir(f"{fold_zarr_file}/gene_ids"):
                fold_zarr_root.create_array(
                    "gene_ids",
                    shape=(fold_seqs, max_genes),
                    chunks=(1, max_genes),
                    dtype="<U50",
                )

        # Initialize job counters for this fold
        ex_start = fold_set_start
        ji = 0
        ex_end = min(ex_start + args.seqs_per_write, fold_set_end)

        while ex_start < fold_set_end:
            # check if job has completed successfully
            success_file = f"{examples_dir}/{fold_set}-{ji}-success.txt"
            if not os.path.isfile(success_file):
                # create command
                cmd = "python -m baskerville.scripts.hound_data_write"
                if args.decimals is not None:
                    cmd += f" -d {args.decimals}"
                cmd += f" -f {fold_set}"
                cmd += f" -s {ex_start}"
                cmd += f" -e {ex_end}"
                cmd += f" -j {ji}"
                cmd += f" --umap_clip {args.umap_clip}"
                if args.umap_bed is not None:
                    cmd += f" -u {unmap_npy}"
                cmd += f" -x {args.crop_bp}"
                if args.gtf_file is not None and args.targets_gene_file is not None:
                    cmd += f" --gene_dir {gene_data_dir}"
                if num_targets > 0:
                    cmd += f" --cov_dir {seqs_cov_dir}"

                cmd += f" {args.fasta_file}"
                cmd += f" {seqs_bed_file}"
                cmd += f" {fold_zarr_file}"

                if args.run_local:
                    write_jobs.append(cmd)
                else:
                    job_stem = f"{examples_dir}/{fold_set}-{ji}"
                    j = slurmrunner.Job(
                        cmd,
                        name=f"write_{fold_set}-{ji}",
                        out_file=f"{job_stem}.out",
                        err_file=f"{job_stem}.err",
                        queue="standard",
                        mem=args.write_mem,
                        time="12:0:0",
                    )
                    write_jobs.append(j)

            # update
            ji += 1
            ex_start += args.seqs_per_write
            ex_end = min(ex_start + args.seqs_per_write, fold_set_end)

    if args.run_local:
        utils.exec_par(write_jobs, args.processes, verbose=True)
    else:
        slurmrunner.multi_run(
            write_jobs,
            args.processes,
            verbose=True,
            launch_sleep=0.5,
            update_sleep=3,
        )

    # per-track means, for depth-normalized specificity metrics
    if num_targets > 0:
        dataset.write_target_means(args.out_dir, processes=args.processes)

    ################################################################
    # stats
    ################################################################
    stats_dict = {}
    stats_dict["num_targets"] = num_targets
    stats_dict["seq_length"] = args.seq_length
    stats_dict["seq_1hot"] = True
    stats_dict["pool_width"] = args.pool_width
    stats_dict["crop_bp"] = args.crop_bp
    stats_dict["target_length"] = target_length

    for fi in range(num_folds):
        stats_dict["%s_seqs" % fold_labels[fi]] = len(fold_mseqs[fi])

    if fold_labels[-1] == "free":
        stats_dict["free_seqs"] = len(fold_mseqs[-1])

    # Add gene metadata if available
    if num_targets_gene > 0:
        stats_dict["num_targets_gene"] = num_targets_gene
        stats_dict["max_genes_per_seq"] = max_genes
        stats_dict["gene_coverage_threshold"] = args.gene_coverage_threshold

    with open(f"{args.out_dir}/statistics.json", "w") as stats_json_out:
        json.dump(stats_dict, stats_json_out, indent=4)


################################################################################
def curate_peaks(targets_df, out_dir, pool_width, crop_bp):
    """Merge all peaks, round to nearest pool_width, and add cropped bp."""

    # concatenate and extend peaks
    cat_bed_file = "%s/peaks_cat.bed" % out_dir
    cat_bed_out = open(cat_bed_file, "w")
    for bed_file in targets_df.file:
        if bed_file[-3:] == ".gz":
            bed_in = gzip.open(bed_file, "rt")
        else:
            bed_in = open(bed_file, "r")

        for line in bed_in:
            a = line.rstrip().split("\t")
            chrm = a[0]
            start = int(a[1])
            end = int(a[2])

            # extend to pool width
            length = end - start
            if length < pool_width:
                mid = (start + end) // 2
                start = mid - pool_width // 2
                end = start + pool_width

            # add cropped bp
            start = max(0, start - crop_bp)
            end += crop_bp

            # print
            print("%s\t%d\t%d" % (chrm, start, end), file=cat_bed_out)

        bed_in.close()
    cat_bed_out.close()

    # merge
    merge_bed_file = "%s/peaks_merge.bed" % out_dir
    bedtools_cmd = "bedtools sort -i %s" % cat_bed_file
    bedtools_cmd += " | bedtools merge -i - > %s" % merge_bed_file
    subprocess.call(bedtools_cmd, shell=True)

    # round and add crop_bp
    full_bed_file = "%s/peaks_full.bed" % out_dir
    full_bed_out = open(full_bed_file, "w")

    for line in open(merge_bed_file):
        a = line.rstrip().split("\t")
        chrm = a[0]
        start = int(a[1])
        end = int(a[2])
        mid = (start + end) // 2
        length = end - start

        # round length to nearest pool_width
        bins = int(np.round(length / pool_width))
        assert bins > 0
        start = mid - (bins * pool_width) // 2
        start = max(0, start)
        end = start + (bins * pool_width)

        # write
        print("%s\t%d\t%d" % (chrm, start, end), file=full_bed_out)

    full_bed_out.close()

    return full_bed_file


################################################################################
def divide_contigs_chr(contigs, test_chrs, valid_chrs):
    """Divide list of contigs into train/valid/test lists
    by chromosome."""

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start

        if ctg.chr in test_chrs:
            test_contigs.append(ctg)
            test_nt += ctg_len
        elif ctg.chr in valid_chrs:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        else:
            train_contigs.append(ctg)
            train_nt += ctg_len

    total_nt = train_nt + valid_nt + test_nt

    print("Contigs divided into")
    print(
        " Train: %5d contigs, %10d nt (%.4f)"
        % (len(train_contigs), train_nt, train_nt / total_nt)
    )
    print(
        " Valid: %5d contigs, %10d nt (%.4f)"
        % (len(valid_contigs), valid_nt, valid_nt / total_nt)
    )
    print(
        " Test:  %5d contigs, %10d nt (%.4f)"
        % (len(test_contigs), test_nt, test_nt / total_nt)
    )

    return [train_contigs, valid_contigs, test_contigs]


################################################################################
def divide_contigs_folds(contigs, folds):
    """Divide list of contigs into cross fold lists."""

    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed fold nucleotides
    fold_nt_aim = int(np.ceil(total_nt / folds))

    # initialize current fold nucleotides
    fold_nt = np.zeros(folds)

    # initialize fold contig lists
    fold_contigs = []
    for fi in range(folds):
        fold_contigs.append([])

    # process contigs
    for ctg_len, ctg in length_contigs:
        # compute gap between current and aim
        fold_nt_gap = fold_nt_aim - fold_nt
        fold_nt_gap = np.clip(fold_nt_gap, 0, np.inf)

        # compute sample probability
        fold_prob = fold_nt_gap / fold_nt_gap.sum()

        # sample train/valid/test
        fi = np.random.choice(folds, p=fold_prob)
        fold_contigs[fi].append(ctg)
        fold_nt[fi] += ctg_len

    print("Contigs divided into")
    for fi in range(folds):
        print(
            " Fold%d: %5d contigs, %10d nt (%.4f)"
            % (fi, len(fold_contigs[fi]), fold_nt[fi], fold_nt[fi] / total_nt)
        )

    return fold_contigs


################################################################################
def divide_contigs_pct(contigs, test_pct, valid_pct, pct_abstain=0.2):
    """Divide list of contigs into train/valid/test lists,
    aiming for the specified nucleotide percentages."""

    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed train/valid/test nucleotides
    test_nt_aim = test_pct * total_nt
    valid_nt_aim = valid_pct * total_nt
    train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg_len, ctg in length_contigs:
        # compute gap between current and aim
        test_nt_gap = max(0, test_nt_aim - test_nt)
        valid_nt_gap = max(0, valid_nt_aim - valid_nt)
        train_nt_gap = max(1, train_nt_aim - train_nt)

        # skip if too large
        if ctg_len > pct_abstain * test_nt_gap:
            test_nt_gap = 0
        if ctg_len > pct_abstain * valid_nt_gap:
            valid_nt_gap = 0

        # compute remaining %
        gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
        test_pct_gap = test_nt_gap / gap_sum
        valid_pct_gap = valid_nt_gap / gap_sum
        train_pct_gap = train_nt_gap / gap_sum

        # sample train/valid/test
        ri = np.random.choice(
            range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap]
        )[0]
        if ri == 0:
            train_contigs.append(ctg)
            train_nt += ctg_len
        elif ri == 1:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        elif ri == 2:
            test_contigs.append(ctg)
            test_nt += ctg_len
        else:
            print("TVT random number beyond 0,1,2", file=sys.stderr)
            exit(1)

    print("Contigs divided into")
    print(
        " Train: %5d contigs, %10d nt (%.4f)"
        % (len(train_contigs), train_nt, train_nt / total_nt)
    )
    print(
        " Valid: %5d contigs, %10d nt (%.4f)"
        % (len(valid_contigs), valid_nt, valid_nt / total_nt)
    )
    print(
        " Test:  %5d contigs, %10d nt (%.4f)"
        % (len(test_contigs), test_nt, test_nt / total_nt)
    )

    return [train_contigs, valid_contigs, test_contigs]


################################################################################
def limit_contigs(contigs, filter_bed):
    """Limit to contigs overlapping the given BED.

    Args
     contigs: list of Contigs
     filter_bed: BED file to filter by

    Returns:
     fcontigs: list of Contigs
    """

    # print ctgments to BED
    ctg_fd, ctg_bed_file = tempfile.mkstemp()
    ctg_bed_out = open(ctg_bed_file, "w")
    for ctg in contigs:
        print("%s\t%d\t%d" % (ctg.chr, ctg.start, ctg.end), file=ctg_bed_out)
    ctg_bed_out.close()

    # intersect w/ filter_bed
    fcontigs = []
    p = subprocess.Popen(
        "bedtools intersect -a %s -b %s" % (ctg_bed_file, filter_bed),
        shell=True,
        stdout=subprocess.PIPE,
    )
    for line in p.stdout:
        a = line.decode("utf-8").split()
        chrom = a[0]
        ctg_start = int(a[1])
        ctg_end = int(a[2])
        fcontigs.append(data.Contig(0, chrom, ctg_start, ctg_end))

    p.communicate()

    os.close(ctg_fd)
    os.remove(ctg_bed_file)

    return fcontigs


def map_genes_to_sequences(
    mseqs,
    transcriptome,
    gene_data_dir,
    seq_length,
    pool_width,
    coverage_threshold,
    max_genes_limit,
    gene_span=False,
):
    """Map genes to sequences and save mapping metadata.

    This creates a mapping file for each sequence indicating which genes overlap it.
    The actual gene expression values will be read later per-target.

    When a sequence has more genes than max_genes_limit, genes are ranked by
    score = coverage_fraction × centrality, where centrality measures proximity
    to the sequence center (1 at center, 0 at edges).
    """

    # Build interval trees and gene metadata
    gene_trees = transcriptome.gene_trees()
    gene_ids = {gene: gid for gid, gene in transcriptome.genes.items()}
    exon_lengths = {
        gene: sum(e.end - e.begin for e in gene.get_exons())
        for gene in transcriptome.genes.values()
    }

    # Precompute sequence midpoint in bin coordinates
    target_bins = seq_length // pool_width
    seq_midpoint = target_bins / 2

    # Find max genes per sequence and create mapping
    max_genes = 0
    seq_genes_list = []

    for mseq in tqdm(mseqs, desc="Mapping genes to sequences"):
        seq_genes = []
        seq_end = mseq.start + seq_length

        # Query interval tree for candidate genes (O(log n + k))
        for interval in gene_trees[mseq.chr][mseq.start : seq_end]:
            gene = interval.data
            gene_id = gene_ids[gene]
            total_gene_length = exon_lengths[gene]
            # Get gene slice (which bins overlap this gene)
            gene_slice = gene.output_slice(
                seq_start=mseq.start,
                seq_len=seq_length,
                model_stride=pool_width,
                span=gene_span,
                majority_overlap=False,
            )

            if len(gene_slice) == 0:
                continue

            # Compute coverage fraction
            covered_length = len(gene_slice) * pool_width
            coverage_fraction = covered_length / total_gene_length

            if coverage_fraction >= coverage_threshold:
                # Compute centrality: 1 at center, 0 at edges
                gene_midpoint = (gene_slice[0] + gene_slice[-1] + 1) / 2
                centrality = 1 - abs(gene_midpoint - seq_midpoint) / seq_midpoint
                seq_genes.append(
                    {
                        "gene_id": gene_id,
                        "bin_indices": gene_slice,
                        "strand": gene.strand,
                        "coverage_frac": coverage_fraction,
                        "score": coverage_fraction * centrality,
                    }
                )

        # If too many genes, keep best by score (coverage × centrality)
        if len(seq_genes) > max_genes_limit:
            seq_genes.sort(key=lambda x: x["score"], reverse=True)
            seq_genes = seq_genes[:max_genes_limit]

        # Sort by gene_id for deterministic ordering
        seq_genes.sort(key=lambda x: x["gene_id"])
        seq_genes_list.append(seq_genes)
        max_genes = max(max_genes, len(seq_genes))

    # Create zarr store for gene mappings
    num_seqs = len(seq_genes_list)
    zarr_path = f"{gene_data_dir}/gene_mapping.zarr"
    zarr_root = zarr.open(zarr_path, mode="w")

    # Build arrays in memory for batch write
    target_bins = seq_length // pool_width
    all_gene_ids = np.empty((num_seqs, max_genes), dtype="<U50")
    all_gene_presence = np.zeros((num_seqs, max_genes), dtype="bool")
    all_gene_out_mask = np.zeros((num_seqs, max_genes, target_bins), dtype="bool")

    for si, seq_genes in enumerate(seq_genes_list):
        for gi, gene_info in enumerate(seq_genes):
            all_gene_ids[si, gi] = gene_info["gene_id"].split(".")[0]
            all_gene_presence[si, gi] = True
            all_gene_out_mask[si, gi, gene_info["bin_indices"]] = True

    # Create and write arrays in single operations
    zarr_root.create_array(
        "gene_ids",
        data=all_gene_ids,
        chunks=(1, max_genes),
    )
    zarr_root.create_array(
        "gene_presence",
        data=all_gene_presence,
        chunks=(1, max_genes),
    )
    zarr_root.create_array(
        "gene_out_mask",
        data=all_gene_out_mask,
        chunks=(1, max_genes, target_bins),
    )

    return max_genes


if __name__ == "__main__":
    main()
