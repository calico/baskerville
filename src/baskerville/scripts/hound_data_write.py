#!/usr/bin/env python
# Copyright 2019 Calico Life Sciences LLC

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
import glob
import os
import time

import h5py
from natsort import natsorted
import numpy as np
import pdb
import pysam
from tqdm import tqdm
import zarr

from baskerville import data
from baskerville import dna

"""
hound_data_write

Write compressed sequence/target examples.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Write compressed sequence/target examples."
    )
    parser.add_argument(
        "--cov_dir",
        dest="cov_dir",
        default=None,
        help="Directory with sequence coverage files [Default: %(default)s]",
    )
    parser.add_argument(
        "-d",
        dest="decimals",
        default=None,
        type=int,
        help="Round values to given decimals [Default: %(default)s]",
    )
    parser.add_argument(
        "-f",
        dest="fold",
        default=None,
        help="Fold label [Default: %(default)s]",
    )
    parser.add_argument(
        "--gene_dir",
        dest="gene_dir",
        default=None,
        help="Directory with gene data files [Default: %(default)s]",
    )
    parser.add_argument(
        "-j",
        dest="job_i",
        default=None,
        type=int,
        help="Job index [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        dest="start_i",
        default=0,
        type=int,
        help="Sequence start index [Default: %(default)s]",
    )
    parser.add_argument(
        "-e",
        dest="end_i",
        default=None,
        type=int,
        help="Sequence end index [Default: %(default)s]",
    )
    parser.add_argument(
        "--te",
        dest="target_extend",
        default=None,
        type=int,
        help="Extend targets vector [Default: %(default)s]",
    )
    parser.add_argument("-u", dest="umap_npy", help="Unmappable array numpy file")
    parser.add_argument(
        "--umap_clip",
        dest="umap_clip",
        default=1,
        type=float,
        help="Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %(default)s]",
    )
    parser.add_argument(
        "-x",
        dest="extend_bp",
        default=0,
        type=int,
        help="Extend sequences on each side [Default: %(default)s]",
    )
    parser.add_argument("fasta_file", help="FASTA file")
    parser.add_argument("seqs_bed_file", help="BED file with sequences")
    parser.add_argument("zarr_file", help="Output Zarr file")
    args = parser.parse_args()

    ################################################################
    # read model sequences

    model_seqs = []
    for line in open(args.seqs_bed_file):
        a = line.split()
        model_seqs.append(data.ModelSeq(0, a[0], int(a[1]), int(a[2]), a[3]))

    if args.end_i is None:
        args.end_i = len(model_seqs)
    num_seqs = args.end_i - args.start_i

    fold_set_indexes = [
        i for i in range(len(model_seqs)) if model_seqs[i].label == args.fold
    ]
    write_start = args.start_i - fold_set_indexes[0]

    # validate at least one data type exists
    has_coverage = args.cov_dir is not None
    has_gene = args.gene_dir is not None and os.path.exists(args.gene_dir)
    if not has_coverage and not has_gene:
        raise ValueError(
            "Must provide either coverage targets (--cov_dir) or gene data (--gene_dir) or both"
        )

    # load unmappable mask
    unmap_mask = None
    if args.umap_npy is not None:
        unmap_mask = np.load(args.umap_npy)

    ################################################################
    # read coverage targets

    targets = None
    if has_coverage:
        print("Reading coverage targets...")

        # Read coverage target files
        seqs_cov_files = (
            natsorted(glob.glob(f"{args.cov_dir}/*.h5")) if args.cov_dir else []
        )
        num_targets = len(seqs_cov_files)

        # determine sequence pool length
        with h5py.File(seqs_cov_files[0], "r") as seqs_cov_open:
            tk = list(seqs_cov_open.keys())[0]
            seq_pool_len = seqs_cov_open[tk].shape[1]

        # initialize targets
        targets = np.zeros((num_seqs, num_targets, seq_pool_len), dtype="float16")

        # read each target
        print("Reading targets...")
        for ti in tqdm(range(num_targets)):
            with h5py.File(seqs_cov_files[ti], "r") as seqs_cov_open:
                tk = list(seqs_cov_open.keys())[0]
                targets[:, ti, :] = seqs_cov_open[tk][args.start_i : args.end_i, :]

        # modify unmappable
        if unmap_mask is not None and args.umap_clip < 1:
            for si in range(num_seqs):
                msi = args.start_i + si

                # determine unmappable null value
                seq_target_clip = np.quantile(targets[si], q=args.umap_clip, axis=1)
                seq_target_clip = seq_target_clip[np.newaxis, :]

                # set unmappable positions to null
                targets[si, :, unmap_mask[msi]] = np.minimum(
                    targets[si, :, unmap_mask[msi]], seq_target_clip
                )

        # truncate decimals (which aids compression)
        if args.decimals is not None:
            for si in range(num_seqs):
                targets_si = targets[si].astype("float32")
                targets_si = np.around(targets_si, decimals=args.decimals)
                targets[si] = targets_si.astype("float16")

    ################################################################
    # read gene data

    gene_targets = None
    gene_presence = None
    gene_out_mask = None
    gene_ids = None

    if has_gene:
        print("Reading gene targets...")

        # Determine number of gene targets from H5 files
        gene_target_files = natsorted(glob.glob(f"{args.gene_dir}/*.h5"))
        num_gene_targets = len(gene_target_files)

        # Read gene values for each gene target
        gene_targets_per_sample = []
        for gene_file in gene_target_files:
            with h5py.File(gene_file, "r") as h5f:
                gene_values = h5f["target"][
                    args.start_i : args.end_i
                ]  # [num_seqs, max_genes]
                gene_targets_per_sample.append(gene_values)

        # Stack to get [num_seqs, num_gene_targets, max_genes]
        gene_targets = np.stack(gene_targets_per_sample, axis=1).astype("float16")

        # Read gene mask, slices, and IDs from zarr store
        gene_mapping = zarr.open(f"{args.gene_dir}/gene_mapping.zarr", mode="r")
        gene_presence = gene_mapping["gene_presence"][args.start_i : args.end_i]
        gene_out_mask = gene_mapping["gene_out_mask"][args.start_i : args.end_i]
        gene_ids = gene_mapping["gene_ids"][args.start_i : args.end_i]

    ################################################################
    # read sequences

    # open FASTA
    fasta_open = pysam.Fastafile(args.fasta_file)

    sequences = []
    unmaps = []

    print("Reading sequences...")
    for si in tqdm(range(num_seqs)):
        msi = args.start_i + si
        mseq = model_seqs[msi]
        mseq_start = mseq.start - args.extend_bp
        mseq_end = mseq.end + args.extend_bp

        # read FASTA
        seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)

        # one hot code
        seq_1hot = dna.dna_1hot_index(seq_dna)

        # save
        sequences.append(seq_1hot)
        if unmap_mask is not None:
            unmaps.append(unmap_mask[msi])

    fasta_open.close()

    ################################################################
    # # write
    t0 = time.time()
    print("Writing...")
    write_end = write_start + num_seqs
    zarr_open = zarr.open(args.zarr_file, mode="a")
    zarr_open["sequence"][write_start:write_end] = sequences

    # Write coverage targets if available
    if has_coverage:
        zarr_open["target"][write_start:write_end] = targets

    # Write gene data if available
    if has_gene:
        zarr_open["gene_target"][write_start:write_end] = gene_targets
        zarr_open["gene_presence"][write_start:write_end] = gene_presence
        zarr_open["gene_out_mask"][write_start:write_end] = gene_out_mask
        zarr_open["gene_ids"][write_start:write_end] = gene_ids
    print(f"Done in {time.time() - t0:.1f} sec")

    # write to file indicating successful completion
    success_file = (
        f"{os.path.split(args.zarr_file)[0]}/{args.fold}-{args.job_i}-success.txt"
    )

    with open(success_file, "wt") as f:
        f.write("success\n")


def fetch_dna(fasta_open, chrm, start, end):
    """Fetch DNA when start/end may reach beyond chromosomes."""

    # initialize sequence
    seq_len = end - start
    seq_dna = ""

    # add N's for left over reach
    if start < 0:
        seq_dna = "N" * (-start)
        start = 0

    # get dna
    seq_dna += fasta_open.fetch(chrm, start, end)

    # add N's for right over reach
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    return seq_dna


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
