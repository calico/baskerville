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
import os
import pdb
import sys

import h5py
import intervaltree
import numpy as np
import pandas as pd
import pyBigWig
import scipy.interpolate
from tqdm import tqdm

from baskerville import data

"""
hound_data_read

Read sequence values from coverage files.
"""


################################################################################
# main
################################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Read sequence values from coverage files."
    )
    parser.add_argument(
        "-b",
        dest="blacklist_bed",
        help="Set blacklist nucleotides to a baseline value.",
    )
    parser.add_argument(
        "--black_pct",
        dest="black_pct",
        default=0.5,
        type=float,
        help="Clip blacklisted regions to this distribution value [Default: %(default)s]",
    )
    parser.add_argument(
        "--extreme_pct",
        dest="extreme_pct",
        default=0.9999999,
        type=float,
        help="Clip extreme values to this distribution value [Default: %(default)s]",
    )
    parser.add_argument(
        "--crop",
        dest="crop_bp",
        default=0,
        type=int,
        help="Crop bp off each end [Default: %(default)s]",
    )
    parser.add_argument(
        "-d",
        dest="data_dir",
        required=True,
        help="Data directory",
    )
    parser.add_argument(
        "-i",
        dest="interp_nan",
        default=False,
        action="store_true",
        help="Interpolate NaNs [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        dest="target_index",
        type=int,
        required=True,
        help="Target index [Default: %(default)s]",
    )
    parser.add_argument(
        "-w",
        dest="pool_width",
        default=1,
        type=int,
        help="Average pooling width [Default: %(default)s]",
    )
    args = parser.parse_args()

    if args.crop_bp < 0:
        raise ValueError(f"Crop bp {args.crop_bp} must be non-negative.")

    # read dataset files
    seqs_bed_file = f"{args.data_dir}/sequences.bed"
    seqs_cov_file = f"{args.data_dir}/seqs_cov/{args.target_index}.h5"
    targets_file = f"{args.data_dir}/targets.txt"

    # read target settings
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    target = targets_df.iloc[args.target_index]

    # read model sequences
    model_seqs = []
    for line in open(seqs_bed_file):
        a = line.split()
        model_seqs.append(data.ModelSeq(0, a[0], int(a[1]), int(a[2]), None))

    # read blacklist regions
    black_chr_trees = read_blacklist(args.blacklist_bed)

    # compute dimensions
    num_seqs = len(model_seqs)
    seq_len_nt = model_seqs[0].end - model_seqs[0].start
    seq_len_nt -= 2 * args.crop_bp
    target_length = seq_len_nt // args.pool_width
    if target_length <= 0:
        raise ValueError(
            f"Target length {target_length} must be positive. Check pool width {args.pool_width} and crop bp {args.crop_bp}."
        )

    # collect target coverage
    target_cov = []

    # open genome coverage file
    genome_cov_open = CovFace(target.file)

    # for each model sequence
    for si in tqdm(range(num_seqs)):
        mseq = model_seqs[si]

        # read coverage
        seq_cov_nt = genome_cov_open.read(mseq.chr, mseq.start, mseq.end)
        seq_cov_nt = seq_cov_nt.astype("float32")

        # interpolate NaN
        if args.interp_nan:
            seq_cov_nt = interp_nan(seq_cov_nt)

        # determine baseline coverage
        if target_length >= 8:
            baseline_cov = np.percentile(seq_cov_nt, 100 * args.black_pct)
            baseline_cov = np.nan_to_num(baseline_cov)
        else:
            baseline_cov = 0

        # set blacklist to baseline
        if mseq.chr in black_chr_trees:
            for black_interval in black_chr_trees[mseq.chr][mseq.start : mseq.end]:
                # adjust for sequence indexes
                black_seq_start = black_interval.begin - mseq.start
                black_seq_end = black_interval.end - mseq.start
                black_seq_values = seq_cov_nt[black_seq_start:black_seq_end]
                seq_cov_nt[black_seq_start:black_seq_end] = np.clip(
                    black_seq_values, -baseline_cov, baseline_cov
                )

        # set NaN's to baseline
        if not args.interp_nan:
            nan_mask = np.isnan(seq_cov_nt)
            seq_cov_nt[nan_mask] = baseline_cov

        # crop
        if args.crop_bp > 0:
            seq_cov_nt = seq_cov_nt[args.crop_bp : -args.crop_bp]

        # scale
        seq_cov_nt = target.scale * seq_cov_nt

        # sum pool
        seq_cov = seq_cov_nt.reshape(target_length, args.pool_width)
        if target.sum_stat == "sum":
            seq_cov = seq_cov.sum(axis=1, dtype="float32")
        elif target.sum_stat == "sum_sqrt":
            seq_cov = seq_cov.sum(axis=1, dtype="float32")
            seq_cov = -1 + np.sqrt(1 + seq_cov)
        elif target.sum_stat == "sum_exp75":
            seq_cov = seq_cov.sum(axis=1, dtype="float32")
            seq_cov = -1 + (1 + seq_cov) ** 0.75
        elif target.sum_stat in ["mean", "avg"]:
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
        elif target.sum_stat in ["mean_sqrt", "avg_sqrt"]:
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
            seq_cov = -1 + np.sqrt(1 + seq_cov)
        elif target.sum_stat == "median":
            seq_cov = seq_cov.median(axis=1)
        elif target.sum_stat == "max":
            seq_cov = seq_cov.max(axis=1)
        elif target.sum_stat == "peak":
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
            seq_cov = np.clip(np.sqrt(seq_cov * 4), 0, 1)
        else:
            raise ValueError(f"Unrecognized summary statistic {target.sum_stat}.")

        # clip
        if hasattr(target, "clip_soft") and target.clip_soft is not None:
            clip_mask = seq_cov > target.clip_soft
            seq_cov[clip_mask] = (
                target.clip_soft
                - 1
                + np.sqrt(seq_cov[clip_mask] - target.clip_soft + 1)
            )
        if hasattr(target, "clip") and target["clip"] is not None:
            seq_cov = np.clip(seq_cov, -target["clip"], target["clip"])

        # save
        target_cov.append(seq_cov)

    # clip extreme values
    target_cov = np.array(target_cov)
    extreme_clip = np.quantile(target_cov, args.extreme_pct)
    target_cov = np.clip(target_cov, -extreme_clip, extreme_clip)
    print("Targets sum: %.3f" % target_cov.sum(dtype="float64"))

    # clip to float16
    target_cov = np.clip(target_cov, np.finfo(np.float16).min, np.finfo(np.float16).max)
    target_cov = target_cov.astype("float16")

    # write
    with h5py.File(seqs_cov_file, "w") as seqs_cov_open:
        seqs_cov_open.create_dataset(
            "target", data=target_cov, dtype="float16", compression="gzip"
        )


def interp_nan(x, kind="linear"):
    """Linearly interpolate to fill NaN."""

    # pad zeroes
    xp = np.zeros(len(x) + 2)
    xp[1:-1] = x

    # find NaN
    x_nan = np.isnan(xp)

    if np.sum(x_nan) == 0:
        # unnecessary
        return x

    else:
        # interpolate
        inds = np.arange(len(xp))
        interpolator = scipy.interpolate.interp1d(
            inds[~x_nan], xp[~x_nan], kind=kind, bounds_error=False
        )

        loc = np.where(x_nan)
        xp[loc] = interpolator(loc)

        # slice off pad
        return xp[1:-1]


def read_blacklist(blacklist_bed, black_buffer=20):
    """Construct interval trees of blacklist
    regions for each chromosome."""
    black_chr_trees = {}

    if blacklist_bed is not None and os.path.isfile(blacklist_bed):
        for line in open(blacklist_bed):
            a = line.split()
            chrm = a[0]
            start = max(0, int(a[1]) - black_buffer)
            end = int(a[2]) + black_buffer

            if chrm not in black_chr_trees:
                black_chr_trees[chrm] = intervaltree.IntervalTree()

            black_chr_trees[chrm][start:end] = True

    return black_chr_trees


class CovFace:
    def __init__(self, cov_file):
        self.cov_file = cov_file

        cov_ext = os.path.splitext(self.cov_file)[1].lower()
        if cov_ext == ".gz":
            cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

        if cov_ext in [".bed", ".narrowpeak"]:
            self.preprocess_bed()

        elif cov_ext in [".bw", ".bigwig"]:
            self.preprocess_bigwig()

        elif cov_ext == ".hw":
            self.preprocess_hdwig()

        elif cov_ext in [".h5", ".hdf5", ".w5", ".wdf5"]:
            self.preprocess_wig5()

        else:
            print(
                f'Cannot identify coverage file extension "{cov_ext}".',
                file=sys.stderr,
            )
            exit(1)

    def preprocess_bigwig(self):
        self.chr_cov = {}
        with pyBigWig.open(self.cov_file, "r") as cov_open:
            chrm_lengths = cov_open.chroms()
            for chrm in cov_open.chroms():
                self.chr_cov[chrm] = cov_open.values(
                    chrm, 0, chrm_lengths[chrm], numpy=True
                ).astype("float16")

    def preprocess_bed(self):
        # read BED
        bed_df = pd.read_csv(
            self.cov_file, sep="\t", usecols=range(3), names=["chr", "start", "end"]
        )

        # for each chromosome
        self.chr_cov = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm]

            # find max pos
            pos_max = bed_chr_df.end.max()

            # initialize array
            self.chr_cov[chrm] = np.zeros(pos_max, dtype="bool")

            # set peaks
            for peak in bed_chr_df.itertuples():
                self.chr_cov[peak.chr][peak.start : peak.end] = 1

    def preprocess_hdwig(self):
        # lazy: hdwig is optional, needed only for its own .hw format
        try:
            import hdwig
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{self.cov_file}: reading .hw coverage needs hdwig "
                '(pip install ".[hdwig]"; not yet public)'
            ) from e

        # true values; hdwig divides out the file's storage scale, and picks
        # float16 or, for a track that does not fit it, float32
        with hdwig.open(self.cov_file) as track:
            info = np.finfo(track.dtype)
            self.chr_cov = {
                # handle mysterious inf's; clip passes NaN through untouched
                chrm: np.clip(track.load(chrm), info.min, info.max)
                for chrm in track.contigs
            }

    def preprocess_wig5(self):
        """Read legacy .w5 (one dataset per contig) without hdwig."""
        self.chr_cov = {}
        with h5py.File(self.cov_file, "r") as cov_open:
            # factor applied at write time purely to fit float16's range; it
            # carries no meaning, so divide it out (in the stored dtype, as
            # hdwig does for a legacy file, which declares no maximum).
            file_scale = float(cov_open.attrs.get("w5_scale", 1.0))
            for chrm in cov_open.keys():
                cov = cov_open[chrm][:]
                if file_scale != 1.0:
                    cov = np.divide(cov, file_scale, dtype=cov.dtype)
                # handle mysterious inf's; clip passes NaN through untouched
                info = np.finfo(cov.dtype)
                self.chr_cov[chrm] = np.clip(cov, info.min, info.max)

    def read(self, chrm, start, end):
        if chrm in self.chr_cov:
            # read coverage
            cov = self.chr_cov[chrm][start:end]

            # pad
            pad_zeros = end - start - len(cov)
            if pad_zeros > 0:
                cov_pad = np.zeros(pad_zeros, dtype="bool")
                cov = np.concatenate([cov, cov_pad])

        else:
            print(
                "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros."
                % (self.cov_file, chrm, start, end),
                file=sys.stderr,
            )
            cov = np.zeros(end - start, dtype="float16")

        return cov


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
