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
import os
import tempfile
import shutil

from baskerville.snps import parse_mix_dtype, score_snps, score_gene_snps
from baskerville.helpers.gcs_utils import (
    upload_folder_gcs,
    download_rename_inputs,
)

"""
hound_snp

Compute variant effect predictions for SNPs in a VCF file.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Compute variant effect predictions for SNPs in a VCF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        dest="cluster_pct",
        default=0,
        type=float,
        help="Cluster SNPs/genes within a %% of the seq length to make a single ref pred",
    )
    parser.add_argument("-f", dest="genome_fasta", required=True, help="Genome FASTA")
    parser.add_argument(
        "-g",
        dest="genes_gtf",
        default=None,
        help="GTF for gene annotations. Enables gene scoring (coverage + gene head logFC).",
    )
    parser.add_argument(
        "--center_gene",
        dest="center_gene",
        default=False,
        action="store_true",
        help="Center sequences on genes instead of variants (requires -g)",
    )
    parser.add_argument(
        "--gcs",
        dest="gcs",
        default=False,
        action="store_true",
        help="Input and output are in gcs",
    )
    parser.add_argument(
        "--head",
        dest="head",
        default=0,
        type=int,
        help="Model head with which to predict.",
    )
    parser.add_argument(
        "--index_end",
        dest="index_end",
        default=None,
        type=int,
        help="End index for SNP processing (0-based, exclusive)",
    )
    parser.add_argument(
        "--index_start",
        dest="index_start",
        default=None,
        type=int,
        help="Start index for SNP processing (0-based, inclusive)",
    )
    parser.add_argument(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts",
    )
    parser.add_argument(
        "--local_window",
        default=2048,
        type=int,
        help="Local window size in bp for targets with window=local",
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
        "--compile",
        default=False,
        action="store_true",
        help="Compile the model with torch.compile",
    )
    parser.add_argument(
        "-n",
        "--norm",
        dest="norm_file",
        default=None,
        help="HDF5 file to use for normalization quantiles instead of current variant set",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        default="snp_out",
        help="Output directory for tables and plots",
    )
    parser.add_argument(
        "--pregrouped_seqs",
        default=False,
        action="store_true",
        help="Use pre-grouped sequences for scoring, defined in the VCF",
    )
    parser.add_argument(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions",
    )
    parser.add_argument(
        "--shifts",
        dest="shifts",
        default="0",
        type=str,
        help="Ensemble prediction shifts",
    )
    parser.add_argument(
        "--gene_cov_t",
        type=float,
        default=0.5,
        help="Minimum gene coverage fraction to include [Default: %(default)s]",
    )
    parser.add_argument(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="In gene scoring mode, aggregate entire gene span",
    )
    parser.add_argument(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save.",
    )
    parser.add_argument(
        "-t",
        dest="targets_file",
        required=True,
        type=str,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--targets_gene",
        dest="targets_gene_file",
        default=None,
        type=str,
        help="Gene targets file (for gene/ scores). Required when using gene/ stats.",
    )
    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("model_file", help="Model file")
    parser.add_argument("vcf_file", help="VCF file")

    args = parser.parse_args()

    # validate flag dependencies
    if args.pregrouped_seqs and args.genes_gtf is None:
        parser.error("--pregrouped_seqs requires -g/--genes_gtf")
    if args.center_gene and args.genes_gtf is None:
        parser.error("--center_gene requires -g/--genes_gtf")

    # create output directory (if output is local)
    if not args.gcs:
        os.makedirs(args.out_dir, exist_ok=True)

    else:
        # assume that output_dir will be gcs
        gcs_output_dir = args.out_dir
        temp_dir = tempfile.mkdtemp()
        args.out_dir = temp_dir + "/output_dir"
        os.makedirs(args.out_dir, exist_ok=True)

        # download input files from gcs to a local file
        args.params_file = download_rename_inputs(args.params_file, temp_dir)
        args.vcf_file = download_rename_inputs(args.vcf_file, temp_dir)
        args.model_file = download_rename_inputs(args.model_file, temp_dir)

        args.genome_fasta = download_rename_inputs(args.genome_fasta, temp_dir)
        if args.genes_gtf is not None:
            args.genes_gtf = download_rename_inputs(args.genes_gtf, temp_dir)
        if args.targets_file is not None:
            args.targets_file = download_rename_inputs(args.targets_file, temp_dir)

    # parse options
    args.mix_dtype = parse_mix_dtype(args.mix_dtype)
    args.shifts = [int(shift) for shift in args.shifts.split(",")]
    args.snp_stats = args.snp_stats.split(",")

    # calculate SNP scores
    if args.genes_gtf is not None and (args.center_gene or args.pregrouped_seqs):
        score_gene_snps(args)
    else:
        score_snps(args)

    if args.gcs:
        # synchronize
        upload_folder_gcs(args.out_dir, gcs_output_dir)
        # clean up temp
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
