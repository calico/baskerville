import concurrent
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from tqdm import tqdm
import torch

from baskerville import dna
from baskerville import dataset
from baskerville.gene import Transcriptome, find_overlapping_genes
from baskerville import seqnn
from baskerville.vcf import VCF, SNPCluster

GENE_ASSAYS = {"RNA", "RNA3", "CAGE"}

MIX_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_mix_dtype(name: str) -> torch.dtype:
    """Map a --mix_dtype name to a torch dtype, warning on reduced precision."""
    if name != "float32":
        print(
            f"WARNING: --mix_dtype {name} adds rounding noise that can dominate "
            "variant effect scores (small alt - ref differences). Use float32 "
            "unless validated for your model and stats.",
            file=sys.stderr,
        )
    return MIX_DTYPES[name]


def gene_track_mask(targets_df) -> np.ndarray:
    """Boolean mask over targets selecting gene-appropriate tracks.

    Uses the ``gene`` column if present (truthy = include); otherwise falls
    back to matching ``assay`` (case-insensitively) against :data:`GENE_ASSAYS`.
    Raises ``ValueError`` if neither column is present.
    """
    if "gene" in targets_df.columns:
        return targets_df["gene"].astype(bool).values
    if "assay" in targets_df.columns:
        return targets_df["assay"].str.upper().isin(GENE_ASSAYS).values
    raise ValueError(
        "Targets file must have a 'gene' or 'assay' column to compute "
        "covgene/ stats. Add a binary 'gene' column to opt tracks in/out, "
        f"or an 'assay' column (gene assays: {sorted(GENE_ASSAYS)})."
    )


def gene_targets(targets_df):
    """Subset targets to gene tracks; strand pairs must be both in or both out."""
    mask = gene_track_mask(targets_df)
    if "strand_pair" in targets_df.columns:
        if (mask != mask[dataset.strand_pair_indices(targets_df)]).any():
            raise ValueError(
                "gene tracks must include both strands of each strand pair"
            )
    return targets_df[mask]


def partition_snp_stats(snp_stats):
    """Return cov, covgene, and gene stat lists; bare names default to cov/."""
    cov_stats, covgene_stats, gene_stats = [], [], []
    for stat in snp_stats:
        if stat.startswith("covgene/"):
            covgene_stats.append(stat)
        elif stat.startswith("gene/"):
            gene_stats.append(stat)
        elif stat.startswith("cov/"):
            cov_stats.append(stat)
        else:
            cov_stats.append(f"cov/{stat}")
    return cov_stats, covgene_stats, gene_stats


def compute_score_quantiles(scores, quantile_thresholds, quantiles):
    """Compute quantile positions for scores.

    Args:
        scores: Array of scores (can be 1D or 2D)
        quantile_thresholds: Array of quantile threshold values for each target
        quantiles: Array of quantile positions (e.g., 0.001, 0.002, ..., 0.999)

    Returns:
        Array of quantile positions corresponding to input scores
    """
    if len(scores.shape) == 1:
        # Single target case
        # Find the quantile bin that each score falls into
        quantile_indices = np.searchsorted(quantile_thresholds, scores, side="right")
        # Handle edge case: scores higher than max threshold get max quantile
        quantile_indices = np.where(
            quantile_indices >= len(quantiles), len(quantiles) - 1, quantile_indices
        )
        # Handle edge case: scores lower than min threshold get min quantile
        quantile_indices = np.maximum(quantile_indices, 0)
        return quantiles[quantile_indices]
    else:
        # Multiple targets case
        result = np.zeros_like(scores)
        for target_idx in range(scores.shape[1]):
            target_scores = scores[:, target_idx]
            target_thresholds = quantile_thresholds[target_idx, :]
            # Find the quantile bin that each score falls into
            quantile_indices = np.searchsorted(
                target_thresholds, target_scores, side="right"
            )
            # Handle edge case: scores higher than max threshold get max quantile
            quantile_indices = np.where(
                quantile_indices >= len(quantiles), len(quantiles) - 1, quantile_indices
            )
            # Handle edge case: scores lower than min threshold get min quantile
            quantile_indices = np.maximum(quantile_indices, 0)
            result[:, target_idx] = quantiles[quantile_indices]
        return result


def score_snps(args):
    """Score SNPs in a VCF file with a SeqNN model.

    Variant-centered sequences. Without a GTF, computes gene-agnostic coverage
    stats. With a GTF, also finds overlapping genes and computes gene scores
    (coverage-based + gene head logFC).

    Args:
        args: Arguments from hound_snp.
    """

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # read targets
    targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)

    # gene-level scoring
    score_genes = args.genes_gtf is not None

    cov_stats, covgene_stats, gene_stats = partition_snp_stats(args.snp_stats)

    # covgene/ only: predict gene tracks only
    if score_genes and not cov_stats:
        targets_df = gene_targets(targets_df)

    # per-target local window mask
    if "window" in targets_df.columns:
        local_mask = (targets_df["window"] == "local").values
        if not local_mask.any():
            local_mask = None
    else:
        local_mask = None

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

    #################################################################
    # setup model

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model, output_slice=targets_df.index)
    seqnn_model.restore(args.model_file)
    seqnn_model.ensemble_rc = args.rc
    seqnn_model.mix_dtype = args.mix_dtype
    seqnn_model.model.eval()
    seqnn_model.model_di = params.get("train", {}).get("model_di", None)

    # shift outside seqnn
    num_shifts = len(args.shifts)
    output_length = seqnn_model.output_length()
    output_stride = seqnn_model.output_stride()
    output_crop = seqnn_model.output_crop_bp()

    # local window in bins
    window_bins = args.local_window // output_stride if local_mask is not None else None

    # gene head depth
    gene_depth = (
        seqnn_model.output_depth(args.head, head_type="gene") if score_genes else 0
    )
    has_gene_head = gene_depth > 0

    # construct strand masks (gene mode)
    if score_genes and "strand_pair" in targets_df.columns:
        plus_mask = (targets_df.strand != "-").values
        plus_mask = torch.tensor(plus_mask, dtype=torch.bool, device=seqnn_model.device)
        minus_mask = (targets_df.strand != "+").values
        minus_mask = torch.tensor(
            minus_mask, dtype=torch.bool, device=seqnn_model.device
        )
    else:
        plus_mask = minus_mask = None

    # gene-track masks for covgene/ slicing
    if score_genes:
        gene_mask_np = gene_track_mask(targets_df)
        gene_mask = torch.tensor(
            gene_mask_np, dtype=torch.bool, device=seqnn_model.device
        )
        if plus_mask is not None:
            gene_plus_mask = plus_mask & gene_mask
            gene_minus_mask = minus_mask & gene_mask
        else:
            gene_plus_mask = gene_minus_mask = gene_mask
        gene_mask_strand = gene_track_mask(targets_strand_df)
    else:
        gene_plus_mask = gene_minus_mask = None
        gene_mask_strand = None

    #################################################################
    # load SNPs

    # clustering SNPs requires sorted VCF and no reference flips
    snps_clustered = args.cluster_pct > 0

    # filter for worker SNPs
    if hasattr(args, "index_start") and args.index_start is not None:
        start_i = args.index_start
        end_i = args.index_end
    else:
        start_i = None
        end_i = None

    # read SNPs
    vcf_obj = VCF(
        args.vcf_file,
        require_sorted=snps_clustered,
        flip_ref=~snps_clustered,
        validate_ref_fasta=args.genome_fasta,
        start_i=start_i,
        end_i=end_i,
    )
    snps = vcf_obj.snps

    # cluster SNPs
    if snps_clustered:
        snp_clusters = cluster_snps(snps, params_model["seq_length"], args.cluster_pct)
    else:
        snp_clusters = []
        for snp in snps:
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)

    # delimit sequence boundaries
    [sc.delimit(params_model["seq_length"]) for sc in snp_clusters]

    #################################################################
    # load genes (if GTF provided)

    if score_genes:
        transcriptome = Transcriptome(args.genes_gtf)
        gene_trees = transcriptome.gene_trees()

        # convert SNP clusters to GeneSNPClusters with overlapping genes
        gene_cov_t = getattr(args, "gene_cov_t", 0)
        genesnp_clusters = []
        sc_to_gsc = {}
        for ci, sc in enumerate(snp_clusters):
            pstart = sc.start + output_crop
            pend = sc.end - output_crop
            pred_length = pend - pstart
            genes = find_overlapping_genes(gene_trees, sc.chr, pstart, pend)
            # filter by coverage fraction
            if gene_cov_t > 0:
                genes = [
                    g
                    for g in genes
                    if g.coverage_fraction(
                        pstart,
                        pred_length,
                        output_stride,
                        span=getattr(args, "span", False),
                    )
                    >= gene_cov_t
                ]
            if genes:
                gsc = GeneSNPCluster()
                gsc.chr = sc.chr
                gsc.start = sc.start
                gsc.end = sc.end
                gsc.pstart = pstart
                gsc.pend = pend
                for gene in genes:
                    gsc.add_gene(gene)
                for snp in sc.snps:
                    gsc.add_snp(snp)
                genesnp_clusters.append(gsc)
                sc_to_gsc[ci] = gsc

    # open genome FASTA
    genome_open = pysam.Fastafile(args.genome_fasta)

    #################################################################
    # predict SNP scores, write output

    # setup output
    targets_out_df = targets_strand_df.reset_index(drop=True)
    targets_out_df.to_csv(f"{args.out_dir}/targets_cov.txt", sep="\t")

    # validate: gene/ stats require gene head
    if gene_stats and not has_gene_head:
        print(
            f"WARNING: gene/ stats {gene_stats} requested but model has no gene head. Ignoring."
        )
        gene_stats = []

    # write gene targets table
    if gene_stats and getattr(args, "targets_gene_file", None):
        targets_gene_df = pd.read_csv(args.targets_gene_file, sep="\t", index_col=0)
        targets_gene_df.to_csv(f"{args.out_dir}/targets_gene.txt", sep="\t")

    # write covgene targets table (gene-track subset of strand-collapsed targets)
    if covgene_stats and score_genes:
        targets_out_df[gene_mask_strand].to_csv(
            f"{args.out_dir}/targets_covgene.txt", sep="\t"
        )

    # HDF5 keys for pair-indexed stats (already prefixed)
    pair_h5_keys = covgene_stats + gene_stats

    scores_out = initialize_output_h5(
        args.out_dir,
        snps,
        output_length,
        targets_out_df.shape[0],
        num_shifts,
        cov_stats=cov_stats,
        covgene_stats=covgene_stats if score_genes else (),
        gene_stats=gene_stats if score_genes else (),
        genesnp_clusters=genesnp_clusters if score_genes else None,
        gene_depth=gene_depth,
        raw_depth=targets_df.shape[0],
        covgene_depth=int(gene_mask_strand.sum()) if score_genes else None,
    )

    # write to HDF5 in CPU thread
    def hdf5_write_cov(scores, si):
        for cov_stat in cov_stats:
            scores_out[cov_stat][si] = scores[cov_stat]

    def hdf5_write_pair(cov_scores, gene_head_scores, score_index):
        for stat in covgene_stats:
            scores_out[stat][score_index] = cov_scores[stat]
        for stat in gene_stats:
            scores_out[stat][score_index] = gene_head_scores[stat]

    # futures for cov writes; drained after the loop so a write failure
    # (e.g. a shape mismatch) raises instead of silently leaving zeros
    cov_write_futures = []

    # SNP-gene pair index
    sgi = 0

    # SNP index
    si = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with (
            torch.no_grad(),
            torch.autocast(device_type=seqnn_model.device, dtype=args.mix_dtype),
        ):
            # initialize 1 hot encoding
            sc0 = snp_clusters[0]
            s1l = executor.submit(sc0.get_1hots, genome_open)

            for ci, sc in enumerate(
                tqdm(snp_clusters, unit=" cluster", desc="SNP clusters")
            ):
                # pull latest 1 hot encoding
                snp_1hot_list = s1l.result()
                ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)
                ref_1hot = torch.tensor(
                    ref_1hot, device=seqnn_model.device, dtype=torch.float32
                )

                # submit next 1 hot encoding
                if ci + 1 < len(snp_clusters):
                    sc1 = snp_clusters[ci + 1]
                    s1l = executor.submit(sc1.get_1hots, genome_open)

                # build gene masks if this cluster has genes
                gsc = sc_to_gsc.get(ci) if score_genes else None
                if gsc is not None and (covgene_stats or gene_stats):
                    gom, gp = build_gene_masks(
                        gsc.genes,
                        gsc.pstart,
                        gsc.pend - gsc.pstart,
                        output_stride,
                        output_length,
                        seqnn_model.device,
                        span=getattr(args, "span", False),
                    )
                else:
                    gom = gp = None

                # only pass masks to model when gene head exists
                model_gom = gom if has_gene_head else None
                model_gp = gp if has_gene_head else None

                # predict reference
                ref_cov_preds = []
                ref_gene_preds = []
                for shift in args.shifts:
                    ref_1hot_shift = dna.torch_shift(ref_1hot, shift)
                    output = seqnn_model(
                        ref_1hot_shift,
                        args.head,
                        gene_out_mask=model_gom,
                        gene_presence=model_gp,
                    )
                    ref_cov_shift = seqnn_model.untransform_fn(
                        output.coverage.squeeze(0), targets_df
                    )
                    ref_cov_preds.append(ref_cov_shift)
                    if model_gom is not None and output.has_gene:
                        ref_gene_preds.append(output.gene.squeeze(0))

                for ai, alt_1hot in enumerate(snp_1hot_list[1:]):
                    alt_1hot = np.expand_dims(alt_1hot, axis=0)
                    alt_1hot = torch.tensor(
                        alt_1hot, device=seqnn_model.device, dtype=torch.float32
                    )

                    # variant bin position
                    indel_size = sc.snps[ai].indel_size()
                    snp_seq_pos = sc.snps[ai].pos - sc.start - output_crop
                    snp_seq_bin = snp_seq_pos // output_stride

                    # add left/right shifts for indels
                    if indel_size == 0:
                        alt_shifts = args.shifts
                    else:
                        alt_shifts = []
                        for shift in args.shifts:
                            alt_shifts.append(shift)
                            alt_shifts.append(shift - indel_size)

                    # predict alternate
                    alt_cov_preds = []
                    alt_gene_preds = []
                    for shift in alt_shifts:
                        alt_1hot_shift = dna.torch_shift(alt_1hot, shift)
                        output = seqnn_model(
                            alt_1hot_shift,
                            args.head,
                            gene_out_mask=model_gom,
                            gene_presence=model_gp,
                        )
                        alt_cov_shift = seqnn_model.untransform_fn(
                            output.coverage.squeeze(0), targets_df
                        )
                        alt_cov_preds.append(alt_cov_shift)
                        if model_gom is not None and output.has_gene:
                            alt_gene_preds.append(output.gene.squeeze(0))

                    # stitch indel shifts (coverage only)
                    if indel_size != 0 and args.indel_stitch:
                        alt_cov_preds = stitch_preds(
                            alt_cov_preds, args.shifts, snp_seq_bin
                        )

                    # flip reference and alternate
                    if snps[si].flipped:
                        rp_cov = torch.stack(alt_cov_preds)
                        ap_cov = torch.stack(ref_cov_preds)
                        if ref_gene_preds:
                            rp_gene = torch.stack(alt_gene_preds)
                            ap_gene = torch.stack(ref_gene_preds)
                    else:
                        rp_cov = torch.stack(ref_cov_preds)
                        ap_cov = torch.stack(alt_cov_preds)
                        if ref_gene_preds:
                            rp_gene = torch.stack(ref_gene_preds)
                            ap_gene = torch.stack(alt_gene_preds)

                    # repeat reference predictions for indels w/o stitching
                    if indel_size != 0 and not args.indel_stitch:
                        rp_cov = rp_cov.repeat_interleave(2, dim=0)
                        if ref_gene_preds:
                            rp_gene = rp_gene.repeat_interleave(2, dim=0)

                    # full-sequence coverage scores (always)
                    if cov_stats:
                        scores = compute_scores_cov(
                            rp_cov,
                            ap_cov,
                            cov_stats,
                            strand_transform,
                            local_mask=local_mask,
                            snp_seq_bin=snp_seq_bin,
                            window_bins=window_bins,
                        )
                        cov_write_futures.append(
                            executor.submit(hdf5_write_cov, scores, si)
                        )

                    # gene-sliced scores (when GTF provided and SNP overlaps genes)
                    if gsc is not None and (covgene_stats or gene_stats):
                        for gi, gene in enumerate(gsc.genes):
                            gene_mask = gom[0, gi]
                            if not gene_mask.any():
                                print(
                                    f"WARNING: {gene.kv['gene_id']} exons fall outside prediction boundaries."
                                )
                            else:
                                rp_cov_gene = rp_cov[:, :, gene_mask]
                                ap_cov_gene = ap_cov[:, :, gene_mask]

                                # filter by strand AND gene-track membership
                                if gene.strand == "-":
                                    track_mask = gene_minus_mask
                                else:
                                    track_mask = gene_plus_mask
                                rp_cov_gene = rp_cov_gene[:, track_mask, :]
                                ap_cov_gene = ap_cov_gene[:, track_mask, :]

                                # coverage-based gene-sliced scores
                                cov_scores = compute_scores_cov(
                                    rp_cov_gene, ap_cov_gene, covgene_stats
                                )

                                # gene head scores
                                gene_head_scores = {}
                                if gene_stats and ref_gene_preds:
                                    gene_head_scores = compute_scores_gene(
                                        rp_gene[:, :, gi],
                                        ap_gene[:, :, gi],
                                        gene_stats,
                                    )

                                executor.submit(
                                    hdf5_write_pair, cov_scores, gene_head_scores, sgi
                                )

                            sgi += 1

                    del ap_cov, alt_cov_preds
                    si += 1

                # clean reference
                del rp_cov, ref_cov_preds
                torch.cuda.empty_cache()

    # surface any cov-write exceptions (executor swallows them otherwise)
    for f in cov_write_futures:
        f.result()

    # close genome
    genome_open.close()

    # verify gene index count
    if score_genes and (covgene_stats or gene_stats):
        out_snpgene_num = len(scores_out["snp_idx"])
        assert sgi == out_snpgene_num, (
            f"SNP-gene index {sgi} does not match expected {out_snpgene_num}"
        )

    # compute quantiles
    if cov_stats:
        write_quantiles(scores_out, cov_stats, args.norm_file)
    if score_genes and pair_h5_keys:
        write_quantiles(scores_out, pair_h5_keys, args.norm_file)

    # Mark job as completed before closing
    del scores_out["progress_status"]
    scores_out.create_dataset("progress_status", data="completed".encode("utf-8"))
    scores_out.close()


def score_gene_snps(args):
    """Score SNPs with gene-centered sequences.

    Centers sequences on gene clusters. Computes coverage-based gene scores
    and optionally gene head scores (logFC_cov, logFC_gene).

    Args:
        args: Arguments from hound_snp.
    """

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # read targets; gene-centered mode scores gene tracks only
    targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)
    targets_df = gene_targets(targets_df)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        params_model["strand_pair"] = dataset.strand_pair_indices(targets_df)
    else:
        targets_strand_df = targets_df

    #################################################################
    # setup model

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model, output_slice=targets_df.index)
    seqnn_model.restore(args.model_file)
    seqnn_model.ensemble_rc = args.rc
    seqnn_model.mix_dtype = args.mix_dtype
    seqnn_model.model.eval()
    seqnn_model.model_di = params.get("train", {}).get("model_di", None)

    # shift outside seqnn
    num_shifts = len(args.shifts)
    output_length = seqnn_model.output_length()
    output_stride = seqnn_model.output_stride()
    output_crop = seqnn_model.output_crop_bp()

    # gene head depth (0 if no gene head)
    gene_depth = seqnn_model.output_depth(args.head, head_type="gene")
    has_gene_head = gene_depth > 0

    # construct strand masks
    plus_mask = (targets_df.strand != "-").values
    plus_mask = torch.tensor(plus_mask, dtype=torch.bool, device=seqnn_model.device)
    minus_mask = (targets_df.strand != "+").values
    minus_mask = torch.tensor(minus_mask, dtype=torch.bool, device=seqnn_model.device)

    #################################################################
    # load SNPs

    # filter for worker SNPs
    if hasattr(args, "index_start") and args.index_start is not None:
        start_i = args.index_start
        end_i = args.index_end
    else:
        start_i = None
        end_i = None

    # read SNPs
    vcf_obj = VCF(
        args.vcf_file,
        require_sorted=True,
        flip_ref=False,
        validate_ref_fasta=args.genome_fasta,
        start_i=start_i,
        end_i=end_i,
        pregrouped_seqs=args.pregrouped_seqs,
    )
    snps = vcf_obj.snps

    # read genes
    transcriptome = Transcriptome(args.genes_gtf)

    if args.pregrouped_seqs:
        if output_crop != 0:
            raise NotImplementedError(
                "Pregrouped sequences are currently not supported with output cropping."
            )
        genesnp_clusters = construct_pregrouped_geneclusters(transcriptome, snps)

    else:
        # cluster genes
        genesnp_clusters = cluster_genes(
            transcriptome, params_model["seq_length"], args.cluster_pct
        )

        # delimit sequence boundaries
        [
            gsc.delimit(params_model["seq_length"], output_crop)
            for gsc in genesnp_clusters
        ]

        # assign SNPs to genes
        map_snps_genes(snps, genesnp_clusters)

        # remove genes w/o SNPs
        genesnp_clusters = [gsc for gsc in genesnp_clusters if len(gsc.snps) > 0]

    # open genome FASTA
    genome_open = pysam.Fastafile(args.genome_fasta)

    #################################################################
    # predict SNP scores, write output

    # setup output
    targets_out_df = targets_strand_df.reset_index(drop=True)
    targets_out_df.to_csv(f"{args.out_dir}/targets_cov.txt", sep="\t")

    # Full-sequence coverage stats are ignored on gene-centered sequences.
    _, covgene_stats, gene_stats = partition_snp_stats(args.snp_stats)

    # validate: gene/ stats require gene head
    if gene_stats and not has_gene_head:
        print(
            f"WARNING: gene/ stats {gene_stats} requested but model has no gene head. Ignoring."
        )
        gene_stats = []

    # write gene targets table
    if gene_stats and getattr(args, "targets_gene_file", None):
        targets_gene_df = pd.read_csv(args.targets_gene_file, sep="\t", index_col=0)
        targets_gene_df.to_csv(f"{args.out_dir}/targets_gene.txt", sep="\t")

    # write covgene targets table (all targets are gene tracks here)
    if covgene_stats:
        targets_out_df.to_csv(f"{args.out_dir}/targets_covgene.txt", sep="\t")

    pair_h5_keys = covgene_stats + gene_stats

    scores_out = initialize_output_h5(
        args.out_dir,
        snps,
        output_length,
        targets_out_df.shape[0],
        num_shifts,
        covgene_stats=covgene_stats,
        gene_stats=gene_stats if has_gene_head else (),
        genesnp_clusters=genesnp_clusters,
        gene_depth=gene_depth,
    )

    # write to HDF5 in CPU thread
    def hdf5_write_pair(cov_scores, gene_head_scores, score_index):
        for stat in covgene_stats:
            scores_out[stat][score_index] = cov_scores[stat]
        for stat in gene_stats:
            scores_out[stat][score_index] = gene_head_scores[stat]

    # SNP-gene pair index
    sgi = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with (
            torch.no_grad(),
            torch.autocast(device_type=seqnn_model.device, dtype=args.mix_dtype),
        ):
            for gsc in tqdm(genesnp_clusters, unit=" cluster", desc="gene clusters"):
                snp_1hot_list = gsc.get_1hots(genome_open)
                ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)
                ref_1hot = torch.tensor(
                    ref_1hot, device=seqnn_model.device, dtype=torch.float32
                )

                # build gene masks
                gom, gp = build_gene_masks(
                    gsc.genes,
                    gsc.pstart,
                    gsc.pend - gsc.pstart,
                    output_stride,
                    output_length,
                    seqnn_model.device,
                    span=args.span,
                )

                # predict reference
                ref_cov_preds = []
                ref_gene_preds = []
                for shift in args.shifts:
                    ref_1hot_shift = dna.torch_shift(ref_1hot, shift)
                    output = seqnn_model(
                        ref_1hot_shift,
                        args.head,
                        gene_out_mask=gom,
                        gene_presence=gp,
                    )
                    ref_cov_shift = seqnn_model.untransform_fn(
                        output.coverage.squeeze(0), targets_df
                    )
                    ref_cov_preds.append(ref_cov_shift)
                    if has_gene_head and output.has_gene:
                        ref_gene_preds.append(output.gene.squeeze(0))

                for ai, alt_1hot in enumerate(snp_1hot_list[1:]):
                    alt_1hot = np.expand_dims(alt_1hot, axis=0)
                    alt_1hot = torch.tensor(
                        alt_1hot, device=seqnn_model.device, dtype=torch.float32
                    )

                    # add left/right shifts for indels
                    indel_size = gsc.snps[ai].indel_size()
                    if indel_size == 0:
                        alt_shifts = args.shifts
                    else:
                        alt_shifts = []
                        for shift in args.shifts:
                            alt_shifts.append(shift)
                            alt_shifts.append(shift - indel_size)

                    # predict alternate
                    alt_cov_preds = []
                    alt_gene_preds = []
                    for shift in alt_shifts:
                        alt_1hot_shift = dna.torch_shift(alt_1hot, shift)
                        output = seqnn_model(
                            alt_1hot_shift,
                            args.head,
                            gene_out_mask=gom,
                            gene_presence=gp,
                        )
                        alt_cov_shift = seqnn_model.untransform_fn(
                            output.coverage.squeeze(0), targets_df
                        )
                        alt_cov_preds.append(alt_cov_shift)
                        if has_gene_head and output.has_gene:
                            alt_gene_preds.append(output.gene.squeeze(0))

                    # stitch indel shifts
                    if indel_size != 0 and args.indel_stitch:
                        snp_seq_pos = gsc.snps[ai].pos - gsc.start - output_crop
                        snp_seq_bin = snp_seq_pos // output_stride
                        alt_cov_preds = stitch_preds(
                            alt_cov_preds, args.shifts, snp_seq_bin
                        )

                    # flip reference and alternate
                    if gsc.snps[ai].flipped:
                        rp_cov = torch.stack(alt_cov_preds)
                        ap_cov = torch.stack(ref_cov_preds)
                        if has_gene_head and ref_gene_preds:
                            rp_gene = torch.stack(alt_gene_preds)
                            ap_gene = torch.stack(ref_gene_preds)
                    else:
                        rp_cov = torch.stack(ref_cov_preds)
                        ap_cov = torch.stack(alt_cov_preds)
                        if has_gene_head and ref_gene_preds:
                            rp_gene = torch.stack(ref_gene_preds)
                            ap_gene = torch.stack(alt_gene_preds)

                    # repeat reference predictions for indels w/o stitching
                    if indel_size != 0 and not args.indel_stitch:
                        rp_cov = rp_cov.repeat_interleave(2, dim=0)
                        if has_gene_head and ref_gene_preds:
                            rp_gene = rp_gene.repeat_interleave(2, dim=0)

                    for gi, gene in enumerate(gsc.genes):
                        gene_mask = gom[0, gi]
                        if not gene_mask.any():
                            print(
                                f"WARNING: {gene.kv['gene_id']} exons fall outside prediction boundaries."
                            )
                        else:
                            rp_cov_gene = rp_cov[:, :, gene_mask]
                            ap_cov_gene = ap_cov[:, :, gene_mask]

                            # slice gene strand
                            if gene.strand == "+":
                                track_mask = plus_mask
                            else:
                                track_mask = minus_mask
                            rp_cov_gene = rp_cov_gene[:, track_mask, :]
                            ap_cov_gene = ap_cov_gene[:, track_mask, :]

                            # coverage-based gene-sliced scores
                            cov_scores = compute_scores_cov(
                                rp_cov_gene, ap_cov_gene, covgene_stats
                            )

                            # gene head scores
                            gene_head_scores = {}
                            if gene_stats and has_gene_head and ref_gene_preds:
                                gene_head_scores = compute_scores_gene(
                                    rp_gene[:, :, gi],
                                    ap_gene[:, :, gi],
                                    gene_stats,
                                )

                            # write
                            executor.submit(
                                hdf5_write_pair, cov_scores, gene_head_scores, sgi
                            )

                        # update SNP-gene index
                        sgi += 1

                # clean up memory
                del rp_cov, ref_cov_preds
                del ap_cov, alt_cov_preds
                torch.cuda.empty_cache()

    # close open files
    genome_open.close()

    out_snpgene_num = len(scores_out["snp_idx"])
    assert sgi == out_snpgene_num, (
        f"SNP-gene index {sgi} does not match expected {out_snpgene_num}"
    )

    # compute quantiles
    write_quantiles(scores_out, pair_h5_keys, args.norm_file)

    # Mark job as completed before closing
    del scores_out["progress_status"]
    scores_out.create_dataset("progress_status", data="completed".encode("utf-8"))
    scores_out.close()


def cluster_genes(transcriptome, seq_length: int, center_pct: float):
    """Cluster genes into regions that will satisfy the required center_pct.

    Args:
        transcriptome (Transcriptome): Transcriptome object.
        seq_length (int): Sequence length.
        center_pct (float): Percent of sequence length to cluster genes.
    """
    valid_gene_distance = int(seq_length * center_pct)

    gene_clusters = []

    # re-sort genes by midpoint
    chromosomes = set([gene.chrom for gene in transcriptome.genes.values()])
    for chrom in chromosomes:
        gene_pos = []
        gene_objs = []
        for gene in transcriptome.genes.values():
            if gene.chrom == chrom:
                gene_pos.append(gene.midpoint())
                gene_objs.append(gene)

        cluster_pos0 = -valid_gene_distance
        for gi in np.argsort(gene_pos):
            gene = gene_objs[gi]
            if gene_pos[gi] < cluster_pos0 + valid_gene_distance:
                # append to latest cluster
                gene_clusters[-1].add_gene(gene)
            else:
                # initialize new cluster
                gene_clusters.append(GeneSNPCluster())
                gene_clusters[-1].add_gene(gene)
                cluster_pos0 = gene_pos[gi]

    return gene_clusters


def construct_pregrouped_geneclusters(transcriptome, snps):
    """Construct pre-grouped gene clusters.

    Args:
        transcriptome (Transcriptome): Transcriptome object.
        snps [SNP]: List of SNPs.

    Returns:
        List[GeneSNPCluster]: List of pre-grouped gene clusters.
    """

    gene_snp_clusters_dict = {}

    for snp in snps:
        for gene_id in snp.linked_gene_ids:
            gene = transcriptome.genes.get(gene_id, None)
            if gene is None:
                raise ValueError(
                    f"SNP {snp.chr}:{snp.pos} references gene_id {gene_id} not found in Transcriptome."
                )

            if (snp.chr, snp.seq_start, snp.seq_end) in gene_snp_clusters_dict:
                # use existing cluster
                gsc = gene_snp_clusters_dict[(snp.chr, snp.seq_start, snp.seq_end)]

                existing_gene_ids = {g.kv["gene_id"] for g in gsc.genes}
                if gene.kv["gene_id"] not in existing_gene_ids:
                    gsc.add_gene(gene)
                if snp not in gsc.snps:
                    gsc.add_snp(snp)
            else:
                # initialize new GeneSNPCluster
                gsc = GeneSNPCluster()
                gsc.chr = snp.chr
                gsc.start = snp.seq_start
                gsc.end = snp.seq_end
                gsc.pstart = gsc.start
                gsc.pend = gsc.end
                gsc.add_gene(gene)
                gsc.add_snp(snp)
                gene_snp_clusters_dict[(gsc.chr, gsc.start, gsc.end)] = gsc

    gene_snp_clusters = list(gene_snp_clusters_dict.values())
    return gene_snp_clusters


def cluster_snps(snps, seq_len: int, center_pct: float):
    """Cluster a sorted list of SNPs into regions that will satisfy
       the required center_pct.

    Args:
        snps [SNP]: List of SNPs.
        seq_len (int): Sequence length.
        center_pct (float): Percent of sequence length to cluster SNPs.
    """
    valid_snp_distance = int(seq_len * center_pct)

    snp_clusters = []
    cluster_chr = None

    for snp in snps:
        if snp.chr == cluster_chr and snp.pos < cluster_pos0 + valid_snp_distance:
            # append to latest cluster
            snp_clusters[-1].add_snp(snp)
        else:
            # initialize new cluster
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)
            cluster_chr = snp.chr
            cluster_pos0 = snp.pos

    return snp_clusters


def compute_scores_cov(
    ref_preds,
    alt_preds,
    snp_stats,
    strand_transform=None,
    local_mask=None,
    snp_seq_bin=None,
    window_bins=None,
):
    """Compute SNP scores from reference and alternative predictions.

    Args:
        ref_preds (torch.Tensor): Reference predictions SxTxL.
        alt_preds (torch.Tensor): Alternative predictions SxTxL.
        snp_stats [str]: List of SNP stats to compute.
        strand_transform (scipy.sparse): Strand transform matrix.
        local_mask (np.ndarray): Boolean mask (T,) for targets using local window.
        snp_seq_bin (int): SNP bin position in output space.
        window_bins (int): Local window size in bins.
    """

    # strip any cov/ or covgene/ prefix for internal logic; remap at return
    bare_to_orig = {s.split("/")[-1]: s for s in snp_stats}
    bare_stats = list(bare_to_orig)

    # initialize scores dict (keyed by bare names internally)
    scores = {}
    num_shifts = ref_preds.shape[0]
    seq_len = ref_preds.shape[2]

    # precompute windowed slices for local targets
    if local_mask is not None and local_mask.any():
        lm = torch.tensor(local_mask, device=ref_preds.device)
        ws = max(0, snp_seq_bin - window_bins // 2)
        we = min(seq_len, snp_seq_bin + window_bins // 2)
        ref_win = ref_preds[:, lm, ws:we]
        alt_win = alt_preds[:, lm, ws:we]
    else:
        lm = None

    # handle CPU conversion, strand transform, and clipping
    def strand_clip_save(key, score, d2=False):
        score = score.cpu().numpy().T
        if strand_transform is not None:
            if d2:
                score = np.power(score, 2)
                score = score @ strand_transform
                score = np.sqrt(score)
            else:
                score = score @ strand_transform
        score = np.clip(score, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores[key] = score.astype("float16")

    # pre-compute sum statistics
    ref_preds_sum = ref_preds.sum(dim=(0, 2)) / num_shifts
    alt_preds_sum = alt_preds.sum(dim=(0, 2)) / num_shifts
    if lm is not None:
        ref_preds_sum[lm] = ref_win.sum(dim=(0, 2)) / num_shifts
        alt_preds_sum[lm] = alt_win.sum(dim=(0, 2)) / num_shifts

    # compare reference to alternative via sum subtraction
    if "SUM" in bare_stats:
        score_sum = alt_preds_sum - ref_preds_sum
        strand_clip_save("SUM", score_sum)
        del score_sum

    if "logSUM" in bare_stats:
        ref_preds_log_sum = torch.log2(ref_preds + 1).sum(dim=(0, 2)) / num_shifts
        alt_preds_log_sum = torch.log2(alt_preds + 1).sum(dim=(0, 2)) / num_shifts
        if lm is not None:
            ref_preds_log_sum[lm] = torch.log2(ref_win + 1).sum(dim=(0, 2)) / num_shifts
            alt_preds_log_sum[lm] = torch.log2(alt_win + 1).sum(dim=(0, 2)) / num_shifts
        score_sum = alt_preds_log_sum - ref_preds_log_sum
        strand_clip_save("logSUM", score_sum)
        del score_sum, ref_preds_log_sum, alt_preds_log_sum

    # log fold change of summed predictions (logFC and logSED are synonymous)
    if "logFC" in bare_stats or "logSED" in bare_stats:
        ref_sum_log = torch.log2(ref_preds_sum + 1)
        alt_sum_log = torch.log2(alt_preds_sum + 1)
        score_logfc = alt_sum_log - ref_sum_log
        if "logFC" in bare_stats:
            strand_clip_save("logFC", score_logfc)
        if "logSED" in bare_stats:
            strand_clip_save("logSED", score_logfc)
        del ref_sum_log, alt_sum_log, score_logfc

    # L1 norm of difference vector
    if "D1" in bare_stats:
        altref_diff = alt_preds - ref_preds
        score_d1 = torch.linalg.vector_norm(altref_diff, ord=1, dim=2).mean(dim=0)
        if lm is not None:
            win_diff = alt_win - ref_win
            score_d1[lm] = torch.linalg.vector_norm(win_diff, ord=1, dim=2).mean(dim=0)
        strand_clip_save("D1", score_d1)
        del altref_diff, score_d1

    if "logD1" in bare_stats:
        altref_log_diff = torch.log2(alt_preds + 1) - torch.log2(ref_preds + 1)
        score_d1 = torch.linalg.vector_norm(altref_log_diff, ord=1, dim=2).mean(dim=0)
        if lm is not None:
            win_log_diff = torch.log2(alt_win + 1) - torch.log2(ref_win + 1)
            score_d1[lm] = torch.linalg.vector_norm(win_log_diff, ord=1, dim=2).mean(
                dim=0
            )
        strand_clip_save("logD1", score_d1)
        del altref_log_diff, score_d1

    # L2 norm of difference vector
    if "D2" in bare_stats:
        altref_diff = alt_preds - ref_preds
        score_d2 = torch.linalg.vector_norm(altref_diff, ord=2, dim=2).mean(dim=0)
        if lm is not None:
            win_diff = alt_win - ref_win
            score_d2[lm] = torch.linalg.vector_norm(win_diff, ord=2, dim=2).mean(dim=0)
        strand_clip_save("D2", score_d2, d2=True)
        del altref_diff, score_d2

    if "logD2" in bare_stats:
        altref_log_diff = torch.log2(alt_preds + 1) - torch.log2(ref_preds + 1)
        score_d2 = torch.linalg.vector_norm(altref_log_diff, ord=2, dim=2).mean(dim=0)
        if lm is not None:
            win_log_diff = torch.log2(alt_win + 1) - torch.log2(ref_win + 1)
            score_d2[lm] = torch.linalg.vector_norm(win_log_diff, ord=2, dim=2).mean(
                dim=0
            )
        strand_clip_save("logD2", score_d2, d2=True)
        del altref_log_diff, score_d2

    # normalized distribution scores (for splicing QTL detection)
    nD_stats = {"nD1", "nD2", "nDinf", "JSD", "lnD1", "lnD2", "lnDinf", "lJSD"}
    if nD_stats & set(bare_stats):
        eps = 1e-8

        def normalize_to_dist(preds, do_log=False):
            if do_log:
                preds = torch.log2(preds + 1)
            return preds / (preds.sum(dim=2, keepdim=True) + eps)

        for do_log, prefix in [(False, ""), (True, "l")]:
            active = [
                s
                for s in [
                    f"{prefix}nD1",
                    f"{prefix}nD2",
                    f"{prefix}nDinf",
                    f"{prefix}JSD",
                ]
                if s in bare_stats
            ]
            if not active:
                continue

            ref_norm = normalize_to_dist(ref_preds, do_log=do_log)
            alt_norm = normalize_to_dist(alt_preds, do_log=do_log)
            if lm is not None:
                ref_win_norm = normalize_to_dist(ref_win, do_log=do_log)
                alt_win_norm = normalize_to_dist(alt_win, do_log=do_log)

            norm_diff = alt_norm - ref_norm
            if lm is not None:
                win_norm_diff = alt_win_norm - ref_win_norm

            if f"{prefix}nD1" in bare_stats:
                score = torch.linalg.vector_norm(norm_diff, ord=1, dim=2).mean(dim=0)
                if lm is not None:
                    score[lm] = torch.linalg.vector_norm(
                        win_norm_diff, ord=1, dim=2
                    ).mean(dim=0)
                strand_clip_save(f"{prefix}nD1", score)
                del score

            if f"{prefix}nD2" in bare_stats:
                score = torch.linalg.vector_norm(norm_diff, ord=2, dim=2).mean(dim=0)
                if lm is not None:
                    score[lm] = torch.linalg.vector_norm(
                        win_norm_diff, ord=2, dim=2
                    ).mean(dim=0)
                strand_clip_save(f"{prefix}nD2", score, d2=True)
                del score

            if f"{prefix}nDinf" in bare_stats:
                score = torch.linalg.vector_norm(
                    norm_diff, ord=float("inf"), dim=2
                ).mean(dim=0)
                if lm is not None:
                    score[lm] = torch.linalg.vector_norm(
                        win_norm_diff, ord=float("inf"), dim=2
                    ).mean(dim=0)
                strand_clip_save(f"{prefix}nDinf", score)
                del score

            if f"{prefix}JSD" in bare_stats:
                m = 0.5 * (ref_norm + alt_norm)
                jsd = 0.5 * torch.xlogy(ref_norm, ref_norm / (m + eps)).sum(dim=2)
                jsd += 0.5 * torch.xlogy(alt_norm, alt_norm / (m + eps)).sum(dim=2)
                score = jsd.mean(dim=0)
                if lm is not None:
                    m_win = 0.5 * (ref_win_norm + alt_win_norm)
                    jsd_win = 0.5 * torch.xlogy(
                        ref_win_norm, ref_win_norm / (m_win + eps)
                    ).sum(dim=2)
                    jsd_win += 0.5 * torch.xlogy(
                        alt_win_norm, alt_win_norm / (m_win + eps)
                    ).sum(dim=2)
                    score[lm] = jsd_win.mean(dim=0)
                    del m_win, jsd_win
                strand_clip_save(f"{prefix}JSD", score)
                del score

            del ref_norm, alt_norm, norm_diff
            if lm is not None:
                del ref_win_norm, alt_win_norm, win_norm_diff

    # predictions
    if "REF" in bare_stats:
        ref_preds_cpu = ref_preds.cpu().numpy().transpose(0, 2, 1)
        ref_preds_cpu = np.clip(
            ref_preds_cpu, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["REF"] = ref_preds_cpu.astype("float16")
        del ref_preds_cpu
    if "ALT" in bare_stats:
        alt_preds_cpu = alt_preds.cpu().numpy().transpose(0, 2, 1)
        alt_preds_cpu = np.clip(
            alt_preds_cpu, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["ALT"] = alt_preds_cpu.astype("float16")
        del alt_preds_cpu

    return {bare_to_orig[k]: v for k, v in scores.items()}


def compute_scores_gene(ref_preds, alt_preds, gene_stats):
    """Compute gene head scores from reference and alternative predictions.

    Args:
        ref_preds (torch.Tensor): Reference gene predictions (S, T_gene).
        alt_preds (torch.Tensor): Alternative gene predictions (S, T_gene).
        gene_stats [str]: List of gene head stats to compute (may include gene/ prefix).

    Returns:
        dict: Mapping original stat name to float16 numpy array.
    """
    bare_to_orig = {s.split("/")[-1]: s for s in gene_stats}
    bare_stats = list(bare_to_orig)
    scores = {}
    if "logFC" in bare_stats:
        ref_mean = ref_preds.mean(dim=0)
        alt_mean = alt_preds.mean(dim=0)
        logfc = torch.log2(alt_mean + 1) - torch.log2(ref_mean + 1)
        logfc = logfc.cpu().numpy()
        logfc = np.clip(logfc, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores["logFC"] = logfc.astype("float16")
    return {bare_to_orig[k]: v for k, v in scores.items()}


def initialize_output_h5(
    out_dir,
    snps,
    output_length,
    output_depth,
    num_shifts,
    cov_stats=(),
    covgene_stats=(),
    gene_stats=(),
    genesnp_clusters=None,
    gene_depth=0,
    raw_depth=None,
    covgene_depth=None,
):
    """Initialize an output HDF5 file for SNP scoring.

    Args:
        out_dir (str): Output directory.
        snps [SNP]: List of SNPs.
        output_length (int): Targets' sequence length.
        output_depth (int): Number of coverage targets.
        num_shifts (int): Number of shifts.
        cov_stats (list[str]): Coverage stats, prefixed with cov/ (e.g. cov/logD2).
        covgene_stats (list[str]): Gene-sliced coverage stats, prefixed with covgene/.
        gene_stats (list[str]): Gene head stats, prefixed with gene/.
        genesnp_clusters [GeneSNPCluster]: Gene sequence clusters (only for gene mode).
        gene_depth (int): Number of gene head targets (0 if no gene head).
        raw_depth (int): Target count BEFORE strand collapse, used for the
            cov/REF and cov/ALT datasets (raw per-target predictions are not
            strand-summed). Defaults to output_depth when None.
        covgene_depth (int): Target count for covgene/ datasets (gene-track
            subset of strand-collapsed targets). Defaults to output_depth.
    """
    num_snps = len(snps)

    scores_out = h5py.File(f"{out_dir}/scores.h5", "w")

    # Write progress status as initialized
    scores_out.create_dataset("progress_status", data="initialized".encode("utf-8"))

    # write SNPs
    snp_ids = [snp.rsid for snp in snps]
    snp_ids_np = np.array([snp.rsid for snp in snps], "S")
    scores_out.create_dataset("snp", data=snp_ids_np)

    # write SNP chr
    snp_chr = np.array([snp.chr for snp in snps], "S")
    scores_out.create_dataset("chr", data=snp_chr)

    # write SNP pos
    snp_pos = np.array([snp.pos for snp in snps], dtype="uint32")
    scores_out.create_dataset("pos", data=snp_pos)

    # write SNP reference allele
    snp_refs = []
    snp_alts = []
    for snp in snps:
        if snp.flipped:
            snp_refs.append(snp.alt_allele)
            snp_alts.append(snp.ref_allele)
        else:
            snp_refs.append(snp.ref_allele)
            snp_alts.append(snp.alt_allele)
    scores_out.create_dataset("ref_allele", data=np.array(snp_refs, "S"))
    scores_out.create_dataset("alt_allele", data=np.array(snp_alts, "S"))

    # SNP-indexed coverage stats (cov/)
    for cov_stat in cov_stats:
        if cov_stat in ["cov/REF", "cov/ALT"]:
            scores_out.create_dataset(
                cov_stat,
                shape=(
                    num_snps,
                    num_shifts,
                    output_length,
                    output_depth if raw_depth is None else raw_depth,
                ),
                dtype="float16",
            )
        else:
            scores_out.create_dataset(
                cov_stat, shape=(num_snps, output_depth), dtype="float16"
            )

    # SNP-gene pair-indexed stats (covgene/ and gene/)
    has_pair_stats = (covgene_stats or gene_stats) and genesnp_clusters is not None
    if has_pair_stats:
        # Collect all unique gene IDs from the gene clusters
        gene_ids = sorted(
            list(
                set(
                    gene.kv["gene_id"]
                    for gsc in genesnp_clusters
                    for gene in gsc.genes
                    if gsc.snps  # Only include clusters with SNPs
                )
            )
        )
        scores_out.create_dataset("gene_ids", data=np.array(gene_ids, "S"))

        # Map SNPs/genes to their positions in the global SNP list
        snp_to_index = {snp_id: idx for idx, snp_id in enumerate(snp_ids)}
        gene_to_index = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        score_map_snp_idx = []
        score_map_gene_idx = []
        for gsc in genesnp_clusters:
            for snp_in_gsc in gsc.snps:
                for gene in gsc.genes:
                    score_map_snp_idx.append(snp_to_index[snp_in_gsc.rsid])
                    score_map_gene_idx.append(gene_to_index[gene.kv["gene_id"]])

        scores_out.create_dataset("snp_idx", data=np.array(score_map_snp_idx))
        scores_out.create_dataset("gene_idx", data=np.array(score_map_gene_idx))

        num_pairs = len(score_map_snp_idx)

        covgene_out_depth = covgene_depth if covgene_depth is not None else output_depth
        for covgene_stat in covgene_stats:
            scores_out.create_dataset(
                covgene_stat,
                shape=(num_pairs, covgene_out_depth),
                dtype="float16",
            )
        for gene_stat in gene_stats:
            scores_out.create_dataset(
                gene_stat, shape=(num_pairs, gene_depth), dtype="float16"
            )

    return scores_out


def make_gene_bedt(genesnp_clusters):
    """Make a BedTool object for all gene sequences."""
    gene_bed_lines = []
    for gi, gsc in enumerate(genesnp_clusters):
        geneseq_start = max(0, gsc.start)
        gene_bed_lines.append("%s %d %d %d" % (gsc.chr, geneseq_start, gsc.end, gi))
    gene_bedt = pybedtools.BedTool("\n".join(gene_bed_lines), from_string=True)
    return gene_bedt


def make_snp_bedt(snps):
    """Make a BedTool object for all SNPs"""
    snp_bed_lines = []
    for si, snp in enumerate(snps):
        snp_bed_lines.append("%s %d %d %d" % (snp.chr, snp.pos - 1, snp.pos, si))
    snp_bedt = pybedtools.BedTool("\n".join(snp_bed_lines), from_string=True)
    return snp_bedt


def map_snps_genes(snps, genesnp_clusters):
    """Map SNPs to gene sequences."""
    geneseq_bedt = make_gene_bedt(genesnp_clusters)
    snp_bedt = make_snp_bedt(snps)

    for overlap in geneseq_bedt.intersect(snp_bedt, wa=True, wb=True):
        gchr, gstart, gend, gi, schr, spos, send, si = overlap
        gi, si = int(gi), int(si)
        genesnp_clusters[gi].add_snp(snps[si])


def stitch_preds(preds, shifts, pos=None):
    """Stitch indel left and right compensation shifts.

    Args:
        preds [list of torch.Tensor]: List of TxL prediction tensors
        shifts [list of int]: List of shifts.
        pos (int): SNP position to stitch at.

    Returns:
        list of torch.Tensor: List of TxL prediction tensors
    """
    if pos is None:
        pos = preds[0].shape[1] // 2
    preds_stitch = []
    for hi, shift in enumerate(shifts):
        hil = 2 * hi
        hir = hil + 1
        preds_stitch_i = torch.cat((preds[hil][:, :pos], preds[hir][:, pos:]), dim=1)
        preds_stitch.append(preds_stitch_i)
    return preds_stitch


def write_quantiles(scores_out, snp_stats, norm_file=None):
    """Compute quantile values for each target and write to HDF5.

    Args:
        scores_out (h5py.File): Output HDF5 file.
        snp_stats [str]: List of SNP stats to compute.
        norm_file (str): Optional HDF5 file to use for normalization quantiles instead of current scores.
    """
    # define quantiles
    d_fine = 0.001
    d_coarse = 0.01
    quantiles_neg = np.arange(d_fine, 0.1, d_fine)
    quantiles_base = np.arange(0.1, 0.9, d_coarse)
    quantiles_pos = np.arange(0.9, 1, d_fine)

    # Create quantiles dataset (if not already present)
    quantiles = np.concatenate([quantiles_neg, quantiles_base, quantiles_pos])
    if "quantiles" not in scores_out:
        scores_out.create_dataset("quantiles", data=quantiles)

    for snp_stat in snp_stats:
        if snp_stat not in ["REF", "ALT"]:
            quantiles_key = f"{snp_stat}_quantiles"

            # Check if the dataset exists
            if snp_stat not in scores_out:
                raise KeyError(f"SNP stat '{snp_stat}' missing in quantile stage.")

            if norm_file is not None:
                if not os.path.exists(norm_file):
                    raise FileNotFoundError(
                        f"Normalization file not found: {norm_file}"
                    )

                # Use normalization file for quantile calculation
                with h5py.File(norm_file, "r") as norm_h5:
                    if snp_stat not in norm_h5:
                        raise KeyError(
                            f"SNP statistic '{snp_stat}' not found in normalization file {norm_file}"
                        )

                    # compute quantiles from normalization file
                    score_q = norm_h5[quantiles_key][:]
            else:
                # compute quantiles from current scores
                score_q = np.quantile(scores_out[snp_stat], quantiles, axis=0).T

            # convert to float16 and save
            score_q = score_q.astype("float16")
            scores_out.create_dataset(quantiles_key, data=score_q, dtype="float16")


class GeneSNPCluster(SNPCluster):
    def __init__(self):
        super().__init__()
        self.genes = []

    def add_gene(self, gene):
        """Add gene to cluster."""
        self.genes.append(gene)

    def delimit(self, seq_len, crop=0):
        """Delimit sequence boundaries."""
        self.chr = self.genes[0].chrom
        midp = int(np.mean([g.midpoint() for g in self.genes]))
        self.start = midp - seq_len // 2
        self.end = self.start + seq_len
        self.pstart = self.start + crop
        self.pend = self.end - crop


def build_gene_masks(
    genes, pred_start, pred_length, output_stride, output_length, device, span=False
):
    """Construct gene_out_mask and gene_presence tensors.

    Masks index into the prediction region (post-crop), where bin 0
    corresponds to pred_start. Used for both gene head inference and
    coverage gene-slicing.

    Args:
        genes (list[Gene]): Gene objects overlapping the sequence.
        pred_start (int): Genomic start of prediction region (seq_start + crop).
        pred_length (int): Prediction region length in bp (seq_length - 2*crop).
        output_stride (int): Model output stride in bp.
        output_length (int): Number of output bins.
        device: Torch device.
        span (bool): Use gene span instead of exons.

    Returns:
        gene_out_mask: (1, num_genes, output_length) bool tensor.
        gene_presence: (1, num_genes) bool tensor.
    """
    num_genes = len(genes)
    gene_out_mask = torch.zeros(
        1, num_genes, output_length, dtype=torch.bool, device=device
    )
    gene_presence = torch.ones(1, num_genes, dtype=torch.bool, device=device)

    for gi, gene in enumerate(genes):
        gene_slice = gene.output_slice(
            pred_start, pred_length, output_stride, span=span
        )
        if len(gene_slice) > 0:
            gene_out_mask[0, gi, gene_slice] = True
        else:
            gene_presence[0, gi] = False

    return gene_out_mask, gene_presence
