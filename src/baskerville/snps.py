import json
import pdb
import sys

import h5py
import numpy as np
import pandas as pd
import pysam
from scipy.special import rel_entr
from tqdm import tqdm

from baskerville import dna
from baskerville import dataset
from baskerville import seqnn
from baskerville import vcf as bvcf
from baskerville.helpers.trt_optimized_model import OptimizedModel


def score_snps(params_file, model_file, vcf_file, worker_index, options):
    """
    Score SNPs in a VCF file with a SeqNN model.

    :param params_file: Model parameters
    :param model_file: Saved model weights
    :param vcf_file: VCF
    :param worker_index
    :param options: options from cmd args
    :return:
    """

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # read targets
    if options.targets_file is None:
        print("Must provide targets file to clarify stranded datasets", file=sys.stderr)
        exit(1)
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]

        # construct strand sum transform
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None

    #################################################################
    # setup model

    # can we sum on GPU?
    seqnn_model = seqnn.SeqNN(params_model)

    # load model
    sum_length = options.snp_stats == "SAD"
    if options.tensorrt:
        seqnn_model.model = OptimizedModel(model_file, seqnn_model.strand_pair)
        input_shape = tuple(seqnn_model.model.loaded_model_fn.inputs[0].shape.as_list())
    else:
        seqnn_model.restore(model_file)
        seqnn_model.build_slice(targets_df.index)
        if sum_length:
            seqnn_model.build_sad()
        seqnn_model.build_ensemble(options.rc)
        input_shape = seqnn_model.model.input_shape

    # make dummy predictions to warm up model
    dummy_input_shape = (1,) + input_shape[1:]
    dummy_input = np.random.random(dummy_input_shape).astype(np.float32)
    dummy_output = seqnn_model(dummy_input)

    # shift outside seqnn
    num_shifts = len(options.shifts)
    targets_length = seqnn_model.target_lengths[0]

    #################################################################
    # load SNPs

    # clustering SNPs requires sorted VCF and no reference flips
    snps_clustered = options.cluster_snps_pct > 0

    # filter for worker SNPs
    if options.processes is None:
        start_i = None
        end_i = None
    else:
        # determine boundaries
        num_snps = bvcf.vcf_count(vcf_file)
        worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype="int")
        start_i = worker_bounds[worker_index]
        end_i = worker_bounds[worker_index + 1]

    # read SNPs
    snps = bvcf.vcf_snps(
        vcf_file,
        require_sorted=snps_clustered,
        flip_ref=~snps_clustered,
        validate_ref_fasta=options.genome_fasta,
        start_i=start_i,
        end_i=end_i,
    )

    # cluster SNPs
    if snps_clustered:
        snp_clusters = cluster_snps(
            snps, params_model["seq_length"], options.cluster_snps_pct
        )
    else:
        snp_clusters = []
        for snp in snps:
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)

    # delimit sequence boundaries
    [sc.delimit(params_model["seq_length"]) for sc in snp_clusters]

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    #################################################################
    # predict SNP scores, write output

    # setup output
    scores_out = initialize_output_h5(
        options.out_dir,
        options.snp_stats,
        snps,
        targets_length,
        targets_strand_df,
        num_shifts,
    )

    # SNP index
    si = 0

    for sc in tqdm(snp_clusters):
        snp_1hot_list = sc.get_1hots(genome_open)
        ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)

        # predict reference
        ref_preds = []
        for shift in options.shifts:
            ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
            ref_preds_shift = seqnn_model(ref_1hot_shift)[0]

            # untransform predictions
            if options.targets_file is not None:
                if options.untransform_old:
                    ref_preds_shift = dataset.untransform_preds1(
                        ref_preds_shift, targets_df
                    )
                else:
                    ref_preds_shift = dataset.untransform_preds(
                        ref_preds_shift, targets_df
                    )

            # save shift prediction
            ref_preds.append(ref_preds_shift)
        ref_preds = np.array(ref_preds)

        ai = 0
        for alt_1hot in snp_1hot_list[1:]:
            alt_1hot = np.expand_dims(alt_1hot, axis=0)

            # add compensation shifts for indels
            indel_size = sc.snps[ai].indel_size()
            if indel_size == 0:
                alt_shifts = options.shifts
            else:
                # repeat reference predictions, unless stitching
                if not options.indel_stitch:
                    ref_preds = np.repeat(ref_preds, 2, axis=0)

                # add compensation shifts
                alt_shifts = []
                for shift in options.shifts:
                    alt_shifts.append(shift)
                    alt_shifts.append(shift - indel_size)

            # predict alternate
            alt_preds = []
            for shift in alt_shifts:
                alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                alt_preds_shift = seqnn_model(alt_1hot_shift)[0]

                # untransform predictions
                if options.targets_file is not None:
                    if options.untransform_old:
                        alt_preds_shift = dataset.untransform_preds1(
                            alt_preds_shift, targets_df
                        )
                    else:
                        alt_preds_shift = dataset.untransform_preds(
                            alt_preds_shift, targets_df
                        )

                # save shift prediction
                alt_preds.append(alt_preds_shift)

            # stitch indel compensation shifts
            if indel_size != 0 and options.indel_stitch:
                alt_preds = stitch_preds(alt_preds, options.shifts)

            # flip reference and alternate
            if snps[si].flipped:
                rp_snp = np.array(alt_preds)
                ap_snp = np.array(ref_preds)
            else:
                rp_snp = np.array(ref_preds)
                ap_snp = np.array(alt_preds)

            # write SNP
            if sum_length:
                write_snp(rp_snp, ap_snp, scores_out, si, options.snp_stats)
            else:
                # write_snp_len(rp_snp, ap_snp, scores_out, si, options.snp_stats)
                scores = compute_scores(rp_snp, ap_snp, options.snp_stats, strand_transform)
                for snp_stat in options.snp_stats:
                    scores_out[snp_stat][si] = scores[snp_stat]

            # update SNP index
            si += 1

    # close genome
    genome_open.close()

    # compute SAD distributions across variants
    write_pct(scores_out, options.snp_stats)
    scores_out.close()


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


def compute_scores(ref_preds, alt_preds, snp_stats, strand_transform):
    """Compute SNP scores from reference and alternative predictions.

    Args:
        ref_preds (np.array): Reference allele predictions.
        alt_preds (np.array): Alternative allele predictions.
        snp_stats [str]: List of SAD stats to compute.
        strand_transform (scipy.sparse): Strand transform matrix.
    """
    num_shifts, seq_length, num_targets = ref_preds.shape

    # log/sqrt
    ref_preds_log = np.log2(ref_preds + 1)
    alt_preds_log = np.log2(alt_preds + 1)
    ref_preds_sqrt = np.sqrt(ref_preds)
    alt_preds_sqrt = np.sqrt(alt_preds)

    # sum across length, mean across shifts
    ref_preds_sum = ref_preds.sum(axis=(0, 1)) / num_shifts
    alt_preds_sum = alt_preds.sum(axis=(0, 1)) / num_shifts
    ref_preds_log_sum = ref_preds_log.sum(axis=(0, 1)) / num_shifts
    alt_preds_log_sum = alt_preds_log.sum(axis=(0, 1)) / num_shifts
    ref_preds_sqrt_sum = ref_preds_sqrt.sum(axis=(0, 1)) / num_shifts
    alt_preds_sqrt_sum = alt_preds_sqrt.sum(axis=(0, 1)) / num_shifts

    # difference
    altref_diff = alt_preds - ref_preds
    altref_log_diff = alt_preds_log - ref_preds_log
    altref_sqrt_diff = alt_preds_sqrt - ref_preds_sqrt

    # initialize scores dict
    scores = {}

    def strand_clip_save(key, score, d2=False):
        if strand_transform is not None:
            if d2:
                score = np.power(score, 2)
                score = score @ strand_transform
                score = np.sqrt(score)
            else:
                score = score @ strand_transform
        score = np.clip(score, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores[key] = score.astype("float16")

    # compare reference to alternative via sum subtraction
    if "SUM" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        strand_clip_save("SUM", sad)
    if "logSUM" in snp_stats:
        log_sad = alt_preds_log_sum - ref_preds_log_sum
        strand_clip_save("logSUM", log_sad)
    if "sqrtSUM" in snp_stats:
        sqrt_sad = alt_preds_sqrt_sum - ref_preds_sqrt_sum
        strand_clip_save("sqrtSUM", sqrt_sad)

    # compare reference to alternative via max subtraction
    if "SAX" in snp_stats:
        altref_adiff = np.abs(altref_diff)
        sax = []
        for s in range(num_shifts):
            max_i = np.argmax(altref_adiff[s], axis=0)
            sax.append(altref_diff[s, max_i, np.arange(num_targets)])
        sax = np.array(sax).mean(axis=0)
        strand_clip_save("SAX", sax)

    # L1 norm of difference vector
    if "D1" in snp_stats:
        sad_d1 = np.linalg.norm(altref_diff, ord=1, axis=1)
        strand_clip_save("D1", sad_d1.mean(axis=0))
    if "logD1" in snp_stats:
        log_d1 = np.linalg.norm(altref_log_diff, ord=1, axis=1)
        strand_clip_save("logD1", log_d1.mean(axis=0))
    if "sqrtD1" in snp_stats:
        sqrt_d1 = np.linalg.norm(altref_sqrt_diff, ord=1, axis=1)
        strand_clip_save("sqrtD1", sqrt_d1.mean(axis=0))

    # L2 norm of difference vector
    if "D2" in snp_stats:
        sad_d2 = np.linalg.norm(altref_diff, ord=2, axis=1)
        strand_clip_save("D2", sad_d2.mean(axis=0), d2=True)
    if "logD2" in snp_stats:
        log_d2 = np.linalg.norm(altref_log_diff, ord=2, axis=1)
        strand_clip_save("logD2", log_d2.mean(axis=0), d2=True)
    if "sqrtD2" in snp_stats:
        sqrt_d2 = np.linalg.norm(altref_sqrt_diff, ord=2, axis=1)
        strand_clip_save("sqrtD2", sqrt_d2.mean(axis=0), d2=True)

    if "JS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds, 25, axis=1)
        ref_preds_norm = ref_preds + pseudocounts
        ref_preds_norm /= ref_preds_norm.sum(axis=1)
        alt_preds_norm = alt_preds + pseudocounts
        alt_preds_norm /= alt_preds_norm.sum(axis=1)

        # compare normalized JS
        js_dist = []
        for s in range(num_shifts):
            ref_alt_entr = rel_entr(ref_preds_norm[s], alt_preds_norm[s]).sum(axis=0)
            alt_ref_entr = rel_entr(alt_preds_norm[s], ref_preds_norm[s]).sum(axis=0)
            js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        js_dist = np.mean(js_dist, axis=0)
        # handling strand this way is incorrect, but I'm punting for now
        strand_clip_save("JS", js_dist)

    if "logJS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds_log, 25, axis=0)
        ref_preds_log_norm = ref_preds_log + pseudocounts
        ref_preds_log_norm /= ref_preds_log_norm.sum(axis=0)
        alt_preds_log_norm = alt_preds_log + pseudocounts
        alt_preds_log_norm /= alt_preds_log_norm.sum(axis=0)

        # compare normalized JS
        log_js_dist = []
        for s in range(num_shifts):
            rps = ref_preds_log_norm[s]
            aps = alt_preds_log_norm[s]
            ref_alt_entr = rel_entr(rps, aps).sum(axis=0)
            alt_ref_entr = rel_entr(aps, rps).sum(axis=0)
            log_js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        log_js_dist = np.mean(log_js_dist, axis=0)
        # handling strand this way is incorrect, but I'm punting for now
        strand_clip_save("logJS", log_js_dist)

    # predictions
    if "REF" in snp_stats:
        ref_preds = np.clip(
            ref_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["REF"] = ref_preds.astype("float16")
    if "ALT" in snp_stats:
        alt_preds = np.clip(
            alt_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["ALT"] = alt_preds.astype("float16")

    return scores


def initialize_output_h5(
    out_dir, snp_stats, snps, targets_length, targets_df, num_shifts
):
    """Initialize an output HDF5 file for SAD stats.

    Args:
        out_dir (str): Output directory.
        snp_stats [str]: List of SAD stats to compute.
        snps [SNP]: List of SNPs.
        targets_length (int): Targets' sequence length
        targets_df (pd.DataFrame): Targets DataFrame.
        num_shifts (int): Number of shifts.
    """

    num_targets = targets_df.shape[0]
    num_snps = len(snps)

    scores_out = h5py.File("%s/scores.h5" % out_dir, "w")

    # write SNPs
    snp_ids = np.array([snp.rsid for snp in snps], "S")
    scores_out.create_dataset("snp", data=snp_ids)

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
            snp_refs.append(snp.alt_alleles[0])
            snp_alts.append(snp.ref_allele)
        else:
            snp_refs.append(snp.ref_allele)
            snp_alts.append(snp.alt_alleles[0])
    snp_refs = np.array(snp_refs, "S")
    snp_alts = np.array(snp_alts, "S")
    scores_out.create_dataset("ref_allele", data=snp_refs)
    scores_out.create_dataset("alt_allele", data=snp_alts)

    # write targets
    scores_out.create_dataset("target_ids", data=np.array(targets_df.identifier, "S"))
    scores_out.create_dataset(
        "target_labels", data=np.array(targets_df.description, "S")
    )

    # initialize SAD stats
    for snp_stat in snp_stats:
        if snp_stat in ["REF", "ALT"]:
            scores_out.create_dataset(
                snp_stat,
                shape=(num_snps, num_shifts, targets_length, num_targets),
                dtype="float16",
            )
        else:
            scores_out.create_dataset(
                snp_stat, shape=(num_snps, num_targets), dtype="float16"
            )

    return scores_out


def make_alt_1hot(ref_1hot, snp_seq_pos, ref_allele, alt_allele):
    """Return alternative allele one hot coding.

    Args:
        ref_1hot (np.array): Reference allele one hot coding.
        snp_seq_pos (int): SNP position in sequence.
        ref_allele (str): Reference allele.
        alt_allele (str): Alternative allele.

    Returns:
        np.array: Alternative allele one hot coding.
    """
    ref_n = len(ref_allele)
    alt_n = len(alt_allele)

    # copy reference
    alt_1hot = np.copy(ref_1hot)

    if alt_n == ref_n:
        # SNP
        dna.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

    elif ref_n > alt_n:
        # deletion
        delete_len = ref_n - alt_n
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_delete(alt_1hot, snp_seq_pos + 1, delete_len)
        else:
            print(
                "WARNING: Deletion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    else:
        # insertion
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_insert(alt_1hot, snp_seq_pos + 1, alt_allele[1:])
        else:
            print(
                "WARNING: Insertion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    return alt_1hot


def stitch_preds(preds, shifts):
    """Stitch indel left and right compensation shifts.

    Args:
        preds [np.array]: List of predictions.
        shifts [int]: List of shifts.
    """
    cp = preds[0].shape[0] // 2
    preds_stitch = []
    for hi, shift in enumerate(shifts):
        hil = 2 * hi
        hir = hil + 1
        preds_stitch_i = np.concatenate((preds[hil][:cp], preds[hir][cp:]), axis=0)
        preds_stitch.append(preds_stitch_i)
    return preds_stitch


def write_pct(scores_out, snp_stats):
    """Compute percentile values for each target and write to HDF5.

    Args:
        scores_out (h5py.File): Output HDF5 file.
        snp_stats [str]: List of SAD stats to compute.
    """
    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    scores_out.create_dataset("percentiles", data=percentiles)

    for snp_stat in snp_stats:
        if snp_stat not in ["REF", "ALT"]:
            snp_stat_pct = "%s_pct" % snp_stat

            # compute
            sad_pct = np.percentile(scores_out[snp_stat], 100 * percentiles, axis=0).T
            sad_pct = sad_pct.astype("float16")

            # save
            scores_out.create_dataset(snp_stat_pct, data=sad_pct, dtype="float16")


def write_snp(ref_preds_sum, alt_preds_sum, scores_out, si, snp_stats):
    """Write SNP predictions to HDF, assuming the length dimension has
    been collapsed.

    Args:
        ref_preds_sum (np.array): Reference allele predictions.
        alt_preds_sum (np.array): Alternative allele predictions.
        scores_out (h5py.File): Output HDF5 file.
        si (int): SNP index.
        snp_stats [str]: List of SAD stats to compute.
    """

    # compare reference to alternative via mean subtraction
    if "SAD" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        sad = sad.mean(axis=0)
        scores_out["SAD"][si] = sad.astype("float16")


def write_snp_len(ref_preds, alt_preds, scores_out, si, snp_stats):
    """Write SNP predictions to HDF, assuming the length dimension has
    been maintained.

    Args:
        ref_preds (np.array): Reference allele predictions.
        alt_preds (np.array): Alternative allele predictions.
        scores_out (h5py.File): Output HDF5 file.
        si (int): SNP index.
        snp_stats [str]: List of SAD stats to compute.
    """
    num_shifts, seq_length, num_targets = ref_preds.shape

    # log/sqrt
    ref_preds_log = np.log2(ref_preds + 1)
    alt_preds_log = np.log2(alt_preds + 1)
    ref_preds_sqrt = np.sqrt(ref_preds)
    alt_preds_sqrt = np.sqrt(alt_preds)

    # sum across length
    ref_preds_sum = ref_preds.sum(axis=(0, 1))
    alt_preds_sum = alt_preds.sum(axis=(0, 1))
    ref_preds_log_sum = ref_preds_log.sum(axis=(0, 1))
    alt_preds_log_sum = alt_preds_log.sum(axis=(0, 1))
    ref_preds_sqrt_sum = ref_preds_sqrt.sum(axis=(0, 1))
    alt_preds_sqrt_sum = alt_preds_sqrt.sum(axis=(0, 1))

    # difference
    altref_diff = alt_preds - ref_preds
    altref_adiff = np.abs(altref_diff)
    altref_log_diff = alt_preds_log - ref_preds_log
    altref_log_adiff = np.abs(altref_log_diff)
    altref_sqrt_diff = alt_preds_sqrt - ref_preds_sqrt
    altref_sqrt_adiff = np.abs(altref_sqrt_diff)

    # compare reference to alternative via sum subtraction
    if "SAD" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        sad = np.clip(sad, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores_out["SAD"][si] = sad.astype("float16")
    if "logSAD" in snp_stats:
        log_sad = alt_preds_log_sum - ref_preds_log_sum
        log_sad = np.clip(log_sad, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores_out["logSAD"][si] = log_sad.astype("float16")
    if "sqrtSAD" in snp_stats:
        sqrt_sad = alt_preds_sqrt_sum - ref_preds_sqrt_sum
        sqrt_sad = np.clip(sqrt_sad, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores_out["sqrtSAD"][si] = sqrt_sad.astype("float16")

    # compare reference to alternative via max subtraction
    if "SAX" in snp_stats:
        sax = []
        for s in range(num_shifts):
            max_i = np.argmax(altref_adiff[s], axis=0)
            sax.append(altref_diff[s, max_i, np.arange(num_targets)])
        sax = np.array(sax).mean(axis=0)
        scores_out["SAX"][si] = sax.astype("float16")

    # L1 norm of difference vector
    if "D1" in snp_stats:
        sad_d1 = altref_adiff.sum(axis=1)
        sad_d1 = np.clip(sad_d1, np.finfo(np.float16).min, np.finfo(np.float16).max)
        sad_d1 = sad_d1.mean(axis=0)
        scores_out["D1"][si] = sad_d1.mean().astype("float16")
    if "logD1" in snp_stats:
        log_d1 = altref_log_adiff.sum(axis=1)
        log_d1 = np.clip(log_d1, np.finfo(np.float16).min, np.finfo(np.float16).max)
        log_d1 = log_d1.mean(axis=0)
        scores_out["logD1"][si] = log_d1.astype("float16")
    if "sqrtD1" in snp_stats:
        sqrt_d1 = altref_sqrt_adiff.sum(axis=1)
        sqrt_d1 = np.clip(sqrt_d1, np.finfo(np.float16).min, np.finfo(np.float16).max)
        sqrt_d1 = sqrt_d1.mean(axis=0)
        scores_out["sqrtD1"][si] = sqrt_d1.astype("float16")

    # L2 norm of difference vector
    if "D2" in snp_stats:
        altref_diff2 = np.power(altref_diff, 2)
        sad_d2 = np.sqrt(altref_diff2.sum(axis=1))
        sad_d2 = np.clip(sad_d2, np.finfo(np.float16).min, np.finfo(np.float16).max)
        sad_d2 = sad_d2.mean(axis=0)
        scores_out["D2"][si] = sad_d2.astype("float16")
    if "logD2" in snp_stats:
        altref_log_diff2 = np.power(altref_log_diff, 2)
        log_d2 = np.sqrt(altref_log_diff2.sum(axis=1))
        log_d2 = np.clip(log_d2, np.finfo(np.float16).min, np.finfo(np.float16).max)
        log_d2 = log_d2.mean(axis=0)
        scores_out["logD2"][si] = log_d2.astype("float16")
    if "sqrtD2" in snp_stats:
        altref_sqrt_diff2 = np.power(altref_sqrt_diff, 2)
        sqrt_d2 = np.sqrt(altref_sqrt_diff2.sum(axis=1))
        sqrt_d2 = np.clip(sqrt_d2, np.finfo(np.float16).min, np.finfo(np.float16).max)
        sqrt_d2 = sqrt_d2.mean(axis=0)
        scores_out["sqrtD2"][si] = sqrt_d2.astype("float16")

    if "JS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds, 25, axis=1)
        ref_preds_norm = ref_preds + pseudocounts
        ref_preds_norm /= ref_preds_norm.sum(axis=1)
        alt_preds_norm = alt_preds + pseudocounts
        alt_preds_norm /= alt_preds_norm.sum(axis=1)

        # compare normalized JS
        js_dist = []
        for s in range(num_shifts):
            ref_alt_entr = rel_entr(ref_preds_norm[s], alt_preds_norm[s]).sum(axis=0)
            alt_ref_entr = rel_entr(alt_preds_norm[s], ref_preds_norm[s]).sum(axis=0)
            js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        js_dist = np.mean(js_dist, axis=0)
        scores_out["JS"][si] = js_dist.astype("float16")
    if "logJS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds_log, 25, axis=0)
        ref_preds_log_norm = ref_preds_log + pseudocounts
        ref_preds_log_norm /= ref_preds_log_norm.sum(axis=0)
        alt_preds_log_norm = alt_preds_log + pseudocounts
        alt_preds_log_norm /= alt_preds_log_norm.sum(axis=0)

        # compare normalized JS
        log_js_dist = []
        for s in range(num_shifts):
            ref_alt_entr = rel_entr(ref_preds_log_norm[s], alt_preds_log_norm[s]).sum(
                axis=0
            )
            alt_ref_entr = rel_entr(alt_preds_log_norm[s], ref_preds_log_norm[s]).sum(
                axis=0
            )
            log_js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        log_js_dist = np.mean(log_js_dist, axis=0)
        scores_out["logJS"][si] = log_js_dist.astype("float16")

    # predictions
    if "REF" in snp_stats:
        ref_preds = np.clip(
            ref_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores_out["REF"][si] = ref_preds.astype("float16")
    if "ALT" in snp_stats:
        alt_preds = np.clip(
            alt_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores_out["ALT"][si] = alt_preds.astype("float16")


class SNPCluster:
    def __init__(self):
        self.snps = []
        self.chr = None
        self.start = None
        self.end = None

    def add_snp(self, snp):
        """Add SNP to cluster."""
        self.snps.append(snp)

    def delimit(self, seq_len):
        """Delimit sequence boundaries."""
        positions = [snp.pos for snp in self.snps]
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_mid = (pos_min + pos_max) // 2

        self.chr = self.snps[0].chr
        self.start = pos_mid - seq_len // 2
        self.end = self.start + seq_len

        for snp in self.snps:
            snp.seq_pos = snp.pos - 1 - self.start

    def get_1hots(self, genome_open):
        """Get list of one hot coded sequences."""
        seqs1_list = []

        # extract reference
        if self.start < 0:
            ref_seq = (
                "N" * (-self.start) + genome_open.fetch(self.chr, 0, self.end).upper()
            )
        else:
            ref_seq = genome_open.fetch(self.chr, self.start, self.end).upper()

        # extend to full length
        if len(ref_seq) < self.end - self.start:
            ref_seq += "N" * (self.end - self.start - len(ref_seq))

        # verify reference alleles
        for snp in self.snps:
            ref_n = len(snp.ref_allele)
            ref_snp = ref_seq[snp.seq_pos : snp.seq_pos + ref_n]
            if snp.ref_allele != ref_snp:
                print(
                    "ERROR: %s does not match reference %s" % (snp, ref_snp),
                    file=sys.stderr,
                )
                exit(1)

        # 1 hot code reference sequence
        ref_1hot = dna.dna_1hot(ref_seq)
        seqs1_list = [ref_1hot]

        # make alternative 1 hot coded sequences
        # (assuming SNP is 1-based indexed)
        for snp in self.snps:
            alt_1hot = make_alt_1hot(
                ref_1hot, snp.seq_pos, snp.ref_allele, snp.alt_alleles[0]
            )
            seqs1_list.append(alt_1hot)

        return seqs1_list
