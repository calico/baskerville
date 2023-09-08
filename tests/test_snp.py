import os
import subprocess

import h5py
import numpy as np
import pandas as pd

from baskerville.dataset import targets_prep_strand

params_file = "tests/data/eval/params.json"
model_file = "tests/data/eval/model.h5"
targets_file = "tests/data/tiny/hg38/targets.txt"
out_dir = "tests/data/snp"
vcf_file = "/home/drk/seqnn/data/gtex_fine/susie_pip90/Kidney_Cortex_pos.vcf"
fasta_file = "%s/assembly/ucsc/hg38.fa" % os.environ["HG38"]
stat_keys = ["logSAD", "logD2"]

def test_snp():
    test_out_dir = f"{out_dir}/full"
    scores_file = f"{test_out_dir}/scores.h5"
    if os.path.isfile(scores_file):
        os.remove(scores_file)
    
    cmd = [
        "src/baskerville/scripts/hound_snp.py",
        "-f",
        fasta_file,
        "-o",
        test_out_dir,
        "--stats",
        ",".join(stat_keys),
        "--rc",
        "-t",
        targets_file,
        params_file,
        model_file,
        vcf_file
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    with h5py.File(scores_file, "r") as scores_h5:
        for sk in stat_keys:
            score = scores_h5[sk][:]
            score_var = score.var(axis=0, dtype='float32')
            assert (score_var> 0).all()


def test_slice():
    test_full_dir = f"{out_dir}/full"
    test_slice_dir = f"{out_dir}/sub"
    os.makedirs(test_slice_dir, exist_ok=True)
    scores_file = f"{test_slice_dir}/scores.h5"
    if os.path.isfile(scores_file):
        os.remove(scores_file)

    # slice targets
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)
    rna_mask = np.array([desc.startswith("RNA") for desc in targets_df.description])
    targets_rna_df = targets_df[rna_mask]
    targets_rna_file = f"{test_slice_dir}/targets_rna.txt"
    targets_rna_df.to_csv(targets_rna_file, sep="\t")

    cmd = [
        "src/baskerville/scripts/hound_snp.py",
        "-f",
        fasta_file,
        "-o",
        test_slice_dir,
        "--rc",
        "--stats",
        ",".join(stat_keys),
        "-t",
        targets_rna_file,
        params_file,
        model_file,
        vcf_file
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # stranded mask
    targets_strand_df = targets_prep_strand(targets_df)
    rna_strand_mask = np.array([desc.startswith("RNA") for desc in targets_strand_df.description])

    for sk in stat_keys:         
        with h5py.File(f"{test_full_dir}/scores.h5", "r") as scores_h5:
            score_full = scores_h5[sk][:].astype('float32')
            score_full = score_full[...,rna_strand_mask]
        with h5py.File(f"{test_slice_dir}/scores.h5", "r") as scores_h5:
            score_slice = scores_h5[sk][:].astype('float32')
        assert np.allclose(score_full, score_slice)