# Borzoi PyTorch — Predicting RNA-seq from DNA Sequence

PyTorch port of Borzoi, ported from the original TensorFlow version at
[github.com/calico/borzoi](https://github.com/calico/borzoi).

Borzoi is a convolutional neural network trained to predict RNA-seq coverage
at **32 bp resolution** given **524 kb** input sequences. The model is
described in the following [manuscript](https://www.nature.com/articles/s41588-024-02053-6)

Borzoi was trained on a large set of RNA-seq experiments from ENCODE and
GTEx, as well as re-processed versions of the original Enformer training data
(ChIP-seq and DNase from ENCODE, ATAC-seq from CATlas, and CAGE from FANTOM5).
Full target lists in this directory.
The `file` column records where each track was originally processed, not a
path you are expected to have. It is used only to rebuild training data with
`hound_data`. The processed training data is public in the requester-pays bucket
`gs://borzoi-paper/data/` (see the
[calico/borzoi README](https://github.com/calico/borzoi#data-availability)).

## Model weights

Weights for 4 folds × {human, mouse} are hosted in a public GCS bucket.
`params.json` and `targets.txt` are shared across folds and live alongside the fold directories.

```
gs://seqnn-share/borzoi_pt/models_human/params.json
gs://seqnn-share/borzoi_pt/models_human/targets.txt
gs://seqnn-share/borzoi_pt/models_human/f{0..3}c0/model_best.pth

gs://seqnn-share/borzoi_pt/models_mouse/params.json
gs://seqnn-share/borzoi_pt/models_mouse/targets.txt
gs://seqnn-share/borzoi_pt/models_mouse/f{0..3}c0/model_best.pth
```

Use the download script (requires only `curl`):

```sh
# all human folds into ./
./download.sh

# specific species / fold / destination
./download.sh human 0 /data/models
./download.sh mouse all /data/models
```

Or grab files directly:

```sh
curl -fLO https://storage.googleapis.com/seqnn-share/borzoi_pt/models_human/f0c0/model_best.pth
```

## Verify your download

After downloading, confirm the weights load and reproduce the published forward
pass. Run `download.sh` from inside this directory (the default) so the weights
land at `models_<species>/f<n>c0/model_best.pth`, then:

```sh
python -m baskerville.scripts.hound_verify --family borzoi
```

It runs a deterministic CPU forward for every fold you downloaded and checks the
output against the committed reference (Pearson r and relative RMSE):

```
verify: 100%|████████████████████| 4/4 [02:48<00:00]

family       species fold          r    rmse_rel  result
-------------------------------------------------------
borzoi       human   f0     1.000000   0.000e+00  PASS
borzoi       human   f1     1.000000   0.000e+00  PASS
borzoi       mouse   f0     1.000000   0.000e+00  PASS
borzoi       mouse   f1     1.000000   0.000e+00  PASS
-------------------------------------------------------
PASS
```

A `FAIL` or `ERROR` means the downloaded `model_best.pth` is corrupt/incomplete,
or the code no longer matches the published architecture. If you have not
downloaded any weights, the same command instead checks the architecture itself.
Exit code is non-zero on any failure.
