# Cerberus — Bidirectional State Space Models of Regulatory Sequence

Cerberus predicts functional genomics coverage at **32 bp resolution** from
**786 kb** input sequences. A convolution tower reduces the sequence to 32 bp
positions, which then pass through 8 Hydra blocks (bidirectional Mamba-2 state
space mixers). Cerberus has no attention and no U-Net decoder. It has 138.3M
parameters (Borzoi: 185.9M), and a single forward pass takes about 0.1 s on an
A100. Each model has a human (hg38) head and a mouse (mm10) head:

| Assay      | Human | Mouse |
| ---------- | ----: | ----: |
| RNA-seq    |  2028 |   953 |
| 3′ RNA-seq |  1741 |   214 |
| CAGE       |  1678 |   720 |
| ChIP-seq   |  1282 |   560 |
| DNase-seq  |   674 |    93 |
| CLIP-seq   |   496 |     — |
| ATAC-seq   |   462 |   562 |
| **Total**  |  8361 |  3102 |

The full target lists are in `targets_human.txt` and `targets_mouse.txt`. Row
order matches the head outputs. Predictions are in transformed units; invert the
transform with `baskerville.dataset.untransform_preds` and these tables.

The training recipe is in the `train` block of `params.json`. Release of the
processed training data is pending.

## Requirements

The Hydra scan is a Triton kernel from `mamba-ssm`, so Cerberus needs an NVIDIA
GPU (compute capability 7.0+) and the `cuda` extra:

```sh
pip install -e ".[cuda]"
```

## Model weights

Cerberus is an ensemble of 8 replicate models, each trained with a different
held-out fold. Each model predicts both species, so `params.json` and the targets
tables are shared by all folds:

```
gs://seqnn-share/cerberus/models/params.json
gs://seqnn-share/cerberus/models/targets_{human,mouse}.txt
gs://seqnn-share/cerberus/models/f{0..7}c0/model_best.pth
```

Use the download script (requires only `curl`):

```sh
# all folds into ./
./download.sh

# specific fold / destination
./download.sh 0 /data/models
```

Or grab files directly:

```sh
curl -fLO https://storage.googleapis.com/seqnn-share/cerberus/models/f0c0/model_best.pth
```

For best accuracy, average predictions across the 8 folds and across forward
and reverse-complement inputs (`SeqNN.ensemble_rc = True`, or `--rc` in the
scoring scripts).

## Verify your download

After downloading, confirm the weights load and reproduce the published forward
pass. Run `download.sh` from inside this directory (the default) so the weights
land at `models/f<n>c0/model_best.pth`, then:

```sh
python -m baskerville.scripts.hound_verify --family cerberus
```

For every fold you downloaded, it runs a deterministic GPU forward pass through
both heads. It checks each output against the committed reference using Pearson
r and relative RMSE:

```
verify: 100%|████████████████████| 16/16 [01:21<00:00]

family       species fold           r    rmse_rel  result
-------------------------------------------------------
cerberus     human   f0      1.000000   0.000e+00  PASS
cerberus     human   f1      1.000000   0.000e+00  PASS
cerberus     human   f2      1.000000   0.000e+00  PASS
cerberus     human   f3      1.000000   0.000e+00  PASS
cerberus     human   f4      1.000000   0.000e+00  PASS
cerberus     human   f5      1.000000   0.000e+00  PASS
cerberus     human   f6      1.000000   0.000e+00  PASS
cerberus     human   f7      1.000000   0.000e+00  PASS
cerberus     mouse   f0      1.000000   0.000e+00  PASS
cerberus     mouse   f1      1.000000   0.000e+00  PASS
cerberus     mouse   f2      1.000000   0.000e+00  PASS
cerberus     mouse   f3      1.000000   0.000e+00  PASS
cerberus     mouse   f4      1.000000   0.000e+00  PASS
cerberus     mouse   f5      1.000000   0.000e+00  PASS
cerberus     mouse   f6      1.000000   0.000e+00  PASS
cerberus     mouse   f7      1.000000   0.000e+00  PASS
-------------------------------------------------------
PASS
```

A `FAIL` or `ERROR` means one of two things: the downloaded `model_best.pth` is
corrupt or incomplete, or the code no longer matches the published architecture.
If you have not downloaded any weights, the same command checks the architecture
itself. The exit code is non-zero on any failure.
