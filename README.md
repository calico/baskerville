# baskerville

A PyTorch implementation of Baskerville (Borzoi), a deep learning model for predicting regulatory activity from DNA sequences.

## Overview

Baskerville is a convolutional neural network architecture designed to predict chromatin accessibility and gene expression from DNA sequences. This implementation provides a PyTorch version.

## Installation

Requires Python 3.11+ and, for gene and SNP scoring, `bedtools` on `PATH`
(e.g. `apt install bedtools` or `conda install -c bioconda bedtools`).

```bash
git clone https://github.com/calico/baskerville.git
cd baskerville
pip install .
```

For a development setup, install in editable mode with the `dev` extras (and the
`cuda` extra on a GPU host):

```bash
pip install -e ".[dev]"        # add ,cuda for mamba-ssm on GPU hosts
```

The `*_folds.py` scripts run jobs locally by default, or on GCP Batch with
`--backend gcp` (see [GCP configuration](#gcp-configuration)). `--backend slurm`
needs `slurmrunner`, which is not publicly available.

The `hdwig` extra ([calico/hdwig](https://github.com/calico/hdwig), not yet
public) is needed only to read `.hw` coverage files in `hound_data`.

## Quickstart

Download one pretrained Borzoi replicate and check that it reproduces the
published forward pass (CPU, a few minutes). `hound_verify` reads the weights
from the clone, so install it editable (`pip install -e .`):

```bash
cd releases/borzoi
./download.sh human 0     # -> models_human/f0c0/model_best.pth
hound_verify --family borzoi --species human
```

See [Model releases](#model-releases) for all weights, and the guides below for
training and scoring.

## Training, Attribution & Evaluation

For detailed instructions on dataset construction, training, and evaluating models, see:

**Training**

- [Training models](docs/train.md)
- [Cross‑fold training](docs/train_folds.md)
- [Transfer learning from a pretrained model](docs/transfer.md)
- [Target distillation](docs/distill.md)
- [Distillation training (`hound_train_distill`)](docs/train_distill.md)

**Attribution**

- [Gradients & attribution](docs/gradients.md)
- [In silico mutagenesis (ISM)](docs/ism.md)

**SNP analysis**

- [SNP analysis](docs/snps.md)
- [SNP score naming](docs/snp_scores.md)
- [SNP dashboard](docs/dash_snp.md)

**Reference**

- [Targets table](docs/targets.md)
- [Updating batch-norm statistics](docs/updatenorm.md)
- [GCP Batch execution (gcprunner)](docs/gcprunner.md)

## GCP Batch Execution

Training, evaluation, SNP scoring, and ISM fold scripts support Google Cloud Batch
via `--backend gcp`. Gradient and distillation fold scripts support local and
Slurm execution.

### GCP configuration

There are no built-in GCP resources: before using `--backend gcp`, point the
runner at your own project and bucket with these environment variables (or the
matching `--gcp_*` flags). Commands fail fast with a clear error if a required
one is unset.

| Variable                  | Required | Meaning                                                                                       |
| ------------------------- | -------- | --------------------------------------------------------------------------------------------- |
| `GCPRUNNER_PROJECT`       | yes      | GCP project that runs the Batch jobs (`--gcp_project`)                                        |
| `GCPRUNNER_CACHE_PREFIX`  | yes      | `gs://` prefix for content-addressed staged inputs                                            |
| `GCPRUNNER_OUTPUT_PREFIX` | yes      | `gs://` prefix for run outputs (unless `--gcp_output_dir` is given)                           |
| `GCPRUNNER_REGION`        | no       | Batch region; defaults to `us-central1` (`--gcp_region`)                                      |
| `GCPRUNNER_IMAGE_BASE`    | no       | Image repo; defaults to `<region>-docker.pkg.dev/<project>/baskerville/baskerville`           |
| `GCPRUNNER_IMAGE`         | no       | Image tag or full URI (`--gcp_image`); else the newest `commit-<sha>` image built from `main` |

```bash
export GCPRUNNER_PROJECT=my-gcp-project
export GCPRUNNER_CACHE_PREFIX=gs://my-bucket/cache
export GCPRUNNER_OUTPUT_PREFIX=gs://my-bucket/output
```

Build the image from [dockerfiles/baskerville.Dockerfile](dockerfiles/baskerville.Dockerfile)
and push it to that repo, or pass `--gcp_image`. Tag it
`commit-$(git rev-parse HEAD)`: the default image and `--gcp_branch` only find
images tagged that way. See
[docs/gcprunner.md §8](docs/gcprunner.md#8-setup-checklist) for the one-time
project, bucket, and image setup.

### Example Usage

```bash
hound_snp_folds \
    --backend gcp \
    --gcp_output_dir gs://my-bucket/runs/2026-05-snp \
    -q l4 \
    -j 1024 \
    -p 64 \
    -f hg38.fa \
    -t targets.txt \
    params.json models/borzoi vcf.vcf
```

To run the latest built image from a specific git branch, pass its name (the
image is resolved to that branch's newest built commit and digest-pinned):

```bash
    --gcp_branch my-feature
```

To pin an arbitrary image instead, pass a full URI, a custom tag name, or a
numeric build id via `--gcp_image` (which overrides `--gcp_branch`):

```bash
    --gcp_image my-tag
```

For complete setup instructions (IAM, registry configurations, etc.), see [GCP Batch execution](docs/gcprunner.md).

## Model releases

Pretrained weights for two published model families are distributed under
[`releases/`](releases/).

- **[Borzoi](releases/borzoi/)**
  **524 kb** input sequences.
- **[Borzoi Prime](releases/borzoi_prime/)**

Each family ships weights for **4 replicates × {human, mouse}**
(`f{0..3}c0/model_best.pth`) plus a shared `params.json`, hosted in a public GCS
bucket. See the per-family README for the exact download commands and verification
instructions.

## Citation

If you use this implementation, please cite the Borzoi and/or Borzoi Prime papers:

```bibtex
@article{linder2025predicting,
  title={Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation},
  author={Linder, Johannes and Srivastava, Divyanshi and Yuan, Han and Agarwal, Vikram and Kelley, David R},
  journal={Nature Genetics},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group US New York},
  doi={10.1038/s41588-024-02053-6}
}
```

```bibtex
@article{linder2025predictingcelltype,
  title={Predicting cell type-specific coverage profiles from DNA sequence},
  author={Linder, Johannes and Yuan, Han and Kelley, David R},
  journal={bioRxiv},
  pages={2025--06},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contributing

This repository accompanies the Borzoi and Borzoi Prime papers. Bug reports and
questions are welcome as GitHub issues; pull requests are reviewed on a
best-effort basis.

To run the checks CI runs:

```bash
pip install -e ".[dev]"
pytest -m "not slow"
uvx ruff@0.15 format --check .
npx prettier@3.8.2 --check .
```

## License

This project is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
Third-party code and its licenses are listed in [NOTICE](NOTICE).
