# Baskerville

#### Sequential regulatory activity predictions with deep convolutional neural networks.

Baskerville provides researchers with tools to:

1. Train deep convolutional neural networks to predict regulatory activity along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the specific nucleotides that drive regulatory element function.

---

### Documentations

Documentation page: https://calico.github.io/baskerville/index.html

- [Document page for transfer learning to hg38 tracks](docs/transfer_human/transfer.md)
- [Document page for transfer learning to mm10 tracks](docs/transfer_mouse/transfer_mouse.md)

---

### Installation

`git clone git@github.com:calico/baskerville.git`
`cd baskerville`
`pip install .`

To set up the required environment variables:
`cd baskerville`
`conda activate <conda_env>`
`./env_vars.sh`

*Note:* Change the two lines of code at the top of './env_vars.sh' to the correct local paths.

Alternatively, the environment variables can be set manually:
```sh
export BASKERVILLE_DIR=/home/<user_path>/baskerville
export PATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PATH
export PYTHONPATH=$BASKERVILLE_DIR/src/baskerville/scripts:$PYTHONPATH

export BASKERVILLE_CONDA=/home/<user>/anaconda3/etc/profile.d/conda.sh
```

---

#### Contacts

Dave Kelley (codeowner)
