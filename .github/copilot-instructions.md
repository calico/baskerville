# Project Overview

This project, named baskerville, contains tools for data processing for genomic language models, a PyTorch-based approach for training genomic language models, and downstream inference.

As input, our models generally take local DNA sequences of hundreds of thousands of base pairs (often encoded as one-hots, e.g. input is shaped [4, seq_len_bp]). As an output, our models generally produce coverage track predictions for thousands of conditions (spanning combinations of biological conditions, e.g. young mouse liver male, young mouse liver female, and experimental assay conditions, e.g. RNA-seq, ATAC-seq, ChIP-seq (H3k27ac), etc.) encoded as continuous values and aggregated over bins of base pairs (i.e. targets shaped [num_conditions, seq_len_bin]).

After training these models, we are interested in using them for a variety of downstream tasks, e.g. (a) identifying regulators of a genomic feature in a specific biological context (e.g. identifying candidate transcription factor regulators of a gene in a disease) (b) predicting the effects of genetic variants.

## Key files for data processing

- `src/baskerville/data/dataset.py`: Contains the `SeqDataset` class, which handles loading of sequences and targets from disk, as well as data augmentations.
- `src/baskerville/scripts/hound_data.py`: Create a new dataset (that can be loaded with SeqDataset) from a targets dataframe. Targets dataframe includes paths to a coverage track file (e.g. BigWig or .w5 format) and metadata containing information to determine settings to process each target. Broken into two steps: (1) hound_data_read.py (which generates binned / processed genome-wide coverage tracks for each target) (2) hound_data_write.py (which generates training examples for local genomic windows -- containing the DNA sequences and corresponding binned coverage tracks over all of the targets / conditions -- and writes these training examples out in a Zarr format).

## Key files for model definition

- `/src/baskerville/seqnn.py`: Contains the base `SeqNNMod` class, which contains the core logic for architecture definition and forward pass logic of the genomic language model. Additionally contains the `SeqNN` class, a wrapper class that handles forward passes that ensemble over stochastic augmentations and gradient computation logic, in addition to other features.
- `src/baskerville/blocks.py`: Defines high-level building blocks for seqnn.py, such as TransformerTower's, ConvTower's etc. The building blocks are accessible via blocks.name_module["module_name"].
- `src/baskerville/layers.py`: Contains low-level building blocks used by blocks.py

## Key files for model training/evaluation

- `src/baskerville/trainer.py`: Contains the `Trainer` class, which handles the instantiation of a SeqNN model, training loop, evaluation loop, checkpointing, and logging.
- `src/baskerville/scripts/hound_train.py`: Train a model on the given dataset.
- `src/baskerville/scripts/hound_eval.py`: Evaluate a trained model on the given dataset and output performance metrics.

## Cross-fold analysis

In most cases, we're training sets of replicate models with different train/test splits based on a cross-fold validation scheme. `src/baskerville/scripts/hound_data.py` will determine the cross folds. Scripts ending in \*\_folds.py launch jobs (locally, on Slurm, or on GCP Batch) for each of the available replicate models. For example:

- `src/baskerville/scripts/hound_train_folds.py`: Create cross-validation train/test splits via symbolic links and launch training of models on specified folds.
- `src/baskerville/scripts/hound_snp_folds.py`: Launch jobs to compute SNP scores with all replicate models.

## Key files for model inference

- `src/baskerville/scripts/hound_grad.py`: Compute saliency maps (gradients) for a set of genomic regions provided in a GTF file using a trained SeqNN model.
- `src/baskerville/scripts/hound_ism_bed.py`: Perform in-silico mutagenesis (ISM) on a set of genomic regions provided in a BED file using a trained SeqNN model.
- `src/baskerville/scripts/hound_snp.py`: Assess the predicted effects of genetic variants using a trained SeqNN model.

## Reverse complements

Gene regulation is agnostic to strand, and it's equally valid to model the data as is on the forward strand as well as reversed on the reverse strand. During training, we treat reverse complement as a data augmentation strategy and half of steps will reverse complement the DNA and reverse the tracks. Importantly, some of the tracks are stranded where one track represents the 5' and another represents the 3'. These are matched via the 'strand_pair' attribute. When you reverse complement the DNA, you have to swap these strand_pairs. During inference, we generally ensemble predictions for the forward strand with reversed predictions for the reverse strand.

## Other key folders and directories

- `/docs`: Contains documentation for the project, including API specifications and user guides.
- `/src/tests`: Contains unit tests and integration tests for the various components of the project.

## Note on implementing new features

When implementing new features we intend to make the minimal set of changes needed and require documentation. After implementing the new feature, tests should be added to `/src/tests` to ensure the new feature works as intended and does not break existing functionality. Tests should be run with Pytest, and can be run with the command `pytest src/tests -v` from the root directory of the project. These tests should ideally be fast and runnable on a local machine. After confirming tests pass, if warranted the new feature can be added to the documentation in `/docs`.
