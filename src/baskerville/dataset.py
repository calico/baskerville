import glob
import json
import multiprocessing

from natsort import natsorted
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import dok_matrix
import zarr

import torch
from torch.utils.data import Dataset, Sampler

from baskerville import dna
from baskerville.types import BatchData
from baskerville.deprecated.dataset import *


def discover_num_folds(data_dir: str) -> int | None:
    """Return the number of ``examples/fold*.zarr`` files, or ``None`` if none exist."""
    fold_zarrs = glob.glob(str(Path(data_dir) / "examples" / "fold*.zarr"))
    return len(fold_zarrs) if fold_zarrs else None


def compute_fold_splits(num_folds: int, fold: int, cross: int = 0) -> dict:
    """Return train/valid/test fold indices for the given (fold, cross) replicate."""
    if not (0 <= fold < num_folds):
        raise ValueError(f"fold={fold} out of range [0, {num_folds})")
    test_fold = fold
    valid_fold = (fold + 1 + cross) % num_folds
    train_folds = [i for i in range(num_folds) if i not in (test_fold, valid_fold)]
    return {"test": test_fold, "valid": valid_fold, "train": train_folds}


def _resolve_zarr_files(
    data_dir: str, split_label: str, *, fold: int | None, cross: int
) -> list[str]:
    """Resolve the zarr file paths backing a given dataset split.

    Two modes:
    - **folds mode** (``fold`` is set): requires ``examples/fold*.zarr`` to be present.
      ``split_label`` ∈ {"train", "valid", "test", "*"} resolves to fold indices via
      :func:`compute_fold_splits`. An ``examples/free.zarr`` is appended to "train"
      and "*" if present.
    - **legacy mode** (``fold`` is None): preserves the original glob behavior —
      ``"train"``/``"*"`` glob ``{split_label}*.zarr``; anything else maps to a single
      ``{split_label}.zarr``. Used by callers that address folds directly via
      ``split_label="foldN"`` (e.g. ``hound_eval --split fold3``).
    """
    examples_dir = Path(data_dir) / "examples"

    if fold is None:
        if split_label in ("train", "*"):
            return natsorted(glob.glob(str(examples_dir / f"{split_label}*.zarr")))
        return [str(examples_dir / f"{split_label}.zarr")]

    fold_zarrs = natsorted(glob.glob(str(examples_dir / "fold*.zarr")))
    if not fold_zarrs:
        raise ValueError(
            f"fold={fold} requested but no fold*.zarr files found in {examples_dir}"
        )
    num_folds = len(fold_zarrs)
    splits = compute_fold_splits(num_folds, fold, cross)
    free_zarr = examples_dir / "free.zarr"

    if split_label == "train":
        files = [str(examples_dir / f"fold{i}.zarr") for i in splits["train"]]
        if free_zarr.exists():
            files.append(str(free_zarr))
        return files
    if split_label == "valid":
        return [str(examples_dir / f"fold{splits['valid']}.zarr")]
    if split_label == "test":
        return [str(examples_dir / f"fold{splits['test']}.zarr")]
    if split_label == "*":
        files = list(fold_zarrs)
        if free_zarr.exists():
            files.append(str(free_zarr))
        return files
    return [str(examples_dir / f"{split_label}.zarr")]


class MultiDataset(Dataset):
    """Combine multiple datasets and manage index selection."""

    def __init__(self, datasets):
        self.datasets = datasets

        # calculate cumulative lengths for mapping indices
        self.cumulative_lengths = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int):
        # find which dataset this index belongs to
        dataset_idx = 0
        while (
            dataset_idx < len(self.datasets)
            and idx >= self.cumulative_lengths[dataset_idx]
        ):
            dataset_idx += 1

        # calculate the local index within the dataset
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Return the item and which dataset it came from (1-based index)
        return dataset_idx, self.datasets[dataset_idx][local_idx]


class SeqDataset(Dataset):
    """Labeled sequence dataset.

    Args:
      data_dir (str): Dataset directory.
      split_label (str): Dataset split, e.g. train, valid, test.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      targets_file (str): Targets table.
      extra_crop_bp (int): Optional input cropping in basepairs.
      extra_crop_bins (int): Optional output cropping in bins.
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        mode: str = "eval",
        targets_file: str = None,
        extra_crop_bp: int = None,
        extra_crop_bins: int = None,
        skip_genes: bool = False,
        fold: int | None = None,
        cross: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_label = split_label
        self.mode = mode
        self.extra_crop_bp = extra_crop_bp
        self.extra_crop_bins = extra_crop_bins
        self.skip_genes = skip_genes
        self.fold = fold
        self.cross = cross

        # read data parameters
        data_stats_file = f"{self.data_dir}/statistics.json"
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats["seq_length"]
        self.target_length = data_stats.get("target_length", 0)
        self.num_targets = data_stats.get("num_targets", 0)
        self.pool_width = data_stats.get("pool_width", 1)
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.data_crop_bp = data_stats.get("crop_bp", 0)
        self.has_coverage = self.num_targets > 0

        # update sequence statistics based on optional cropping
        if self.extra_crop_bp is not None:
            self.seq_length -= 2 * self.extra_crop_bp
        elif self.extra_crop_bins is not None:
            self.target_length -= 2 * self.extra_crop_bins

        # read targets (conditional on coverage data)
        if self.has_coverage:
            if targets_file is None:
                targets_file = f"{self.data_dir}/targets.txt"
            self.targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")

            # set strand pairs (using new indexing)
            self.strand_pair = strand_pair_indices(self.targets_df)
        else:
            self.targets_df = pd.DataFrame()
            self.strand_pair = None

        # collect fold zarr files
        self.zarr_files = _resolve_zarr_files(
            self.data_dir, self.split_label, fold=fold, cross=cross
        )
        # open zarr temporarily for init-time metadata reads;
        # don't persist handles (not fork-safe with DataLoader workers)
        zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]

        # per-track histograms of the stored targets (write_target_hist)
        self.target_hist = None
        if self.has_coverage and "target_hist" in zarr_data[0]:
            self.target_hist = zarr_data[0]["target_hist"][:]

        # count sequences
        zarr_seqs = [zd["sequence"].shape[0] for zd in zarr_data]
        self.zarr_cumseqs = np.append(0, np.cumsum(zarr_seqs))

        # auto-detect gene data unless explicitly skipped
        self.has_genes = False
        if not self.skip_genes:
            self.has_genes = "gene_target" in zarr_data[0]
        self.zarr_data = None
        if self.has_genes:
            # Read gene metadata files
            genes_file = f"{self.data_dir}/genes.txt"
            if Path(genes_file).exists():
                self.genes_df = pd.read_csv(genes_file, sep="\t")

            targets_gene_file = f"{self.data_dir}/targets_gene.txt"
            if Path(targets_gene_file).exists():
                self.targets_gene_df = pd.read_csv(
                    targets_gene_file, sep="\t", index_col=0
                )
                self.num_gene_targets = len(self.targets_gene_df)
            else:
                self.num_gene_targets = 0

    def _open_zarr(self):
        """Open zarr stores (called lazily so each DataLoader worker gets its own handles)."""
        self.zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]

    def __len__(self):
        return self.zarr_cumseqs[-1]

    def __getitem__(self, idx):
        if self.zarr_data is None:
            self._open_zarr()

        # determine zarr indexes
        zi = np.searchsorted(self.zarr_cumseqs[1:], idx, side="right")
        si = idx - self.zarr_cumseqs[zi]

        # read example
        seq_indexes = self.zarr_data[zi]["sequence"][si]

        # read coverage targets if available
        targets = None
        if self.has_coverage:
            targets = self.zarr_data[zi]["target"][si]

        # read gene data if available
        if self.has_genes:
            gene_targets = self.zarr_data[zi]["gene_target"][si]
            gene_presence = self.zarr_data[zi]["gene_presence"][si]
            gene_out_mask = self.zarr_data[zi]["gene_out_mask"][si]

            # Error if sequence has no valid genes for gene-only datasets
            # (should be filtered during data construction)
            if not gene_presence.any() and not self.has_coverage:
                raise ValueError(
                    f"Sequence {idx} has no valid genes. Gene-only datasets should "
                    "be filtered during data construction with hound_data."
                )

        # optionally crop (only if crop amount > 0)
        if self.extra_crop_bp is not None and self.extra_crop_bp > 0:
            seq_indexes = seq_indexes[..., self.extra_crop_bp : -self.extra_crop_bp]
        if (
            self.has_coverage
            and self.extra_crop_bins is not None
            and self.extra_crop_bins > 0
        ):
            targets = targets[..., self.extra_crop_bins : -self.extra_crop_bins]

        # targets to torch
        if self.has_coverage:
            targets = torch.from_numpy(targets)

        # gene data to torch
        if self.has_genes:
            gene_targets = torch.from_numpy(gene_targets)
            gene_presence = torch.from_numpy(gene_presence)
            gene_out_mask = torch.from_numpy(gene_out_mask)

        # stochastic augmentations
        if self.mode == "train":
            # shift
            shift = np.random.choice(np.arange(-1, 2))
            seq_indexes = dna.seqi_shift(seq_indexes, shift, self.seq_depth)

            # reverse complement
            ex_rc = np.random.choice([True, False])
            if ex_rc:
                seq_indexes = dna.seqi_rc(seq_indexes)
                if self.has_coverage:
                    targets = torch.flip(targets[self.strand_pair], [1])
                if self.has_genes:
                    gene_out_mask = torch.flip(gene_out_mask, [-1])

        # sequence to 1-hot torch
        seq_1hot = np.zeros((self.seq_depth + 1, self.seq_length), dtype="bool")
        seq_1hot[seq_indexes, np.arange(self.seq_length)] = 1
        seq_1hot = torch.from_numpy(seq_1hot[:-1]).half()

        return BatchData(
            sequence=seq_1hot,
            coverage_targets=targets,
            gene_targets=gene_targets if self.has_genes else None,
            gene_presence=gene_presence if self.has_genes else None,
            gene_out_mask=gene_out_mask if self.has_genes else None,
        )


class SeqDatasetMLM(Dataset):
    """Sequence dataset for Masked Language Model training.

    Args:
      data_dir (str): Dataset directory.
      split_label (str): Dataset split, e.g. train, valid, test.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      has_mask (bool): Whether to load exon masks. Defaults to False.
      has_repeat_mask (bool): Whether to load repeat masks. Defaults to False.
      augment_rc (bool): Apply random reverse complement augmentation. Defaults to True.
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        mode: str = "eval",
        has_mask: bool = False,
        has_repeat_mask: bool = False,
        augment_rc: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_label = split_label
        self.mode = mode
        self.has_mask = has_mask
        self.has_repeat_mask = has_repeat_mask
        self.augment_rc = augment_rc

        # read data parameters
        data_stats_file = f"{self.data_dir}/statistics.json"
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats["seq_length"]
        self.target_length = data_stats.get("target_length", self.seq_length)
        self.num_targets = data_stats.get("num_targets", 4)  # 4 nucleotides for MLM
        self.pool_width = data_stats.get("pool_width", 1)
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.num_species = data_stats.get("num_species", 1)

        # collect fold zarr files
        if self.split_label in ["train", "*"]:
            examples_path = (
                Path(self.data_dir) / "examples" / f"{self.split_label}*.zarr"
            )
            self.zarr_files = natsorted(glob.glob(str(examples_path)))
        else:
            examples_path = (
                Path(self.data_dir) / "examples" / f"{self.split_label}.zarr"
            )
            self.zarr_files = [str(examples_path)]
        # count sequences
        # (but, don't persist handles past init--not fork-safe
        zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]
        zarr_seqs = [zd["sequence"].shape[0] for zd in zarr_data]
        self.zarr_cumseqs = np.append(0, np.cumsum(zarr_seqs))
        self.zarr_data = None

    def _open_zarr(self):
        self.zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]

    def __len__(self):
        return self.zarr_cumseqs[-1]

    def __getitem__(self, idx):
        if self.zarr_data is None:
            self._open_zarr()

        zi = np.searchsorted(self.zarr_cumseqs[1:], idx, side="right")
        si = idx - self.zarr_cumseqs[zi]

        seq_indexes = self.zarr_data[zi]["sequence"][si]

        if "label" in self.zarr_data[zi]:
            label = self.zarr_data[zi]["label"][si]
            label = torch.from_numpy(label).float()
            # ensure label is (1, num_species) shape
            if label.dim() == 1:
                label = label.unsqueeze(0)
        else:
            # default to first species if no labels
            label = torch.zeros((1, self.num_species))
            label[0, 0] = 1.0

        exon_mask = None
        repeat_mask = None
        if self.has_mask and "mask" in self.zarr_data[zi]:
            exon_mask = self.zarr_data[zi]["mask"][si]
            exon_mask = torch.from_numpy(exon_mask).float()
        if self.has_repeat_mask and "repeat_mask" in self.zarr_data[zi]:
            repeat_mask = self.zarr_data[zi]["repeat_mask"][si]
            repeat_mask = torch.from_numpy(repeat_mask).float()

        seq_1hot = np.zeros((self.seq_depth + 1, self.seq_length), dtype="bool")
        seq_1hot[seq_indexes, np.arange(self.seq_length)] = 1
        seq_1hot = torch.from_numpy(seq_1hot[:-1]).float()

        if self.mode == "train" and self.augment_rc:
            if np.random.random() < 0.5:
                seq_1hot = torch.flip(seq_1hot, dims=[1])
                # swap A<->T, C<->G
                seq_1hot[[0, 3], :] = seq_1hot[[3, 0], :]
                seq_1hot[[1, 2], :] = seq_1hot[[2, 1], :]

                if exon_mask is not None:
                    exon_mask = torch.flip(exon_mask, dims=[0])
                if repeat_mask is not None:
                    repeat_mask = torch.flip(repeat_mask, dims=[0])

        # zeros fallback when a mask is requested but absent from the zarr,
        # preserving prior behavior; None when the mask is not requested at all.
        if self.has_mask and exon_mask is None:
            exon_mask = torch.zeros(self.seq_length)
        if self.has_repeat_mask and repeat_mask is None:
            repeat_mask = torch.zeros(self.seq_length)

        return BatchData(
            sequence=seq_1hot,
            species_label=label,
            exon_mask=exon_mask if self.has_mask else None,
            repeat_mask=repeat_mask if self.has_repeat_mask else None,
        )


class SeqDatasetTeacher(Dataset):
    """Sequence dataset with on-the-fly teacher predictions for distillation.

    This dataset loads sequences from data_dir and computes targets on-the-fly
    using an ensemble of teacher models. This is useful for knowledge distillation
    where we want to train a student model using soft targets from teacher models.

    Args:
      data_dir (str): Dataset directory containing sequences.
      split_label (str): Dataset split, e.g. train, valid, test.
      teachers (list): List of SeqNN teacher models for computing predictions.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      targets_file (str): Targets table.
      head (int | None): Model head index to use for predictions. None (default) uses
        dataset-specific heads, int forces a specific head, -1 concatenates all heads.
      teacher_subset (int): If specified, randomly sample this many teachers for each
        prediction instead of using all teachers. Useful for computational efficiency
        and regularization. Defaults to None (use all teachers).
      snp_rate (float): Rate of nucleotides to randomly mutate (0.0-1.0).
        If specified, introduces random SNPs before computing teacher predictions.
        Useful for data augmentation and robustness training. Defaults to 0.0 (no mutations).
      del_rate (float): Rate of deletions to randomly introduce (0.0-1.0).
        If specified, introduces random deletions before computing teacher predictions.
        Useful for data augmentation and robustness training. Defaults to 0.0 (no deletions).
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        teachers: list,
        mode: str = "eval",
        targets_file: str = None,
        head: int | None = None,
        teacher_subset: int = None,
        snp_rate: float = 0.0,
        del_rate: float = 0.0,
        fold: int | None = None,
        cross: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_label = split_label
        self.teachers = teachers
        self.mode = mode
        self.head = head
        self.teacher_subset = teacher_subset
        self.snp_rate = snp_rate
        self.del_rate = del_rate
        self.fold = fold
        self.cross = cross

        # validate teacher_subset
        if self.teacher_subset is not None:
            if self.teacher_subset < 1:
                raise ValueError(
                    f"teacher_subset must be >= 1, got {self.teacher_subset}"
                )
            if self.teacher_subset > len(self.teachers):
                raise ValueError(
                    f"teacher_subset ({self.teacher_subset}) cannot be larger than "
                    f"number of teachers ({len(self.teachers)})"
                )

        # validate mutation rates
        if self.snp_rate < 0.0 or self.snp_rate > 1.0:
            raise ValueError(f"snp_rate must be between 0 and 1, got {self.snp_rate}")
        if self.del_rate < 0.0 or self.del_rate > 1.0:
            raise ValueError(f"del_rate must be between 0 and 1, got {self.del_rate}")

        # read data parameters
        data_stats_file = f"{self.data_dir}/statistics.json"
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats["seq_length"]
        self.target_length = data_stats["target_length"]
        self.num_targets = data_stats["num_targets"]
        self.pool_width = data_stats["pool_width"]
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.data_crop_bp = data_stats.get("crop_bp", 0)
        self.has_coverage = self.num_targets > 0
        self.has_genes = False  # Teacher distillation doesn't use gene targets

        # read targets
        if targets_file is None:
            targets_file = f"{self.data_dir}/targets.txt"
        self.targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")

        # set strand pairs (using new indexing)
        self.strand_pair = strand_pair_indices(self.targets_df)

        # collect fold zarr files
        self.zarr_files = _resolve_zarr_files(
            self.data_dir, self.split_label, fold=fold, cross=cross
        )

        # open zarr temporarily for init-time metadata reads;
        # don't persist handles (not fork-safe with DataLoader workers)
        zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]

        # count sequences
        zarr_seqs = [zd["sequence"].shape[0] for zd in zarr_data]
        self.zarr_cumseqs = np.append(0, np.cumsum(zarr_seqs))
        self.zarr_data = None

        # set all teacher models to eval mode
        for teacher in self.teachers:
            teacher.model.eval()

    def _open_zarr(self):
        """Open zarr stores (called lazily so each DataLoader worker gets its own handles)."""
        self.zarr_data = [zarr.open(zf, mode="r") for zf in self.zarr_files]

    def __len__(self):
        return self.zarr_cumseqs[-1]

    def __getitem__(self, idx):
        if self.zarr_data is None:
            self._open_zarr()

        # determine zarr indexes
        zi = np.searchsorted(self.zarr_cumseqs[1:], idx, side="right")
        si = idx - self.zarr_cumseqs[zi]

        # read example sequence
        seq_indexes = self.zarr_data[zi]["sequence"][si]

        # stochastic augmentations
        if self.mode == "train":
            # shift
            shift = np.random.choice(np.arange(-1, 2))
            seq_indexes = dna.seqi_shift(seq_indexes, shift, self.seq_depth)

            # reverse complement
            ex_rc = np.random.choice([True, False])
            if ex_rc:
                seq_indexes = dna.seqi_rc(seq_indexes)

            # mutate
            mutate_sequence(seq_indexes, self.snp_rate, self.del_rate)

        # sequence to 1-hot torch
        seq_1hot = np.zeros((self.seq_depth + 1, self.seq_length), dtype="bool")
        seq_1hot[seq_indexes, np.arange(self.seq_length)] = 1
        seq_1hot = torch.from_numpy(seq_1hot[:-1]).half()

        return seq_1hot

    def collate_fn(self, batch):
        """Custom collate function that performs batched teacher inference."""
        sequences = []
        dataset_indices = []

        for item in batch:
            # Check if this is MultiDataset format
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                # MultiDataset format: (dataset_idx, seq)
                dataset_idx, seq_1hot = item
                dataset_indices.append(dataset_idx)
                sequences.append(seq_1hot)
            else:
                # Direct format: just seq
                sequences.append(item)
                dataset_indices.append(0)

        # Stack into batch tensor
        seq_batch = torch.stack(sequences)

        # Compute teacher predictions on the entire batch
        with torch.no_grad():
            # Select subset of teachers if specified
            if self.teacher_subset is not None:
                teacher_indices = np.random.choice(
                    len(self.teachers), size=self.teacher_subset, replace=False
                )
                selected_teachers = [self.teachers[i] for i in teacher_indices]
            else:
                selected_teachers = self.teachers

            # Move batch to device
            device = selected_teachers[0].device
            seq_device = seq_batch.to(device)

            if self.head is None:
                # Use dataset-specific head (all sequences have same dataset index)
                hi = dataset_indices[0]
            else:
                # Use specified head (either specific int or -1 for all heads)
                hi = self.head

            # Collect predictions from selected teachers
            targets = None
            for teacher in selected_teachers:
                output = teacher(seq_device, hi=hi)
                pred = output.coverage  # Use coverage predictions for distillation
                if targets is None:
                    targets = pred
                else:
                    targets += pred

            # Average (divide by number of teachers)
            targets = targets / len(selected_teachers)

        # Return dataset indices and data, matching the format expected by Trainer
        return torch.tensor(dataset_indices), (seq_batch, targets)


class MultiSampler(Sampler):
    """Sampler to ensure that batches are drawn from the same dataset."""

    def __init__(self, multi_dataset, batch_size, upsampling_rates=None, mode="train"):
        self.multi_dataset = multi_dataset
        self.batch_size = batch_size
        self.upsampling_rates = upsampling_rates
        if self.upsampling_rates is None:
            self.upsampling_rates = [1 for _ in range(len(self.multi_dataset.datasets))]
        self.mode = mode

    def __len__(self):
        return sum(
            [
                (len(dataset) // self.batch_size) * upsampling_rate
                for dataset, upsampling_rate in zip(
                    self.multi_dataset.datasets, self.upsampling_rates
                )
            ]
        )

    def __iter__(self):
        batch_indices = []
        examples_start = 0
        for di, [dataset, upsampling_rate] in enumerate(
            zip(self.multi_dataset.datasets, self.upsampling_rates)
        ):
            # make example index list
            num_examples = len(dataset)
            examples_end = examples_start + num_examples

            # repeat (optionally)
            for _ in range(upsampling_rate):
                examples_indices = list(range(examples_start, examples_end))

                # shuffle examples
                if self.mode == "train":
                    np.random.shuffle(examples_indices)

                # truncate to batch size
                overhang = num_examples % self.batch_size
                num_examples_batch = num_examples - overhang
                examples_indices = examples_indices[:num_examples_batch]

                # form batch tuples
                examples_batches = [
                    examples_indices[i : i + self.batch_size]
                    for i in range(0, len(examples_indices), self.batch_size)
                ]
                batch_indices.extend(examples_batches)

            # update start
            examples_start = examples_end

        # shuffle dataset batches
        if self.mode == "train":
            np.random.shuffle(batch_indices)

        return iter(batch_indices)


def make_strand_transform(targets_df, targets_strand_df):
    """Make a sparse matrix to sum strand pairs.

    Args:
        targets_df (pd.DataFrame): Targets DataFrame.
        targets_strand_df (pd.DataFrame): Targets DataFrame, with strand pairs collapsed.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix to sum strand pairs.
    """

    # initialize sparse matrix
    strand_transform = dok_matrix((targets_df.shape[0], targets_strand_df.shape[0]))

    # fill in matrix
    ti = 0
    sti = 0
    for _, target in targets_df.iterrows():
        strand_transform[ti, sti] = True
        if target.strand_pair == target.name:
            sti += 1
        else:
            if target.identifier[-1] == "-":
                sti += 1
        ti += 1
    strand_transform = strand_transform.tocsr()

    return strand_transform


def mutate_sequence(seq_indexes, snp_rate, del_rate, max_del_bp=10):
    """Mutate a sequence with SNPs and deletions in-place.

    Args:
        seq_indexes (np.array): Index-encoded sequence (0=A, 1=C, 2=G, 3=T, 4=N).
        snp_rate (float): Rate of nucleotides to randomly mutate (0.0-1.0).
        del_rate (float): Rate of deletions to introduce (0.0-1.0).
        max_del_bp (int): Maximum size of deletions in base pairs. Defaults to 10.
    """
    seq_len = len(seq_indexes)

    # Apply SNPs
    if snp_rate > 0:
        # Adjust rate to account for cases where random nucleotide matches original
        adjusted_snp_rate = snp_rate * 4.0 / 3.0

        # Determine which positions to mutate
        snp_mask = np.random.random(seq_len) < adjusted_snp_rate
        num_snps = snp_mask.sum()

        if num_snps > 0:
            # Sample replacement nucleotides
            snp_replacements = np.random.randint(0, 4, num_snps)
            seq_indexes[snp_mask] = snp_replacements

    # Apply deletions
    if del_rate > 0:
        max_del_bp = min(max_del_bp, seq_len // 2)
        max_start_pos = seq_len - max_del_bp

        # Determine which positions get a deletion
        del_mask = np.random.random(max_start_pos) < del_rate
        del_positions = np.where(del_mask)[0]

        # Sort positions in reverse order to avoid index shifting issues
        for pos in sorted(del_positions, reverse=True):
            # Uniformly choose deletion size from 1 to max_del_bp
            del_size = np.random.randint(1, max_del_bp + 1)

            # Save the deleted sequence
            deleted_nucs = seq_indexes[pos : pos + del_size].copy()

            # Shift sequence left to delete forward from pos
            seq_indexes[pos:-del_size] = seq_indexes[pos + del_size :]

            # Compensate by inserting the deleted sequence at the right end
            seq_indexes[-del_size:] = deleted_nucs


def strand_pair_indices(targets_df):
    """Positional strand-partner index (0..N-1) for each target.

    ``targets_df.strand_pair`` holds dataframe index *labels* that reference the
    (full) targets table, not positional indices. This maps each label to its
    positional slot within the given ``targets_df`` so it can index the model's
    output channels / a preds-targets array directly. When there is no
    ``strand_pair`` column, every target is treated as its own partner
    (unstranded) via the identity mapping.

    Args:
        targets_df: pandas DataFrame of targets.

    Returns:
        np.ndarray of int positional partner indices, length ``len(targets_df)``.
    """
    if "strand_pair" in targets_df.columns:
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        return np.array([orig_new_index[ti] for ti in targets_df.strand_pair])
    return np.arange(targets_df.shape[0])


def annotate_strand(targets_df):
    """Add a 'strand' column to a targets table: '+'/'-', or '.' if unstranded.

    A target is unstranded ('.') when it is its own strand pair; otherwise the
    strand is read from the last character of its identifier. No rows are
    dropped, so both strands remain available (e.g. for reverse-complement
    selection). See targets_prep_strand for the variant that also collapses.

    Args:
        targets_df: pandas DataFrame of targets

    Returns:
        targets_df: same DataFrame with a 'strand' column added
    """
    targets_strand = []
    for _, target in targets_df.iterrows():
        if target.strand_pair == target.name:
            targets_strand.append(".")
        else:
            targets_strand.append(target.identifier[-1])
    targets_df["strand"] = targets_strand
    return targets_df


def strand_collapse(targets_df):
    """Plus-representative positions and their strand partners.

    Returns:
        (rep_pos, pair_pos): positional indices, one entry per experiment;
        ``pair_pos == rep_pos`` for unstranded targets.
    """
    pair = strand_pair_indices(targets_df)
    if "strand_pair" in targets_df.columns:
        rep_mask = (annotate_strand(targets_df.copy()).strand != "-").values
    else:
        rep_mask = np.ones(len(targets_df), dtype=bool)
    rep_pos = np.where(rep_mask)[0]
    pair_pos = pair[rep_pos]

    # a representative's partner must be dropped; if both survive (identifiers
    # not ending in +/-) the pair would be counted twice -- fail loudly.
    if set(pair_pos[pair_pos != rep_pos].tolist()) & set(rep_pos.tolist()):
        raise ValueError(
            "stranded pair partners both survived strand collapse; paired target "
            "identifiers must end in +/- (annotate_strand convention)"
        )
    return rep_pos, pair_pos


def target_groups(targets_df):
    """Target group per row: the ``group`` column, else the description prefix.

    Description-derived groups are the text before ':' (``CHIP:x`` -> ``CHIP/x``),
    or ``*`` when there is no ':'.
    """
    if "group" in targets_df.columns:
        return targets_df.group.values
    groups = []
    for description in targets_df.description:
        desc_split = description.split(":")
        if len(desc_split) == 1:
            groups.append("*")
        elif desc_split[0] == "CHIP":
            groups.append("/".join(desc_split[:2]))
        else:
            groups.append(desc_split[0])
    return np.array(groups)


# non-negative finite fp16 bit patterns: 0x0000 (0.0) to 0x7BFF (65504)
NUM_HIST_BINS = 0x7C00


def _target_block_hist(args):
    """Per-track counts of strand-summed fp16 target bit patterns over
    sequences [start, end)."""
    zarr_file, start, end, pair = args
    targets = zarr.open(zarr_file, mode="r")["target"]
    # int32: a block's counts per bin are at most (end - start) * length
    hist = np.zeros((targets.shape[1], NUM_HIST_BINS), dtype=np.int32)
    for si in range(start, end):
        y = targets[si].astype(np.float32)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError(f"{zarr_file} seq {si}: negative or non-finite targets")
        # + 0.0 turns -0.0 into 0.0
        summed = np.minimum(y + y[pair], 65504) + 0.0
        bits = summed.astype(np.float16).view(np.uint16)
        for ti, track_bits in enumerate(bits):
            hist[ti] += np.bincount(track_bits, minlength=NUM_HIST_BINS)
    return hist


def write_target_hist(data_dir, processes=16, block_seqs=256):
    """Store exact per-track histograms of the stored targets in each zarr.

    Counts are over every position of every sequence in all examples/*.zarr,
    of the fp16 value y_t + y_pair(t) (strand_pair from targets.txt; unstranded
    tracks are their own pair, so 2 * y_t), clamped to 65504 and binned by fp16
    bit pattern. Counts are written to each zarr as ``target_hist``
    (num_targets, NUM_HIST_BINS) int64,
    from which SpecPearsonCorrCoef builds its quantile maps.

    Args:
        data_dir: dataset directory with targets.txt and examples/*.zarr.
        processes: parallel readers.
        block_seqs: sequences per read task.

    Returns:
        np.ndarray of per-track histograms.
    """
    targets_df = pd.read_csv(f"{data_dir}/targets.txt", sep="\t", index_col=0)
    pair = strand_pair_indices(targets_df)
    zarr_files = natsorted(glob.glob(f"{data_dir}/examples/*.zarr"))
    tasks = []
    for zarr_file in zarr_files:
        num_seqs = zarr.open(zarr_file, mode="r")["target"].shape[0]
        for start in range(0, num_seqs, block_seqs):
            tasks.append((zarr_file, start, min(start + block_seqs, num_seqs), pair))

    hist = 0
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        for block_hist in pool.imap_unordered(_target_block_hist, tasks):
            hist = hist + block_hist.astype(np.int64)

    for zarr_file in zarr_files:
        root = zarr.open_group(zarr_file, mode="r+")
        root.create_array(
            "target_hist",
            shape=hist.shape,
            dtype="int64",
            chunks=(1, NUM_HIST_BINS),
            overwrite=True,
        )[:] = hist
    return hist


def targets_prep_strand(targets_df):
    """Adjust targets table for merged stranded datasets.

    Args:
        targets_df: pandas DataFrame of targets

    Returns:
        targets_df: pandas DataFrame of targets, with stranded
            targets collapsed into a single row
    """
    # attach strand
    targets_df = annotate_strand(targets_df)

    # collapse stranded
    strand_mask = targets_df.strand != "-"
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df


def untransform_preds(preds, targets_df, unscale=False, unclip=True):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (torch.Tensor/numpy.array): Predictions TxL
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (torch.Tensor/numpy.array): Untransformed predictions TxL.
    """
    if targets_df is not None:
        preds_np = isinstance(preds, np.ndarray)
        if preds_np:
            preds = torch.from_numpy(preds)

        # stick to existing device
        device = preds.device

        # clip soft
        if unclip and "clip_soft" in targets_df.columns:
            cs = torch.tensor(targets_df.clip_soft.values, device=device)
            cs = cs.unsqueeze(1)
            preds_unclip = cs - 1 + (preds - cs + 1) ** 2
            preds = torch.where(preds > cs, preds_unclip, preds)

        # sqrt
        sqrt_mask = [ss.find("_sqrt") != -1 for ss in targets_df.sum_stat.values]
        sqrt_mask = torch.tensor(sqrt_mask, dtype=torch.bool, device=device)

        if sqrt_mask.any():
            sqrt_indices = sqrt_mask.nonzero().squeeze(1)
            preds_sqrt = preds[sqrt_indices]
            preds_sqrt = -1 + (preds_sqrt + 1) ** 2
            preds[sqrt_indices] = preds_sqrt

        # exp75
        exp75_mask = [ss.find("_exp75") != -1 for ss in targets_df.sum_stat.values]
        exp75_mask = torch.tensor(exp75_mask, dtype=torch.bool, device=device)

        if exp75_mask.any():
            exp75_indices = exp75_mask.nonzero().squeeze(1)
            preds_exp75 = preds[exp75_indices]
            preds_exp75 = -1 + (preds_exp75 + 1) ** (4 / 3)
            preds[exp75_indices] = preds_exp75

        # scale
        if unscale:
            scale = torch.tensor(targets_df.scale.values, device=device)
            scale = scale.unsqueeze(1)
            preds = preds / scale

        if preds_np:
            preds = preds.cpu().numpy()

    return preds


def untransform_preds_borzoi(preds, targets_df, unscale=False, unclip=True):
    """Undo the Borzoi squashing transformations (scale -> clip -> sqrt order).

    Args:
        preds (torch.Tensor/numpy.array): Predictions TxL
        targets_df (pd.DataFrame): Targets information table.
            Required columns: "scale" and "sum_stat".
            Optional column: "clip_soft" (only used when present and unclip=True).

    Returns:
      preds (torch.Tensor/numpy.array): Untransformed predictions TxL.
    """
    if targets_df is not None:
        required_columns = {"scale", "sum_stat"}
        missing_columns = required_columns.difference(targets_df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"untransform_preds_borzoi requires targets_df columns: {missing_str}"
            )

        preds_np = isinstance(preds, np.ndarray)
        if preds_np:
            preds = torch.from_numpy(preds)

        device = preds.device

        # scale
        scale = torch.tensor(targets_df.scale.values, device=device).unsqueeze(1)
        preds = preds / scale

        # clip soft
        if unclip and "clip_soft" in targets_df.columns:
            cs = torch.tensor(targets_df.clip_soft.values, device=device).unsqueeze(1)
            preds_unclip = cs + (preds - cs) ** 2
            preds = torch.where(preds > cs, preds_unclip, preds)

        # sqrt meant ** (4/3)
        sqrt_mask = torch.tensor(
            [ss.find("_sqrt") != -1 for ss in targets_df.sum_stat.values],
            dtype=torch.bool,
            device=device,
        )
        if sqrt_mask.any():
            sqrt_indices = sqrt_mask.nonzero().squeeze(1)
            preds[sqrt_indices] = preds[sqrt_indices] ** (4 / 3)

        # undo the scale division if caller did not want unscaling
        if not unscale:
            preds = preds * scale

        if preds_np:
            preds = preds.cpu().numpy()

    return preds


def get_untransform_func(params_model):
    """Return the appropriate untransform function based on params.

    Args:
      params_model (dict): Model params; reads the "untransform" key
        ("borzoi" or "flashzoi" → untransform_preds_borzoi,
         anything else → untransform_preds).

    Returns:
      callable: untransform_preds or untransform_preds_borzoi.
    """
    if params_model.get("untransform") in ("borzoi", "flashzoi"):
        return untransform_preds_borzoi
    return untransform_preds
