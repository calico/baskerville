import json
from pathlib import Path
import pdb
import pytest
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import zarr

from baskerville.dataset import SeqDataset
from baskerville import dna
from baskerville.types import BatchData


class TestSeqDataset:
    # Dataset parameters
    train_seqs = 5
    valid_seqs = 3
    seq_length = 131072
    pool_width = 32
    num_targets = 6
    seq_depth = 4
    target_length = seq_length // pool_width

    @pytest.fixture
    def mock_data_dir(self):
        """Create a temporary directory with mock dataset files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create necessary directory structure
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            # Create statistics.json
            stats = {
                "seq_length": self.seq_length,
                "target_length": self.target_length,
                "num_targets": self.num_targets,
                "pool_width": self.pool_width,
                "seq_depth": self.seq_depth,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # train data
            zarr_train_file = examples_dir / "train.zarr"
            zarr_train = zarr.open(zarr_train_file, mode="w")
            seq_data = np.random.randint(
                0,
                4,
                size=(
                    self.train_seqs,
                    self.seq_length,
                ),
            )
            target_data = np.random.rand(
                self.train_seqs, self.num_targets, self.target_length
            )
            zarr_train.create_array("sequence", data=seq_data.astype("uint8"))
            zarr_train.create_array("target", data=target_data.astype("float16"))

            # valid data
            # zarr_valid_file = examples_dir / "valid.zarr"
            # with zarr.open(zarr_valid_file, "w") as zarr_valid:
            #     seq_data = np.random.randint(0, 4, size=(self.valid_seqs, self.seq_length,))
            #     target_data = np.random.rand(self.valid_seqs, self.target_length, self.num_targets)
            #     zarr_valid.create_dataset("sequence", data=seq_data, dtype="uint8")
            #     zarr_valid.create_dataset("target", data=target_data, dtype="float16")

            # Create targets file
            targets_pairs = np.arange(self.num_targets)
            targets_pairs[::2] = targets_pairs[1::2]  # Even indices get next number
            targets_pairs[1::2] = (
                targets_pairs[::2] - 1
            )  # Odd indices get previous number
            targets_df = pd.DataFrame(
                {"strand_pair": targets_pairs}, index=range(self.num_targets)
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

            yield tmp_dir

    def test_init(self, mock_data_dir):
        """Test basic dataset initialization."""
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="train")

        assert dataset.seq_length == self.seq_length
        assert dataset.target_length == self.target_length
        assert dataset.num_targets == self.num_targets
        assert dataset.pool_width == self.pool_width
        assert dataset.seq_depth == self.seq_depth
        assert len(dataset) == self.train_seqs

    def test_get_eval(self, mock_data_dir):
        """Test __getitem__ in eval mode."""
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="eval")
        batch = dataset[0]

        assert isinstance(batch.sequence, torch.Tensor)
        assert isinstance(batch.coverage_targets, torch.Tensor)
        assert batch.sequence.shape == (self.seq_depth, self.seq_length)
        assert batch.sequence.sum() == self.seq_length
        assert batch.coverage_targets.shape == (self.num_targets, self.target_length)
        assert batch.coverage_targets.var() > 0
        assert batch.sequence.dtype == torch.float16
        assert batch.coverage_targets.dtype == torch.float16

    def test_augment(self, mock_data_dir):
        """Test __getitem__ in train mode with augmentations."""
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="train")

        # read strand pair
        targets_df = pd.read_csv(Path(mock_data_dir) / "targets.txt", sep="\t")
        strand_pair = np.array(targets_df.strand_pair)

        # manually read sequence 0
        zarr_file = Path(mock_data_dir) / "examples" / "train.zarr"
        zarr_data = zarr.open(zarr_file, mode="r")
        seq_index = zarr_data["sequence"][0]
        target = zarr_data["target"][0]

        # Convert sequence to one-hot
        seq_1hot = np.zeros((self.seq_depth, self.seq_length), dtype="bool")
        seq_1hot[seq_index, np.arange(self.seq_length)] = 1
        seq_1hot = torch.from_numpy(seq_1hot).half()
        seq_1hot_rc = dna.torch_rc(seq_1hot)

        # Convert targets to torch
        target = torch.from_numpy(target)
        target_rc = target[strand_pair].flip(1)

        # Check that different random seeds give different augmentations
        observed_cases = set()
        for _ in range(50):
            dbatch = dataset[0]
            dseq_1hot, dtarget = dbatch.sequence, dbatch.coverage_targets
            if torch.allclose(seq_1hot, dseq_1hot):
                # forward, 0 shift
                observed_cases.add(("fwd", 0))
                assert torch.allclose(target, dtarget)
            elif torch.allclose(seq_1hot[:, 1:], dseq_1hot[:, :-1]):
                # forward, 1 shift
                observed_cases.add(("fwd", 1))
                assert torch.allclose(target, dtarget)
            elif torch.allclose(seq_1hot[:, :-1], dseq_1hot[:, 1:]):
                # forward, -1 shift
                observed_cases.add(("fwd", -1))
                assert torch.allclose(target, dtarget)
            elif torch.allclose(seq_1hot_rc, dseq_1hot):
                # reverse complement, 0 shift
                observed_cases.add(("rc", 0))
                assert torch.allclose(target_rc, dtarget)
            elif torch.allclose(seq_1hot_rc[:, 1:], dseq_1hot[:, :-1]):
                # reverse complement, 1 shift
                observed_cases.add(("rc", 1))
                assert torch.allclose(target_rc, dtarget)
            elif torch.allclose(seq_1hot_rc[:, :-1], dseq_1hot[:, 1:]):
                # reverse complement, -1 shift
                observed_cases.add(("rc", -1))
                assert torch.allclose(target_rc, dtarget)
            else:
                # Unrecognized augmentation
                assert False

        assert len(observed_cases) == 6

    def test_oob(self, mock_data_dir):
        """Test that accessing an index out of bounds raises IndexError."""
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="eval")

        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_dataloader_basic(self, mock_data_dir):
        """Test basic DataLoader functionality with single worker."""
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="eval")

        batch_size = 2
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=BatchData.collate
        )

        # Check batch shapes
        batch = next(iter(loader))
        assert batch.sequence.shape == (
            batch_size,
            dataset.seq_depth,
            dataset.seq_length,
        )
        assert batch.sequence.sum() == batch_size * dataset.seq_length
        assert batch.coverage_targets.shape == (
            batch_size,
            dataset.num_targets,
            dataset.target_length,
        )
        assert batch.coverage_targets.var() > 0

        # Verify data type
        assert batch.sequence.dtype == torch.float16
        assert batch.coverage_targets.dtype == torch.float16

    def test_dataloader_multi_worker(self, mock_data_dir):
        """Test DataLoader with multiple workers.

        Triggers warning:
        ".../multiprocessing/popen_fork.py:66:
        DeprecationWarning: This process (pid=899953) is multi-threaded, use of fork() may lead to deadlocks in the child."
        """
        dataset = SeqDataset(data_dir=mock_data_dir, split_label="train", mode="eval")

        batch_size = 2
        num_workers = 2
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=BatchData.collate,
        )

        # Collect all batches to ensure no worker conflicts
        all_seqs = []
        for batch in loader:
            all_seqs.append(batch.sequence)

        # Verify we got all data
        total_samples = sum(seq.shape[0] for seq in all_seqs)
        assert total_samples == len(dataset)


class TestSeqDatasetGene:
    """Test SeqDataset with gene expression data."""

    # Dataset parameters
    train_seqs = 5
    valid_seqs = 3
    seq_length = 131072
    pool_width = 32
    num_targets = 6
    num_gene_targets = 3  # Number of RNA-seq samples
    max_genes = 4  # Maximum genes per sequence
    seq_depth = 4
    target_length = seq_length // pool_width

    @pytest.fixture
    def mock_gene_data_dir(self):
        """Create a temporary directory with mock dataset files including gene data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create necessary directory structure
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            # Create statistics.json with gene metadata
            stats = {
                "seq_length": self.seq_length,
                "target_length": self.target_length,
                "num_targets": self.num_targets,
                "pool_width": self.pool_width,
                "seq_depth": self.seq_depth,
                "num_targets_gene": self.num_gene_targets,
                "max_genes_per_seq": self.max_genes,
                "gene_coverage_threshold": 0.5,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create train zarr with both coverage and gene data
            zarr_train_file = examples_dir / "train.zarr"
            zarr_train = zarr.open(zarr_train_file, mode="w")
            # Sequence data
            seq_data = np.random.randint(0, 4, size=(self.train_seqs, self.seq_length))
            zarr_train.create_array("sequence", data=seq_data.astype("uint8"))

            # Coverage target data
            target_data = np.random.rand(
                self.train_seqs, self.num_targets, self.target_length
            )
            zarr_train.create_array("target", data=target_data.astype("float16"))

            # Gene target data [num_seqs, num_gene_targets, max_genes]
            gene_target_data = (
                np.random.rand(self.train_seqs, self.num_gene_targets, self.max_genes)
                * 100
            )
            zarr_train.create_array(
                "gene_target", data=gene_target_data.astype("float16")
            )

            # Gene mask data [num_seqs, max_genes]
            gene_presence_data = np.zeros((self.train_seqs, self.max_genes), dtype=bool)
            # First 2 genes valid in each sequence, last 2 invalid (padded)
            gene_presence_data[:, :2] = True
            zarr_train.create_array("gene_presence", data=gene_presence_data)

            # Gene bin mask data [num_seqs, max_genes, target_length] - boolean
            gene_out_mask_data = np.zeros(
                (self.train_seqs, self.max_genes, self.target_length),
                dtype="bool",
            )
            # Gene 0: bins 100-200, Gene 1: bins 300-400
            gene_out_mask_data[:, 0, 100:200] = True
            gene_out_mask_data[:, 1, 300:400] = True
            zarr_train.create_array("gene_out_mask", data=gene_out_mask_data)

            # Create targets file for coverage
            targets_pairs = np.arange(self.num_targets)
            targets_df = pd.DataFrame(
                {"strand_pair": targets_pairs}, index=range(self.num_targets)
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

            # Create targets_gene file for gene expression
            targets_gene_df = pd.DataFrame(
                {
                    "identifier": [f"sample{i}" for i in range(self.num_gene_targets)],
                    "description": [
                        f"Sample {i} RNA-seq" for i in range(self.num_gene_targets)
                    ],
                },
                index=range(self.num_gene_targets),
            )
            targets_gene_df.to_csv(data_dir / "targets_gene.txt", sep="\t")

            # Create genes.txt metadata
            genes_df = pd.DataFrame(
                {
                    "gene_id": ["gene1", "gene2", "gene3", "gene4"],
                    "gene_name": ["GENE1", "GENE2", "GENE3", "GENE4"],
                    "chr": ["chr1", "chr1", "chr1", "chr1"],
                    "strand": ["+", "+", "-", "+"],
                    "exon_coords": ["1000-1500", "2000-2500", "3000-3500", "4000-4500"],
                }
            )
            genes_df.to_csv(data_dir / "genes.txt", sep="\t", index=False)

            yield tmp_dir

    def test_init_with_genes(self, mock_gene_data_dir):
        """Test dataset initialization with gene data."""
        dataset = SeqDataset(
            data_dir=mock_gene_data_dir,
            split_label="train",
            mode="eval",
        )

        assert dataset.seq_length == self.seq_length
        assert dataset.target_length == self.target_length
        assert dataset.num_targets == self.num_targets
        assert dataset.skip_genes == False
        assert dataset.has_genes == True
        assert len(dataset) == self.train_seqs

    def test_get_with_genes_eval(self, mock_gene_data_dir):
        """Test __getitem__ returns gene data in eval mode."""
        dataset = SeqDataset(
            data_dir=mock_gene_data_dir,
            split_label="train",
            mode="eval",
        )

        result = dataset[0]
        assert result.has_coverage
        assert result.has_gene

        # Check sequence and coverage targets
        assert isinstance(result.sequence, torch.Tensor)
        assert isinstance(result.coverage_targets, torch.Tensor)
        assert result.sequence.shape == (self.seq_depth, self.seq_length)
        assert result.coverage_targets.shape == (self.num_targets, self.target_length)

        # Check gene targets shape [num_gene_targets, max_genes]
        assert isinstance(result.gene_targets, torch.Tensor)
        assert result.gene_targets.shape == (self.num_gene_targets, self.max_genes)
        assert result.gene_targets.dtype == torch.float16

        # Check gene mask shape [max_genes]
        assert isinstance(result.gene_presence, torch.Tensor)
        assert result.gene_presence.shape == (self.max_genes,)
        assert result.gene_presence.dtype == torch.bool

        # Check gene bin mask shape [max_genes, target_length]
        assert isinstance(result.gene_out_mask, torch.Tensor)
        assert result.gene_out_mask.shape == (self.max_genes, self.target_length)
        assert result.gene_out_mask.dtype == torch.bool

        # Verify gene mask values
        assert result.gene_presence[0] == True  # First gene valid
        assert result.gene_presence[1] == True  # Second gene valid
        assert result.gene_presence[2] == False  # Third gene padded
        assert result.gene_presence[3] == False  # Fourth gene padded

        # Verify gene bin mask
        assert result.gene_out_mask[0, 100:200].all()  # Gene 0 bins
        assert not result.gene_out_mask[0, :100].any()  # Before gene 0
        assert result.gene_out_mask[1, 300:400].all()  # Gene 1 bins
        assert not result.gene_out_mask[1, :300].any()  # Before gene 1

    def test_get_without_genes(self, mock_gene_data_dir):
        """Test __getitem__ without gene data when skipped."""
        dataset = SeqDataset(
            data_dir=mock_gene_data_dir,
            split_label="train",
            mode="eval",
            skip_genes=True,
        )

        result = dataset[0]
        assert result.has_coverage
        assert not result.has_gene

        assert result.sequence.shape == (self.seq_depth, self.seq_length)
        assert result.coverage_targets.shape == (self.num_targets, self.target_length)

    def test_dataloader_with_genes(self, mock_gene_data_dir):
        """Test DataLoader with gene data."""
        dataset = SeqDataset(
            data_dir=mock_gene_data_dir,
            split_label="train",
            mode="eval",
        )

        batch_size = 2
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=BatchData.collate
        )

        # Get first batch
        batch = next(iter(loader))
        assert batch.has_coverage
        assert batch.has_gene

        # Check batch shapes for sequences and coverage targets
        assert batch.sequence.shape == (batch_size, self.seq_depth, self.seq_length)
        assert batch.coverage_targets.shape == (
            batch_size,
            self.num_targets,
            self.target_length,
        )

        # Check batched gene data shapes
        assert batch.gene_targets.shape == (
            batch_size,
            self.num_gene_targets,
            self.max_genes,
        )
        assert batch.gene_presence.shape == (batch_size, self.max_genes)
        assert batch.gene_out_mask.shape == (
            batch_size,
            self.max_genes,
            self.target_length,
        )

        # Verify data types
        assert batch.gene_targets.dtype == torch.float16
        assert batch.gene_presence.dtype == torch.bool
        assert batch.gene_out_mask.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_batchdata_pin_memory():
    """DataLoader(pin_memory=True) pins every BatchData tensor, leaving Nones."""
    items = [
        BatchData(sequence=torch.zeros(4, 8), coverage_targets=torch.ones(2, 4))
        for _ in range(3)
    ]
    loader = DataLoader(
        items, batch_size=3, collate_fn=BatchData.collate, pin_memory=True
    )
    batch = next(iter(loader))
    assert batch.sequence.is_pinned() and batch.coverage_targets.is_pinned()
    assert batch.gene_targets is None
