import json
from pathlib import Path
import pytest
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

from baskerville.dataset import SeqDatasetMLM
from baskerville.trainer import Trainer
from baskerville.types import BatchData


class TestSeqDatasetMLM:
    """Test the SeqDatasetMLM class for MLM data loading."""

    # Dataset parameters
    train_seqs = 5
    seq_length = 16384
    seq_depth = 4
    num_species = 3

    @pytest.fixture
    def mock_mlm_data_dir(self):
        """Create a temporary directory with mock MLM dataset files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            # Create statistics.json
            stats = {
                "seq_length": self.seq_length,
                "target_length": self.seq_length,
                "num_targets": 4,
                "pool_width": 1,
                "seq_depth": self.seq_depth,
                "num_species": self.num_species,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create train zarr with sequence, label, mask, and repeat_mask
            zarr_train_file = examples_dir / "train.zarr"
            zarr_train = zarr.open(zarr_train_file, mode="w")

            # Random sequence indexes (0-3 for ACGT)
            seq_data = np.random.randint(
                0, 4, size=(self.train_seqs, self.seq_length)
            ).astype("uint8")
            zarr_train.create_array("sequence", data=seq_data)

            # One-hot species labels (num_species,)
            label_data = np.zeros((self.train_seqs, self.num_species), dtype="float32")
            for i in range(self.train_seqs):
                label_data[i, i % self.num_species] = 1.0
            zarr_train.create_array("label", data=label_data)

            # Binary exon mask
            mask_data = np.random.randint(
                0, 2, size=(self.train_seqs, self.seq_length)
            ).astype("uint8")
            zarr_train.create_array("mask", data=mask_data)

            # Binary repeat mask
            repeat_mask_data = np.random.randint(
                0, 2, size=(self.train_seqs, self.seq_length)
            ).astype("uint8")
            zarr_train.create_array("repeat_mask", data=repeat_mask_data)

            yield tmp_dir

    def test_init(self, mock_mlm_data_dir):
        """Test basic dataset initialization."""
        dataset = SeqDatasetMLM(
            data_dir=mock_mlm_data_dir,
            split_label="train",
            mode="train",
            has_mask=True,
            has_repeat_mask=True,
        )

        assert dataset.seq_length == self.seq_length
        assert dataset.num_species == self.num_species
        assert dataset.has_mask is True
        assert dataset.has_repeat_mask is True
        assert len(dataset) == self.train_seqs

    def test_getitem_basic(self, mock_mlm_data_dir):
        """Test __getitem__ returns a populated BatchData."""
        dataset = SeqDatasetMLM(
            data_dir=mock_mlm_data_dir,
            split_label="train",
            mode="eval",
            has_mask=True,
            has_repeat_mask=True,
            augment_rc=False,
        )

        result = dataset[0]
        assert isinstance(result, BatchData)

        # Check sequence shape and type
        assert isinstance(result.sequence, torch.Tensor)
        assert result.sequence.shape == (self.seq_depth, self.seq_length)
        assert result.sequence.sum() == self.seq_length  # One-hot encoding

        # Check species label shape
        assert isinstance(result.species_label, torch.Tensor)
        assert result.species_label.shape == (1, self.num_species)
        assert result.species_label.sum() == 1.0  # One-hot species label

        # Check masks shape
        assert isinstance(result.exon_mask, torch.Tensor)
        assert result.exon_mask.shape == (self.seq_length,)
        assert isinstance(result.repeat_mask, torch.Tensor)
        assert result.repeat_mask.shape == (self.seq_length,)

    def test_getitem_no_masks(self, mock_mlm_data_dir):
        """Test __getitem__ without masks leaves mask fields None."""
        dataset = SeqDatasetMLM(
            data_dir=mock_mlm_data_dir,
            split_label="train",
            mode="eval",
            has_mask=False,
            has_repeat_mask=False,
        )

        result = dataset[0]
        assert isinstance(result, BatchData)
        assert result.sequence.shape == (self.seq_depth, self.seq_length)
        assert result.species_label.shape == (1, self.num_species)
        assert result.exon_mask is None
        assert result.repeat_mask is None

    def test_rc_augmentation(self, mock_mlm_data_dir):
        """Test reverse complement augmentation in train mode."""
        dataset = SeqDatasetMLM(
            data_dir=mock_mlm_data_dir,
            split_label="train",
            mode="train",
            has_mask=False,
            has_repeat_mask=False,
            augment_rc=True,
        )

        # Get the same sequence multiple times
        # With RC augmentation, about half should be reverse complemented
        seqs = [dataset[0].sequence for _ in range(20)]

        # Check that we see both orientations
        # This is probabilistic but with 20 samples, very unlikely to fail
        first_seq = seqs[0]
        different_count = sum(not torch.allclose(first_seq, s) for s in seqs[1:])

        assert different_count > 0, "RC augmentation should produce different sequences"

    def test_dataloader(self, mock_mlm_data_dir):
        """Test DataLoader functionality."""
        dataset = SeqDatasetMLM(
            data_dir=mock_mlm_data_dir,
            split_label="train",
            mode="eval",
            has_mask=True,
            has_repeat_mask=True,
        )

        batch_size = 2
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=BatchData.collate,
        )

        batch = next(iter(loader))
        assert isinstance(batch, BatchData)
        assert batch.sequence.shape == (batch_size, self.seq_depth, self.seq_length)
        assert batch.species_label.shape == (batch_size, 1, self.num_species)
        assert batch.exon_mask.shape == (batch_size, self.seq_length)
        assert batch.repeat_mask.shape == (batch_size, self.seq_length)


class TestPrepMLM:
    """Test the prep_mlm function for masking logic."""

    @pytest.fixture
    def mock_trainer_params(self):
        """Create mock trainer parameters for MLM."""
        return {
            "batch_size": 2,
            "learning_rate": 0.001,
            "train_epochs_max": 1,
            "loss": "mlm",
            "mask_rate": 0.15,
            "use_bert": False,
            "exon_loss_scale": 0.1,
            "non_exon_loss_scale": 1.0,
            "repeat_loss_scale": 0.1,
            "non_repeat_loss_scale": 1.0,
        }

    @pytest.fixture
    def mock_mlm_data_dir(self):
        """Create a minimal mock data directory for trainer initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            stats = {
                "seq_length": 1024,
                "target_length": 1024,
                "num_targets": 4,
                "pool_width": 1,
                "seq_depth": 4,
                "num_species": 2,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create minimal train zarr
            zarr_file = examples_dir / "train.zarr"
            z = zarr.open(zarr_file, mode="w")
            z.create_array(
                "sequence", data=np.random.randint(0, 4, (2, 1024)).astype("uint8")
            )
            z.create_array("label", data=np.array([[1, 0], [0, 1]], dtype="float32"))

            # Create minimal valid zarr
            zarr_file = examples_dir / "valid.zarr"
            z = zarr.open(zarr_file, mode="w")
            z.create_array(
                "sequence", data=np.random.randint(0, 4, (2, 1024)).astype("uint8")
            )
            z.create_array("label", data=np.array([[1, 0], [0, 1]], dtype="float32"))

            yield tmp_dir

    def test_basic_masking(self, mock_trainer_params, mock_mlm_data_dir):
        """Test that prep_mlm masks the correct number of positions."""
        # Create datasets
        train_data = [SeqDatasetMLM(mock_mlm_data_dir, "train", "train")]
        eval_data = [SeqDatasetMLM(mock_mlm_data_dir, "valid", "eval")]

        with tempfile.TemporaryDirectory() as out_dir:
            trainer = Trainer(mock_trainer_params, train_data, eval_data, out_dir)

            # Create input tensors
            batch_size = 2
            seq_length = 1024
            x = torch.randn(batch_size, 4, seq_length)

            mask_size = int(0.15 * seq_length)  # 15% masking

            x_masked, x_orig, mask_idx, pos_weights = trainer._mlm_prep(x, mask_size)

            # Check output shapes: 4 DNA channels only (no mask channel)
            assert x_masked.shape == (batch_size, 4, seq_length)
            assert x_orig.shape == (batch_size, 4, seq_length)
            assert mask_idx.shape == (batch_size, mask_size)
            assert pos_weights is None  # No exon/repeat masks provided

            # Check that indices are valid
            assert mask_idx.min() >= 0
            assert mask_idx.max() < seq_length

    def test_loss_weight_scaling(self, mock_trainer_params, mock_mlm_data_dir):
        """Test that exon and repeat masks correctly scale loss weights."""
        train_data = [SeqDatasetMLM(mock_mlm_data_dir, "train", "train")]
        eval_data = [SeqDatasetMLM(mock_mlm_data_dir, "valid", "eval")]

        with tempfile.TemporaryDirectory() as out_dir:
            trainer = Trainer(mock_trainer_params, train_data, eval_data, out_dir)

            batch_size = 2
            seq_length = 1024
            x = torch.randn(batch_size, 4, seq_length)

            # Create masks: first half exon, second half non-exon
            exon_mask = torch.zeros(batch_size, seq_length)
            exon_mask[:, : seq_length // 2] = 1.0

            mask_size = int(0.15 * seq_length)

            _, _, _, pos_weights = trainer._mlm_prep(x, mask_size, exon_mask=exon_mask)

            # Check that position weights are computed
            assert pos_weights is not None
            assert pos_weights.shape == (batch_size, seq_length)

            # Check exon positions have lower weight (0.1)
            assert torch.allclose(pos_weights[:, 0], torch.tensor(0.1))
            # Check non-exon positions have weight 1.0
            assert torch.allclose(pos_weights[:, -1], torch.tensor(1.0))

    def test_loss_weight_scaling_default_non_scale(
        self, mock_trainer_params, mock_mlm_data_dir
    ):
        """Setting only exon_loss_scale should not crash; non-exon weight defaults to 1.0."""
        # drop the "non_*" scales so the defaults are exercised
        mock_trainer_params.pop("non_exon_loss_scale", None)
        mock_trainer_params.pop("non_repeat_loss_scale", None)
        mock_trainer_params.pop("repeat_loss_scale", None)

        train_data = [SeqDatasetMLM(mock_mlm_data_dir, "train", "train")]
        eval_data = [SeqDatasetMLM(mock_mlm_data_dir, "valid", "eval")]

        with tempfile.TemporaryDirectory() as out_dir:
            trainer = Trainer(mock_trainer_params, train_data, eval_data, out_dir)
            assert trainer.non_exon_loss_scale == 1.0
            assert trainer.non_repeat_loss_scale == 1.0

            batch_size = 2
            seq_length = 1024
            x = torch.randn(batch_size, 4, seq_length)

            exon_mask = torch.zeros(batch_size, seq_length)
            exon_mask[:, : seq_length // 2] = 1.0

            mask_size = int(0.15 * seq_length)
            _, _, _, pos_weights = trainer._mlm_prep(x, mask_size, exon_mask=exon_mask)

            assert pos_weights is not None
            # exon positions downweighted to 0.1, non-exon at default 1.0
            assert torch.allclose(pos_weights[:, 0], torch.tensor(0.1))
            assert torch.allclose(pos_weights[:, -1], torch.tensor(1.0))

    def test_bert_style_masking(self, mock_trainer_params, mock_mlm_data_dir):
        """Test BERT-style masking (80% mask, 10% random, 10% keep)."""
        mock_trainer_params["use_bert"] = True

        train_data = [SeqDatasetMLM(mock_mlm_data_dir, "train", "train")]
        eval_data = [SeqDatasetMLM(mock_mlm_data_dir, "valid", "eval")]

        with tempfile.TemporaryDirectory() as out_dir:
            trainer = Trainer(mock_trainer_params, train_data, eval_data, out_dir)

            batch_size = 4
            seq_length = 1024

            # Create one-hot encoded input
            x = torch.zeros(batch_size, 4, seq_length)
            for b in range(batch_size):
                for p in range(seq_length):
                    x[b, np.random.randint(0, 4), p] = 1.0

            mask_size = int(0.15 * seq_length)

            x_masked, _, _, _ = trainer._mlm_prep(x, mask_size, training=True)

            # Output is 4 DNA channels only (no mask channel)
            assert x_masked.shape == (batch_size, 4, seq_length)


class _DummyOut:
    def __init__(self, coverage):
        self.coverage = coverage


class _DummyModel:
    """Returns per-channel constants independent of the input.

    coverage[0, c, :] = (c + 1) * 0.1, so a fully-predicted sequence is easy to
    assert against and the RC average is deterministic.
    """

    def __call__(self, x, hi=0, di=0):
        seq_length = x.shape[-1]
        cov = torch.zeros((1, 4, seq_length))
        for c in range(4):
            cov[0, c, :] = (c + 1) * 0.1
        return _DummyOut(cov)


class TestMLMUtils:
    """Test the shared MLM helpers in baskerville.mlm."""

    def test_predict_masked_sequence_covers_all_positions(self):
        from baskerville import mlm

        seq_length, mask_size = 20, 7  # mask_size does not divide seq_length
        x = torch.zeros((1, 4, seq_length))
        x[0, 0, :] = 1.0

        out = mlm.predict_masked_sequence(
            _DummyModel(),
            x,
            seq_length,
            mask_size,
            "cpu",
            torch.bfloat16,
            di=0,
            rc=False,
        )

        assert out.shape == (1, 4, seq_length)
        # every position predicted (none left at the zero initializer)
        for c in range(4):
            assert torch.allclose(
                out[0, c, :], torch.full((seq_length,), (c + 1) * 0.1)
            )

    def test_predict_masked_sequence_rc_average(self):
        from baskerville import mlm

        seq_length, mask_size = 16, 5
        x = torch.zeros((1, 4, seq_length))
        x[0, 0, :] = 1.0

        out = mlm.predict_masked_sequence(
            _DummyModel(),
            x,
            seq_length,
            mask_size,
            "cpu",
            torch.bfloat16,
            di=0,
            rc=True,
        )

        # flip(dims=[1, 2]) swaps channel c with 3-c; averaging the constant
        # forward/RC predictions gives 0.25 everywhere
        assert torch.allclose(out, torch.full((1, 4, seq_length), 0.25))

    def test_patch_params_for_old_unet(self):
        from baskerville import mlm

        params = {
            "trunk": [
                {"name": "UnetTower"},
                {"name": "UnetV2Tower"},
                {"name": "ConvBlock"},
            ]
        }
        state_dict = {"trunk.0.conv_depth.0.weight": 1, "other": 2}

        assert mlm.patch_params_for_old_unet(params, state_dict) is True
        assert params["trunk"][0]["type"] == "borzoi"
        assert params["trunk"][1]["type"] == "borzoi"
        assert "type" not in params["trunk"][2]

    def test_patch_params_for_old_unet_noop(self):
        from baskerville import mlm

        params = {"trunk": [{"name": "UnetTower"}]}
        assert mlm.patch_params_for_old_unet(params, {"a": 1}) is False
        assert "type" not in params["trunk"][0]


class TestRunningStatsDetection:
    """_module_has_running_stats gates the schedule-free eval-prep warmup."""

    def test_batchnorm_detected(self):
        from baskerville.trainer import _module_has_running_stats

        model = torch.nn.Sequential(torch.nn.Conv1d(4, 8, 3), torch.nn.BatchNorm1d(8))
        assert _module_has_running_stats(model) is True

    def test_cond_batchnorm_detected(self):
        from baskerville.condnorm import CondBatchNorm1d
        from baskerville.trainer import _module_has_running_stats

        model = torch.nn.Sequential(CondBatchNorm1d(2, 8))
        assert _module_has_running_stats(model) is True

    def test_layernorm_rmsnorm_not_detected(self):
        from baskerville.trainer import _module_has_running_stats

        model = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.RMSNorm(8))
        assert _module_has_running_stats(model) is False

    def test_batchnorm_without_running_stats_not_detected(self):
        from baskerville.trainer import _module_has_running_stats

        model = torch.nn.BatchNorm1d(8, track_running_stats=False)
        assert _module_has_running_stats(model) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
