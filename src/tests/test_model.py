"""
Unit tests for model components: forward passes, loss computation, and gene aggregation.

Training integration tests are in test_train.py.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from baskerville.dataset import SeqDataset
from baskerville.metrics import GenePoissonLoss
from baskerville.seqnn import SeqNN
from baskerville.types import ModelOutput


class TestModelComponents:
    """Unit tests for model components with gene expression data."""

    # Dataset parameters
    train_seqs = 8
    valid_seqs = 4
    seq_length = 131072
    pool_width = 32
    num_targets = 4  # Coverage tracks
    num_gene_targets = 2  # RNA-seq samples
    max_genes = 3  # Maximum genes per sequence
    seq_depth = 4
    target_length = seq_length // pool_width

    @pytest.fixture
    def mock_gene_dataset(self):
        """Create a temporary dataset with both coverage and gene data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
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
                "num_targets_gene": self.num_gene_targets,
                "max_genes_per_seq": self.max_genes,
                "gene_coverage_threshold": 0.5,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create train zarr
            zarr_train_file = examples_dir / "train.zarr"
            zarr_train = zarr.open(zarr_train_file, mode="w")
            seq_data = np.random.randint(0, 4, size=(self.train_seqs, self.seq_length))
            zarr_train.create_array("sequence", data=seq_data.astype("uint8"))

            target_data = (
                np.random.rand(self.train_seqs, self.num_targets, self.target_length)
                * 10
            )
            zarr_train.create_array("target", data=target_data.astype("float16"))

            gene_target_data = (
                np.random.rand(self.train_seqs, self.num_gene_targets, self.max_genes)
                * 100
            )
            zarr_train.create_array(
                "gene_target", data=gene_target_data.astype("float16")
            )

            gene_presence_data = np.ones((self.train_seqs, self.max_genes), dtype=bool)
            zarr_train.create_array("gene_presence", data=gene_presence_data)

            # gene_out_mask spatial dim must match model output
            # (test models use pool_size=2, so output = seq_length // 2)
            model_output_len = self.seq_length // 2
            gene_out_mask_data = np.zeros(
                (self.train_seqs, self.max_genes, model_output_len),
                dtype="bool",
            )
            # Gene positions distributed across sequence
            gene_out_mask_data[:, 0, 1000:1100] = True
            gene_out_mask_data[:, 1, 2000:2100] = True
            gene_out_mask_data[:, 2, 3000:3100] = True
            zarr_train.create_array("gene_out_mask", data=gene_out_mask_data)

            # Create valid zarr
            zarr_valid_file = examples_dir / "valid.zarr"
            zarr_valid = zarr.open(zarr_valid_file, mode="w")
            seq_data = np.random.randint(0, 4, size=(self.valid_seqs, self.seq_length))
            zarr_valid.create_array("sequence", data=seq_data.astype("uint8"))

            target_data = (
                np.random.rand(self.valid_seqs, self.num_targets, self.target_length)
                * 10
            )
            zarr_valid.create_array("target", data=target_data.astype("float16"))

            gene_target_data = (
                np.random.rand(self.valid_seqs, self.num_gene_targets, self.max_genes)
                * 100
            )
            zarr_valid.create_array(
                "gene_target", data=gene_target_data.astype("float16")
            )

            gene_presence_data = np.ones((self.valid_seqs, self.max_genes), dtype=bool)
            zarr_valid.create_array("gene_presence", data=gene_presence_data)

            gene_out_mask_data = np.zeros(
                (self.valid_seqs, self.max_genes, model_output_len),
                dtype="bool",
            )
            gene_out_mask_data[:, 0, 1000:1100] = True
            gene_out_mask_data[:, 1, 2000:2100] = True
            gene_out_mask_data[:, 2, 3000:3100] = True
            zarr_valid.create_array("gene_out_mask", data=gene_out_mask_data)

            # Create targets files
            targets_df = pd.DataFrame(
                {"strand_pair": np.arange(self.num_targets)},
                index=range(self.num_targets),
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

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

            # Create genes.txt
            genes_df = pd.DataFrame(
                {
                    "gene_id": ["gene1", "gene2", "gene3"],
                    "gene_name": ["GENE1", "GENE2", "GENE3"],
                    "chr": ["chr1", "chr1", "chr1"],
                    "strand": ["+", "+", "-"],
                    "exon_coords": ["1000-1500", "2000-2500", "3000-3500"],
                }
            )
            genes_df.to_csv(data_dir / "genes.txt", sep="\t", index=False)

            yield tmp_dir

    def test_model_forward_with_genes(self, mock_gene_dataset):
        """Test SeqNN forward pass with gene metadata."""
        # Create a simple model with coverage and gene heads
        model_def = {
            "seq_length": self.seq_length,
            "target_length": self.target_length,
            "augment_rc": False,
            "trunk": [
                {
                    "name": "ConvBlock",
                    "in_channels": 4,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "pool_size": 2,
                }
            ],
            "heads_cov": [
                {
                    "species": "test",
                    "name": "Final",
                    "in_channels": 8,
                    "num_targets": self.num_targets,
                }
            ],
            "heads_gene": [
                {
                    "species": "test",
                    "name": "FinalGene",
                    "in_channels": 8,
                    "num_targets": self.num_gene_targets,
                    "aggregation": "sum",
                }
            ],
        }

        model = SeqNN(model_def)
        model.model.eval()  # Set model to eval mode

        # Load dataset
        dataset = SeqDataset(
            data_dir=mock_gene_dataset,
            split_label="train",
            mode="eval",
        )

        # Get one example
        batch = dataset[0]
        seq = batch.sequence
        gene_presence = batch.gene_presence
        gene_out_mask = batch.gene_out_mask

        # Add batch dimension and convert to float32 (model is in float32)
        seq = seq.unsqueeze(0).float()
        gene_presence = gene_presence.unsqueeze(0)
        gene_out_mask = gene_out_mask.unsqueeze(0)

        # Forward pass with both heads (hi=-1 uses all heads)
        # Use model.model directly to bypass autocast in SeqNN.__call__
        with torch.no_grad():
            output = model.model(
                seq, hi=-1, gene_out_mask=gene_out_mask, gene_presence=gene_presence
            )

        # Should return ModelOutput with both coverage and gene
        assert isinstance(output, ModelOutput)
        assert output.has_coverage
        assert output.has_gene

        coverage_pred = output.coverage
        gene_pred = output.gene

        # Check shapes
        assert coverage_pred.shape[0] == 1  # batch size
        assert coverage_pred.shape[1] == self.num_targets
        # After pool/2, seq_length=131072 becomes 65536
        assert coverage_pred.shape[2] == self.seq_length // 2

        assert gene_pred.shape[0] == 1  # batch size
        assert gene_pred.shape[1] == self.num_gene_targets
        assert gene_pred.shape[2] == self.max_genes

    def test_model_forward_coverage_only(self, mock_gene_dataset):
        """Test SeqNN forward with only coverage head (backward compatibility)."""
        model_def = {
            "seq_length": self.seq_length,
            "target_length": self.target_length,
            "augment_rc": False,
            "trunk": [
                {
                    "name": "ConvBlock",
                    "in_channels": 4,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "pool_size": 2,
                }
            ],
            "heads_cov": [
                {
                    "species": "test",
                    "name": "Final",
                    "in_channels": 8,
                    "num_targets": self.num_targets,
                }
            ],
        }

        model = SeqNN(model_def)
        model.model.eval()  # Set model to eval mode

        dataset = SeqDataset(
            data_dir=mock_gene_dataset,
            split_label="train",
            mode="eval",
            skip_genes=True,
        )

        batch = dataset[0]
        seq = batch.sequence.unsqueeze(0).float()

        # Use model.model directly to bypass autocast
        with torch.no_grad():
            output = model.model(seq, hi=0)

        # Should return ModelOutput with only coverage
        assert isinstance(output, ModelOutput)
        assert output.has_coverage
        assert not output.has_gene
        assert output.coverage.shape[0] == 1
        assert output.coverage.shape[1] == self.num_targets

    def test_model_forward_gene_only(self, mock_gene_dataset):
        """Test SeqNN forward with only gene head."""
        model_def = {
            "seq_length": self.seq_length,
            "target_length": self.target_length,
            "augment_rc": False,
            "trunk": [
                {
                    "name": "ConvBlock",
                    "in_channels": 4,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "pool_size": 2,
                }
            ],
            "heads_gene": [
                {
                    "species": "test",
                    "name": "FinalGene",
                    "in_channels": 8,
                    "num_targets": self.num_gene_targets,
                    "aggregation": "sum",
                }
            ],
        }

        model = SeqNN(model_def)
        model.model.eval()  # Set model to eval mode

        dataset = SeqDataset(
            data_dir=mock_gene_dataset,
            split_label="train",
            mode="eval",
        )

        batch = dataset[0]
        seq = batch.sequence.unsqueeze(0).float()
        gene_presence = batch.gene_presence.unsqueeze(0)
        gene_out_mask = batch.gene_out_mask.unsqueeze(0)

        # Use model.model directly to bypass autocast
        with torch.no_grad():
            output = model.model(
                seq, hi=0, gene_out_mask=gene_out_mask, gene_presence=gene_presence
            )

        # Should return ModelOutput with only gene predictions
        assert isinstance(output, ModelOutput)
        assert not output.has_coverage
        assert output.has_gene
        assert output.gene.shape == (1, self.num_gene_targets, self.max_genes)

    def test_gene_loss_computation(self, mock_gene_dataset):
        """Test gene expression loss computation."""
        loss_fn = GenePoissonLoss()

        # Create fake predictions and targets (ensure predictions are positive for Poisson)
        batch_size = 2
        y_pred = (
            torch.rand(batch_size, self.num_gene_targets, self.max_genes) * 100 + 1.0
        )
        y_true = torch.rand(batch_size, self.num_gene_targets, self.max_genes) * 100
        gene_presence = torch.ones(batch_size, self.max_genes, dtype=torch.bool)

        # Mask out last gene
        gene_presence[:, -1] = False

        # Compute loss
        loss = loss_fn(y_pred, y_true, gene_presence)

        # Check loss is scalar and finite (Poisson loss can be positive or negative)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert torch.isfinite(loss)

    def test_gene_aggregation_sum(self, mock_gene_dataset):
        """Test FinalGene block with sum aggregation."""
        from baskerville.blocks import FinalGene

        # Create FinalGene block
        block = FinalGene(
            in_channels=8,
            num_targets=self.num_gene_targets,
            aggregation="sum",
        )
        block.eval()  # Set block to eval mode

        # Create fake trunk output
        batch_size = 2
        trunk_len = self.target_length // 2  # After one pool layer
        x = torch.rand(batch_size, 8, trunk_len)

        # Create gene bin mask and gene mask
        gene_out_mask = torch.zeros(
            batch_size, self.max_genes, trunk_len, dtype=torch.bool
        )
        gene_out_mask[0, 0, 100:200] = True  # Seq 1, gene 0
        gene_out_mask[0, 1, 300:400] = True  # Seq 1, gene 1
        gene_out_mask[0, 2, 500:600] = True  # Seq 1, gene 2
        gene_out_mask[1, 0, 150:250] = True  # Seq 2, gene 0
        gene_out_mask[1, 1, 350:450] = True  # Seq 2, gene 1
        gene_out_mask[1, 2, 550:650] = True  # Seq 2, gene 2

        gene_presence = torch.tensor(
            [
                [True, True, False],  # Sequence 1: 2 valid genes
                [True, True, True],  # Sequence 2: 3 valid genes
            ],
            dtype=torch.bool,
        )

        # Forward pass
        with torch.no_grad():
            output = block(x, gene_out_mask, gene_presence)

        # Check output shape
        assert output.shape == (batch_size, self.num_gene_targets, self.max_genes)

        # Check that masked genes have zero predictions
        assert torch.all(output[0, :, 2] == 0)  # Seq 1, gene 2 is masked

    def test_gene_aggregation_mean(self, mock_gene_dataset):
        """Test FinalGene block with mean aggregation."""
        from baskerville.blocks import FinalGene

        block = FinalGene(
            in_channels=8,
            num_targets=self.num_gene_targets,
            aggregation="mean",
        )
        block.eval()  # Set block to eval mode

        batch_size = 2
        trunk_len = self.target_length // 2
        x = torch.rand(batch_size, 8, trunk_len)

        gene_out_mask = torch.zeros(
            batch_size, self.max_genes, trunk_len, dtype=torch.bool
        )
        gene_out_mask[0, 0, 100:200] = True
        gene_out_mask[0, 1, 300:400] = True
        gene_out_mask[0, 2, 500:600] = True
        gene_out_mask[1, 0, 150:250] = True
        gene_out_mask[1, 1, 350:450] = True
        gene_out_mask[1, 2, 550:650] = True

        gene_presence = torch.ones(batch_size, self.max_genes, dtype=torch.bool)

        with torch.no_grad():
            output = block(x, gene_out_mask, gene_presence)

        assert output.shape == (batch_size, self.num_gene_targets, self.max_genes)
        assert torch.all(output >= 0)  # Softplus ensures non-negative

    def test_gene_output_linear(self, mock_gene_dataset):
        """Test FinalGene with linear output: signed masked mean of bin outputs."""
        from baskerville.blocks import FinalGene

        block = FinalGene(
            in_channels=8,
            num_targets=self.num_gene_targets,
            aggregation="mean",
            output_act="linear",
        )
        block.eval()

        trunk_len = self.target_length // 2
        x = torch.randn(1, 8, trunk_len)
        gene_out_mask = torch.zeros(1, self.max_genes, trunk_len, dtype=torch.bool)
        gene_out_mask[0, 0, 100:200] = True
        gene_out_mask[0, 1, 300:310] = True
        gene_presence = torch.tensor([[True, True, False]])

        with torch.no_grad():
            output = block(x, gene_out_mask, gene_presence)
            bins = block.conv(block.act(block.norm(x)))  # (1, T, L)

        assert (bins < 0).any()
        torch.testing.assert_close(output[0, :, 0], bins[0, :, 100:200].mean(-1))
        torch.testing.assert_close(output[0, :, 1], bins[0, :, 300:310].mean(-1))
        assert torch.all(output[0, :, 2] == 0)
