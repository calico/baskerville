"""
Tests for SeqNN.eval() method - evaluation with metrics computation.
"""

import json
from functools import partial
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from baskerville import metrics
from baskerville.dataset import NUM_HIST_BINS, SeqDataset
from baskerville.seqnn import SeqNN


class TestSeqNNEval:
    """Tests for SeqNN.eval() method."""

    # Dataset parameters
    num_seqs = 4
    seq_length = 8192
    pool_width = 32
    num_targets = 3
    num_gene_targets = 2
    max_genes = 3
    seq_depth = 4
    target_length = seq_length // pool_width

    def _create_model_def(self, has_coverage=True, has_gene=False):
        """Create model definition with specified heads."""
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
                    "pool_size": self.pool_width,
                }
            ],
        }

        if has_coverage:
            model_def["heads_cov"] = [
                {
                    "species": "test",
                    "name": "Final",
                    "in_channels": 8,
                    "num_targets": self.num_targets,
                }
            ]

        if has_gene:
            model_def["heads_gene"] = [
                {
                    "species": "test",
                    "name": "FinalGene",
                    "in_channels": 8,
                    "num_targets": self.num_gene_targets,
                    "aggregation": "sum",
                }
            ]

        return model_def

    @pytest.fixture
    def coverage_only_dataset(self):
        """Create dataset with only coverage targets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            stats = {
                "seq_length": self.seq_length,
                "target_length": self.target_length,
                "num_targets": self.num_targets,
                "pool_width": self.pool_width,
                "seq_depth": self.seq_depth,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create test zarr with coverage only
            zarr_file = examples_dir / "test.zarr"
            z = zarr.open(zarr_file, mode="w")
            z.create_array(
                "sequence",
                data=np.random.randint(
                    0, 4, size=(self.num_seqs, self.seq_length)
                ).astype("uint8"),
            )
            z.create_array(
                "target",
                data=(
                    np.random.rand(self.num_seqs, self.num_targets, self.target_length)
                    * 10
                ).astype("float16"),
            )

            # Create targets.txt
            targets_df = pd.DataFrame(
                {
                    "identifier": [f"track{i}" for i in range(self.num_targets)],
                    "description": [f"Track {i}" for i in range(self.num_targets)],
                    "strand_pair": np.arange(self.num_targets),
                },
                index=range(self.num_targets),
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

            yield tmp_dir

    @pytest.fixture
    def gene_only_dataset(self):
        """Create dataset with only gene targets (no coverage)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            stats = {
                "seq_length": self.seq_length,
                "target_length": self.target_length,
                "num_targets": 0,  # No coverage targets
                "pool_width": self.pool_width,
                "seq_depth": self.seq_depth,
                "num_targets_gene": self.num_gene_targets,
                "max_genes_per_seq": self.max_genes,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create test zarr with gene data only
            zarr_file = examples_dir / "test.zarr"
            z = zarr.open(zarr_file, mode="w")
            z.create_array(
                "sequence",
                data=np.random.randint(
                    0, 4, size=(self.num_seqs, self.seq_length)
                ).astype("uint8"),
            )
            z.create_array(
                "gene_target",
                data=(
                    np.random.rand(self.num_seqs, self.num_gene_targets, self.max_genes)
                    * 100
                ).astype("float16"),
            )
            z.create_array(
                "gene_presence",
                data=np.ones((self.num_seqs, self.max_genes), dtype=bool),
            )
            gene_out_mask = np.zeros(
                (self.num_seqs, self.max_genes, self.target_length),
                dtype="bool",
            )
            gene_out_mask[:, 0, 50:70] = True
            gene_out_mask[:, 1, 100:120] = True
            gene_out_mask[:, 2, 150:170] = True
            z.create_array("gene_out_mask", data=gene_out_mask)

            # Create empty targets.txt (required but empty)
            targets_df = pd.DataFrame(
                {"identifier": [], "description": [], "strand_pair": []},
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

            # Create targets_gene.txt
            targets_gene_df = pd.DataFrame(
                {
                    "identifier": [f"rna{i}" for i in range(self.num_gene_targets)],
                    "description": [
                        f"RNA-seq {i}" for i in range(self.num_gene_targets)
                    ],
                },
                index=range(self.num_gene_targets),
            )
            targets_gene_df.to_csv(data_dir / "targets_gene.txt", sep="\t")

            # Create genes.txt
            genes_df = pd.DataFrame(
                {
                    "gene_id": [f"gene{i}" for i in range(self.max_genes)],
                    "gene_name": [f"GENE{i}" for i in range(self.max_genes)],
                }
            )
            genes_df.to_csv(data_dir / "genes.txt", sep="\t", index=False)

            yield tmp_dir

    @pytest.fixture
    def combined_dataset(self):
        """Create dataset with both coverage and gene targets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            examples_dir = data_dir / "examples"
            examples_dir.mkdir(parents=True)

            stats = {
                "seq_length": self.seq_length,
                "target_length": self.target_length,
                "num_targets": self.num_targets,
                "pool_width": self.pool_width,
                "seq_depth": self.seq_depth,
                "num_targets_gene": self.num_gene_targets,
                "max_genes_per_seq": self.max_genes,
            }
            with open(data_dir / "statistics.json", "w") as f:
                json.dump(stats, f)

            # Create test zarr with both coverage and gene data
            zarr_file = examples_dir / "test.zarr"
            z = zarr.open(zarr_file, mode="w")
            z.create_array(
                "sequence",
                data=np.random.randint(
                    0, 4, size=(self.num_seqs, self.seq_length)
                ).astype("uint8"),
            )
            z.create_array(
                "target",
                data=(
                    np.random.rand(self.num_seqs, self.num_targets, self.target_length)
                    * 10
                ).astype("float16"),
            )
            z.create_array(
                "gene_target",
                data=(
                    np.random.rand(self.num_seqs, self.num_gene_targets, self.max_genes)
                    * 100
                ).astype("float16"),
            )
            z.create_array(
                "gene_presence",
                data=np.ones((self.num_seqs, self.max_genes), dtype=bool),
            )
            gene_out_mask = np.zeros(
                (self.num_seqs, self.max_genes, self.target_length),
                dtype="bool",
            )
            gene_out_mask[:, 0, 50:70] = True
            gene_out_mask[:, 1, 100:120] = True
            gene_out_mask[:, 2, 150:170] = True
            z.create_array("gene_out_mask", data=gene_out_mask)

            # Create targets.txt
            targets_df = pd.DataFrame(
                {
                    "identifier": [f"track{i}" for i in range(self.num_targets)],
                    "description": [f"Track {i}" for i in range(self.num_targets)],
                    "strand_pair": np.arange(self.num_targets),
                },
                index=range(self.num_targets),
            )
            targets_df.to_csv(data_dir / "targets.txt", sep="\t")

            # Create targets_gene.txt
            targets_gene_df = pd.DataFrame(
                {
                    "identifier": [f"rna{i}" for i in range(self.num_gene_targets)],
                    "description": [
                        f"RNA-seq {i}" for i in range(self.num_gene_targets)
                    ],
                },
                index=range(self.num_gene_targets),
            )
            targets_gene_df.to_csv(data_dir / "targets_gene.txt", sep="\t")

            # Create genes.txt
            genes_df = pd.DataFrame(
                {
                    "gene_id": [f"gene{i}" for i in range(self.max_genes)],
                    "gene_name": [f"GENE{i}" for i in range(self.max_genes)],
                }
            )
            genes_df.to_csv(data_dir / "genes.txt", sep="\t", index=False)

            yield tmp_dir

    def test_eval_coverage_only(self, coverage_only_dataset):
        """Test eval() with coverage-only model and data."""
        model_def = self._create_model_def(has_coverage=True, has_gene=False)
        model = SeqNN(model_def)

        dataset = SeqDataset(
            data_dir=coverage_only_dataset,
            split_label="test",
            mode="eval",
        )

        results = model.eval(dataset, batch_size=2)

        # Should only have coverage results
        assert "coverage" in results
        assert "gene" not in results

        cov = results["coverage"]
        assert "r" in cov
        assert "r2" in cov
        assert cov["r"].shape == (self.num_targets,)
        assert cov["r2"].shape == (self.num_targets,)

        # Metrics should be finite
        assert np.all(np.isfinite(cov["r"]))
        assert np.all(np.isfinite(cov["r2"]))

        # No preds/targets without return_values
        assert cov["preds"] is None
        assert cov["targets"] is None

    @pytest.mark.parametrize("subset_metadata", [False, True])
    def test_eval_spec_target_subset(
        self, coverage_only_dataset, monkeypatch, subset_metadata
    ):
        target_slice = np.array([2, 0])
        data = SeqDataset(coverage_only_dataset, split_label="test")
        y = np.random.default_rng(0).gamma(2, size=(3, 1000)).astype(np.float16)
        data.target_hist = np.stack(
            [np.bincount(v.view(np.uint16), minlength=NUM_HIST_BINS) for v in y]
        )
        if subset_metadata:
            data.targets_df = data.targets_df.loc[[0, 2]]

        # Score the small fixture group using the same metric as production.
        spec_metric = metrics.SpecPearsonCorrCoef
        monkeypatch.setattr(
            metrics, "SpecPearsonCorrCoef", partial(spec_metric, group_min=2)
        )
        model = SeqNN(self._create_model_def(), output_slice=target_slice)
        cov = model.eval(data, batch_size=2, return_values=True)["coverage"]
        expected = spec_metric(
            data.targets_df.loc[target_slice],
            data.target_hist[target_slice],
            group_min=2,
        )
        expected.update(
            torch.from_numpy(cov["preds"]), torch.from_numpy(cov["targets"])
        )
        np.testing.assert_allclose(cov["spec"], expected.compute().numpy(), atol=1e-4)
        assert np.isfinite(cov["spec"]).all()

    def test_eval_gene_only(self, gene_only_dataset):
        """Test eval() with gene-only model and data."""
        model_def = self._create_model_def(has_coverage=False, has_gene=True)
        model = SeqNN(model_def)

        dataset = SeqDataset(
            data_dir=gene_only_dataset,
            split_label="test",
            mode="eval",
        )

        results = model.eval(dataset, batch_size=2)

        # Should only have gene results
        assert "coverage" not in results
        assert "gene" in results

        gene = results["gene"]
        assert "r" in gene
        assert "r2" in gene
        assert gene["r"].shape == (self.num_gene_targets,)
        assert gene["r2"].shape == (self.num_gene_targets,)

        # Metrics should be finite
        assert np.all(np.isfinite(gene["r"]))
        assert np.all(np.isfinite(gene["r2"]))

    def test_eval_combined(self, combined_dataset):
        """Test eval() with model having both coverage and gene heads."""
        model_def = self._create_model_def(has_coverage=True, has_gene=True)
        model = SeqNN(model_def)

        dataset = SeqDataset(
            data_dir=combined_dataset,
            split_label="test",
            mode="eval",
        )

        results = model.eval(dataset, batch_size=2)

        # Should have both coverage and gene results
        assert "coverage" in results
        assert "gene" in results

        # Check coverage
        cov = results["coverage"]
        assert cov["r"].shape == (self.num_targets,)
        assert cov["r2"].shape == (self.num_targets,)

        # Check gene
        gene = results["gene"]
        assert gene["r"].shape == (self.num_gene_targets,)
        assert gene["r2"].shape == (self.num_gene_targets,)

    def test_eval_with_zarr_storage(self, combined_dataset):
        """Test eval() with return_values=True and zarr storage."""
        model_def = self._create_model_def(has_coverage=True, has_gene=True)
        model = SeqNN(model_def)

        dataset = SeqDataset(
            data_dir=combined_dataset,
            split_label="test",
            mode="eval",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "eval_results.zarr"

            results = model.eval(
                dataset, batch_size=2, return_values=True, zarr_store=str(zarr_path)
            )

            # Check zarr file was created
            assert zarr_path.exists()

            # Check coverage storage
            cov = results["coverage"]
            assert cov["preds"] is not None
            assert cov["targets"] is not None
            assert cov["preds"].shape == (
                self.num_seqs,
                self.num_targets,
                self.target_length,
            )
            assert cov["targets"].shape == (
                self.num_seqs,
                self.num_targets,
                self.target_length,
            )

            # Check gene storage
            gene = results["gene"]
            assert gene["preds"] is not None
            assert gene["targets"] is not None
            assert gene["masks"] is not None
            assert gene["preds"].shape == (
                self.num_seqs,
                self.num_gene_targets,
                self.max_genes,
            )
            assert gene["targets"].shape == (
                self.num_seqs,
                self.num_gene_targets,
                self.max_genes,
            )
            assert gene["masks"].shape == (self.num_seqs, self.max_genes)

            # Verify zarr structure
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "preds" in root
            assert "targets" in root
            assert "gene_preds" in root
            assert "gene_targets" in root
            assert "gene_presence" in root
            # target chunk caps at num_targets (here 3 < the default 128)
            assert root["preds"].chunks[1] == self.num_targets

    def test_eval_target_chunk(self, coverage_only_dataset):
        """target_chunk chunks the coverage store along the target axis."""
        model = SeqNN(self._create_model_def(has_coverage=True, has_gene=False))
        dataset = SeqDataset(
            data_dir=coverage_only_dataset, split_label="test", mode="eval"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "eval_results.zarr"
            model.eval(
                dataset,
                batch_size=2,
                return_values=True,
                zarr_store=str(zarr_path),
                target_chunk=1,
            )
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["preds"].chunks[1] == 1
            assert root["preds"].shape[1] == self.num_targets

    def test_eval_seq_chunk(self, coverage_only_dataset):
        """seq_chunk widens the store's seq chunk without changing the values.

        seq_chunk=3 with batch_size=2 and num_seqs=4 exercises a batch split
        across a chunk boundary plus a ragged final flush.
        """
        model = SeqNN(self._create_model_def(has_coverage=True, has_gene=False))
        dataset = SeqDataset(
            data_dir=coverage_only_dataset, split_label="test", mode="eval"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base.zarr"
            chunk_path = Path(tmp_dir) / "chunk.zarr"
            base = model.eval(
                dataset, batch_size=2, return_values=True, zarr_store=str(base_path)
            )
            chunked = model.eval(
                dataset,
                batch_size=2,
                return_values=True,
                zarr_store=str(chunk_path),
                seq_chunk=3,
            )
            root = zarr.open_group(str(chunk_path), mode="r")
            assert root["preds"].chunks[0] == 3
            assert root["preds"].shape[0] == self.num_seqs
            np.testing.assert_array_equal(
                base["coverage"]["preds"][:], chunked["coverage"]["preds"][:]
            )
            np.testing.assert_array_equal(
                base["coverage"]["targets"][:], chunked["coverage"]["targets"][:]
            )

    def test_eval_seq_chunk_clamped(self, coverage_only_dataset):
        """seq_chunk exceeding num_seqs is clamped to num_seqs for the store."""
        model = SeqNN(self._create_model_def(has_coverage=True, has_gene=False))
        dataset = SeqDataset(
            data_dir=coverage_only_dataset, split_label="test", mode="eval"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "eval_results.zarr"
            model.eval(
                dataset,
                batch_size=2,
                return_values=True,
                zarr_store=str(zarr_path),
                seq_chunk=self.num_seqs + 100,
            )
            root = zarr.open_group(str(zarr_path), mode="r")
            assert root["preds"].chunks[0] == self.num_seqs

    def test_eval_target_chunk_invalid(self, coverage_only_dataset):
        """A non-positive target_chunk fails fast with a clear error."""
        model = SeqNN(self._create_model_def(has_coverage=True, has_gene=False))
        dataset = SeqDataset(
            data_dir=coverage_only_dataset, split_label="test", mode="eval"
        )
        with pytest.raises(ValueError):
            model.eval(dataset, return_values=True, target_chunk=0)

    def test_eval_seq_chunk_invalid(self, coverage_only_dataset):
        """A non-positive seq_chunk fails fast with a clear error."""
        model = SeqNN(self._create_model_def(has_coverage=True, has_gene=False))
        dataset = SeqDataset(
            data_dir=coverage_only_dataset, split_label="test", mode="eval"
        )
        with pytest.raises(ValueError):
            model.eval(dataset, return_values=True, seq_chunk=0)

    def test_row_writer_buffer_and_skip(self):
        """_RowWriter buffers chunk-aligned blocks and realigns across skips.

        Batches of unequal size cross the chunk boundary; a gap (a batch with no
        coverage) must leave those store rows untouched, not shift later rows.
        """
        from baskerville.seqnn import _RowWriter

        n, w = 10, 3
        arr = np.full((n, w), -1.0, dtype=np.float32)
        rows = np.arange(n * w, dtype=np.float32).reshape(n, w)

        writer = _RowWriter(arr, seq_chunk=4)
        writer.add(0, rows[0:3])
        writer.add(3, rows[3:5])
        writer.add(7, rows[7:10])  # rows 5,6 skipped
        writer.flush()

        for i in (0, 1, 2, 3, 4, 7, 8, 9):
            np.testing.assert_array_equal(arr[i], rows[i])
        for i in (5, 6):  # skipped rows keep the sentinel
            np.testing.assert_array_equal(arr[i], np.full(w, -1.0, dtype=np.float32))

    def test_eval_coverage_zarr_only(self, coverage_only_dataset):
        """Test eval() zarr storage with coverage-only model."""
        model_def = self._create_model_def(has_coverage=True, has_gene=False)
        model = SeqNN(model_def)

        dataset = SeqDataset(
            data_dir=coverage_only_dataset,
            split_label="test",
            mode="eval",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "eval_results.zarr"

            results = model.eval(
                dataset, batch_size=2, return_values=True, zarr_store=str(zarr_path)
            )

            # Check zarr has coverage arrays only
            root = zarr.open_group(str(zarr_path), mode="r")
            assert "preds" in root
            assert "targets" in root
            assert "gene_preds" not in root
            assert "gene_targets" not in root
