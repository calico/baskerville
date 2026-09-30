import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import zarr

from baskerville.scripts.utils.hound_data_subset_targets import (
    subset_targets_dataset,
)

"""
Test data subsetting functionality using mock dataset.

For manual validation with large datasets, run
`python -m baskerville.scripts.utils.hound_data_subset_targets` directly.
"""


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock dataset files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Dataset parameters
        num_seqs = 8
        seq_length = 131072
        pool_width = 32
        num_targets = 3
        seq_depth = 4
        target_length = seq_length // pool_width

        # Create directory structure
        data_dir = Path(tmp_dir)
        examples_dir = data_dir / "examples"
        examples_dir.mkdir(parents=True)

        # Create statistics.json with fold structure
        stats = {
            "seq_length": seq_length,
            "target_length": target_length,
            "num_targets": num_targets,
            "pool_width": pool_width,
            "seq_depth": seq_depth,
            "fold0_seqs": 4,
            "fold1_seqs": 4,
        }
        with open(data_dir / "statistics.json", "w") as f:
            json.dump(stats, f)

        # Create sequences.bed
        with open(data_dir / "sequences.bed", "w") as f:
            for i in range(num_seqs):
                fold = "fold0" if i < 4 else "fold1"
                f.write(f"chr1\t{i * seq_length}\t{(i + 1) * seq_length}\t{fold}\n")

        # Create fold0.zarr and fold1.zarr
        for fold_name, n_seqs in [("fold0", 4), ("fold1", 4)]:
            zarr_file = examples_dir / f"{fold_name}.zarr"
            zarr_fold = zarr.open(zarr_file, mode="w")
            seq_data = np.random.randint(0, 4, size=(n_seqs, seq_length))
            target_data = np.random.rand(n_seqs, num_targets, target_length)
            zarr_fold.create_array("sequence", data=seq_data.astype("uint8"))
            zarr_fold.create_array("target", data=target_data.astype("float16"))

        # Create targets file with strand_pair
        # Each target pairs with itself for simplicity
        targets_pairs = np.arange(num_targets)
        targets_df = pd.DataFrame(
            {"strand_pair": targets_pairs}, index=range(num_targets)
        )
        targets_df.to_csv(data_dir / "targets.txt", sep="\t")

        yield tmp_dir


def test_data_subset(mock_data_dir, tmp_path):
    """Test data subsetting with mock dataset.

    Creates a mock dataset with 3 targets across 2 folds, subsets to just target indices [0, 2].
    Validates that subset creation and data extraction work correctly.
    """
    # Subset to targets 0 and 2
    target_indices = np.array([0, 2])
    indices_file = tmp_path / "indices.txt"
    np.savetxt(indices_file, target_indices, fmt="%d")

    subset_dir = tmp_path / "subset"

    # Create subset using the actual script function
    subset_targets_dataset(
        og_data_dir=mock_data_dir,
        new_data_dir=str(subset_dir),
        old_indices_file=str(indices_file),
        use_slurm=False,
    )

    # Validate basic files exist
    assert os.path.exists(f"{subset_dir}/statistics.json")
    assert os.path.exists(f"{subset_dir}/targets.txt")
    assert os.path.exists(f"{subset_dir}/sequences.bed")

    # Check statistics updated correctly
    with open(f"{subset_dir}/statistics.json") as f:
        stats = json.load(f)
    assert stats["num_targets"] == 2

    # Check subset targets file
    targets_df = pd.read_table(f"{subset_dir}/targets.txt", sep="\t", index_col=0)
    assert len(targets_df) == 2
    # Check strand_pair remapping: original indices [0,2] with strand_pairs [0,2] -> new [0,1]
    assert list(targets_df["strand_pair"]) == [0, 1]

    # Validate zarr data for both folds
    for fold_name in ["fold0", "fold1"]:
        fold_zarr = f"{subset_dir}/examples/{fold_name}.zarr"
        orig_zarr = f"{mock_data_dir}/examples/{fold_name}.zarr"

        zarr_subset = zarr.open(fold_zarr, mode="r")
        zarr_orig = zarr.open(orig_zarr, mode="r")

        # Check target dimension was subsetted correctly
        assert zarr_subset["target"].shape[1] == 2  # Only 2 targets now
        # Check sequence dimension unchanged
        assert zarr_subset["sequence"].shape[0] == 4  # Still 4 sequences per fold

        # Sequences should be identical
        assert np.array_equal(zarr_orig["sequence"][:], zarr_subset["sequence"][:])

        # Subset targets should match original targets [0, 2]
        expected_targets = zarr_orig["target"][:, target_indices, :]
        assert np.array_equal(expected_targets, zarr_subset["target"][:])
