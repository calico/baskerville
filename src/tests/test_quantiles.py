"""
Test quantile computation functionality in snps.py
"""

import numpy as np
import h5py
import tempfile
import os
import pytest

from baskerville.snps import write_quantiles, compute_score_quantiles


class TestQuantiles:
    """Test quantile computation functionality."""

    @pytest.fixture
    def synthetic_scores(self):
        """Create synthetic score data for testing."""
        np.random.seed(42)  # For reproducible results
        n_snps, n_targets = 100, 3

        # Create different distributions for different score types
        scores_sum = np.random.normal(0, 2, (n_snps, n_targets))  # Signed scores
        scores_d1 = np.random.exponential(1, (n_snps, n_targets))  # Unsigned scores

        return {
            "SUM": scores_sum.astype(np.float16),
            "D1": scores_d1.astype(np.float16),
        }

    @pytest.fixture
    def quantile_h5_file(self, synthetic_scores):
        """Create HDF5 file with quantile data for testing."""
        score_types = list(synthetic_scores.keys())

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            h5_path = tmp_file.name

        try:
            # Write synthetic data to HDF5
            with h5py.File(h5_path, "w") as h5f:
                for score_type, scores in synthetic_scores.items():
                    h5f.create_dataset(score_type, data=scores)

            # Run write_quantiles
            with h5py.File(h5_path, "r+") as h5f:
                write_quantiles(h5f, score_types)

            yield h5_path
        finally:
            if os.path.exists(h5_path):
                os.unlink(h5_path)

    def test_write_quantiles(self, synthetic_scores, quantile_h5_file):
        """Test that write_quantiles creates proper quantile datasets."""
        score_types = list(synthetic_scores.keys())

        # Verify quantiles were written correctly
        with h5py.File(quantile_h5_file, "r") as h5f:
            # Check quantiles array exists
            assert "quantiles" in h5f
            quantiles = h5f["quantiles"][:]
            assert len(quantiles) > 100  # Should have many quantile levels
            assert quantiles.min() > 0 and quantiles.max() < 1

            # Check quantile thresholds for each score type
            for score_type in score_types:
                quantiles_key = f"{score_type}_quantiles"
                assert quantiles_key in h5f

                score_quantiles = h5f[quantiles_key][:]
                expected_shape = (synthetic_scores[score_type].shape[1], len(quantiles))
                assert score_quantiles.shape == expected_shape

                # Verify quantiles are sorted
                for target_idx in range(score_quantiles.shape[0]):
                    target_quantiles = score_quantiles[target_idx, :]
                    assert np.all(target_quantiles[:-1] <= target_quantiles[1:])

    def test_compute_score_quantiles_2d(self, synthetic_scores, quantile_h5_file):
        """Test compute_score_quantiles with 2D arrays."""
        # Test compute_score_quantiles
        with h5py.File(quantile_h5_file, "r") as h5f:
            quantiles = h5f["quantiles"][:]

            for score_type, original_scores in synthetic_scores.items():
                quantiles_key = f"{score_type}_quantiles"
                quantile_thresholds = h5f[quantiles_key][:]

                # Test 2D case
                computed_quantiles = compute_score_quantiles(
                    original_scores, quantile_thresholds, quantiles
                )

                # Basic shape and range checks
                assert computed_quantiles.shape == original_scores.shape
                assert np.all(computed_quantiles >= quantiles.min() - 1e-4)
                assert np.all(computed_quantiles <= quantiles.max() + 1e-4)

                # Test that quantiles are monotonic with scores
                for target_idx in range(original_scores.shape[1]):
                    target_scores = original_scores[:, target_idx]
                    target_quantiles = computed_quantiles[:, target_idx]

                    # Sort and check monotonicity
                    sort_idx = np.argsort(target_scores)
                    sorted_quantiles = target_quantiles[sort_idx]
                    assert np.all(sorted_quantiles[:-1] <= sorted_quantiles[1:])

    def test_compute_score_quantiles_1d(self, synthetic_scores, quantile_h5_file):
        """Test compute_score_quantiles with 1D arrays."""
        with h5py.File(quantile_h5_file, "r") as h5f:
            quantiles = h5f["quantiles"][:]

            for score_type, original_scores in synthetic_scores.items():
                quantiles_key = f"{score_type}_quantiles"
                quantile_thresholds = h5f[quantiles_key][:]

                # Test 1D case
                single_target_scores = original_scores[:, 0]
                single_target_thresholds = quantile_thresholds[0, :]
                single_quantiles = compute_score_quantiles(
                    single_target_scores, single_target_thresholds, quantiles
                )

                assert single_quantiles.shape == single_target_scores.shape

                # Compare with 2D result
                computed_quantiles_2d = compute_score_quantiles(
                    original_scores, quantile_thresholds, quantiles
                )
                np.testing.assert_allclose(
                    single_quantiles, computed_quantiles_2d[:, 0], rtol=1e-3, atol=1e-4
                )

    def test_compute_score_quantiles_edge_cases(self):
        """Test edge cases for compute_score_quantiles."""
        quantiles = np.array([0.1, 0.5, 0.9])
        thresholds = np.array([-10, 0, 10])

        # Test extreme values
        extreme_scores = np.array([-100, -5, 5, 100])
        result = compute_score_quantiles(extreme_scores, thresholds, quantiles)

        # Should clip to valid quantile range
        assert np.all(result >= quantiles.min())
        assert np.all(result <= quantiles.max())

        # Test identical scores
        identical_scores = np.array([5.0, 5.0, 5.0, 5.0])
        result = compute_score_quantiles(identical_scores, thresholds, quantiles)

        # All should have the same quantile
        assert np.all(result == result[0])

        # Test single quantile
        single_quantile = np.array([0.5])
        single_threshold = np.array([0.0])
        test_scores = np.array([-1, 0, 1])
        result = compute_score_quantiles(test_scores, single_threshold, single_quantile)

        # All should be the single quantile value
        assert np.all(result == 0.5)
