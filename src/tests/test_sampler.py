import pdb
import pytest
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List

from baskerville.dataset import MultiDataset, MultiSampler


class SimpleDataset(Dataset):
    def __init__(self, data: List[int]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@pytest.fixture
def sample_datasets():
    dataset1 = SimpleDataset(list(range(10)))  # 10 items
    dataset2 = SimpleDataset(list(range(20, 25)))  # 5 items
    dataset3 = SimpleDataset(list(range(30, 40)))  # 10 items
    return [dataset1, dataset2, dataset3]


@pytest.fixture
def multi_dataset(sample_datasets):
    return MultiDataset(sample_datasets)


@pytest.fixture
def batch_size():
    return 3


@pytest.mark.parametrize("mode", ["train", "test"])
def test_sampler_length(multi_dataset, batch_size, mode):
    """Test that sampler length is correct (sum of complete batches)."""
    sampler = MultiSampler(multi_dataset, batch_size, mode=mode)
    # Expected batches: dataset1: 3 batches (9 items), dataset2: 1 batch (3 items), dataset3: 3 batches (9 items)
    expected_batches = (10 // batch_size) + (5 // batch_size) + (10 // batch_size)
    assert len(sampler) == expected_batches


def test_batch_size_consistency(multi_dataset, batch_size):
    """Test that all returned batches have the correct size."""
    sampler = MultiSampler(multi_dataset, batch_size)
    batches = list(sampler)

    for batch in batches:
        assert len(batch) == batch_size


def test_indices_within_bounds(multi_dataset, batch_size):
    """Test that all returned indices are within valid range."""
    sampler = MultiSampler(multi_dataset, batch_size)
    total_examples = len(multi_dataset)

    for batch in sampler:
        for idx in batch:
            assert 0 <= idx < total_examples


def test_test_mode_deterministic(multi_dataset, batch_size):
    """Test that test mode produces deterministic ordering."""
    sampler1 = MultiSampler(multi_dataset, batch_size, mode="test")
    sampler2 = MultiSampler(multi_dataset, batch_size, mode="test")

    batches1 = list(sampler1)
    batches2 = list(sampler2)

    assert batches1 == batches2


def test_train_mode_shuffling(multi_dataset, batch_size):
    """Test that train mode shuffles both examples and batches."""
    # Set seeds for reproducibility
    np.random.seed(42)
    sampler1 = MultiSampler(multi_dataset, batch_size, mode="train")
    batches1 = list(sampler1)

    np.random.seed(43)  # Different seed
    sampler2 = MultiSampler(multi_dataset, batch_size, mode="train")
    batches2 = list(sampler2)

    # Batches should be different due to shuffling
    assert batches1 != batches2


def test_same_dataset_batches(multi_dataset, batch_size):
    """Test that indices in each batch come from the same dataset."""
    sampler = MultiSampler(multi_dataset, batch_size)
    batches = list(sampler)

    for batch in batches:
        # Get dataset index for first item in batch
        dataset_idx1, _ = multi_dataset[batch[0]]

        # Check all other items in batch are from same dataset
        for idx in batch[1:]:
            dataset_idx2, _ = multi_dataset[idx]
            assert dataset_idx1 == dataset_idx2


def test_no_index_duplication(multi_dataset, batch_size):
    """Test that no index appears more than once."""
    sampler = MultiSampler(multi_dataset, batch_size)
    all_indices = [idx for batch in sampler for idx in batch]
    unique_indices = set(all_indices)

    assert len(all_indices) == len(unique_indices)


@pytest.mark.parametrize(
    "batch_size,dataset_sizes",
    [
        (2, [4, 6, 8]),  # Even splits
        (3, [5, 7, 10]),  # With remainder
        (4, [3, 7, 9]),  # Some smaller than batch_size
    ],
)
def test_different_sizes(batch_size, dataset_sizes):
    """Test sampler with different batch and dataset sizes."""
    datasets = [SimpleDataset(list(range(size))) for size in dataset_sizes]
    multi_dataset = MultiDataset(datasets)
    sampler = MultiSampler(multi_dataset, batch_size)

    # Check all batches have correct size
    batches = list(sampler)
    assert all(len(batch) == batch_size for batch in batches)

    # Check total sampled examples matches expected
    total_samples = sum(len(batch) for batch in batches)
    expected_samples = sum((size // batch_size) * batch_size for size in dataset_sizes)
    assert total_samples == expected_samples


def test_empty_dataset():
    """Test sampler behavior with an empty dataset."""
    empty_dataset = SimpleDataset([])
    normal_dataset = SimpleDataset([1, 2, 3, 4])
    multi_dataset = MultiDataset([empty_dataset, normal_dataset])
    sampler = MultiSampler(multi_dataset, batch_size=2)

    batches = list(sampler)
    # Should only get batches from non-empty dataset
    assert len(batches) == 2  # Two batches of size 2 from second dataset


def test_batch_truncation(multi_dataset, batch_size):
    """Test that datasets are properly truncated to fit batch size."""
    sampler = MultiSampler(multi_dataset, batch_size)
    batches = list(sampler)

    # Total number of samples should be divisible by batch_size
    total_samples = sum(len(batch) for batch in batches)
    assert total_samples % batch_size == 0


def test_nonuniform_upsampling(multi_dataset, batch_size):
    """Test that nonuniform upsampling rates work correctly."""
    # Use upsampling rates [1, 3, 2] to test different rates for each dataset
    upsampling_rates = [1, 3, 2]
    sampler = MultiSampler(
        multi_dataset, batch_size, upsampling_rates=upsampling_rates, mode="test"
    )

    # Calculate expected length:
    # Dataset 1 (10 items): (10 // 3) * 1 = 3 batches
    # Dataset 2 (5 items): (5 // 3) * 3 = 1 * 3 = 3 batches
    # Dataset 3 (10 items): (10 // 3) * 2 = 3 * 2 = 6 batches
    # Total: 3 + 3 + 6 = 12 batches
    expected_length = sum(
        (len(dataset) // batch_size) * rate
        for dataset, rate in zip(multi_dataset.datasets, upsampling_rates)
    )
    assert len(sampler) == expected_length
    assert len(sampler) == 12

    # Test that all batches have correct size
    batches = list(sampler)
    assert len(batches) == expected_length
    assert all(len(batch) == batch_size for batch in batches)

    # Test that indices are valid
    total_examples = len(multi_dataset)
    for batch in batches:
        for idx in batch:
            assert 0 <= idx < total_examples

    # Test that batches contain indices from the same dataset
    for batch in batches:
        dataset_idx1, _ = multi_dataset[batch[0]]
        for idx in batch[1:]:
            dataset_idx2, _ = multi_dataset[idx]
            assert dataset_idx1 == dataset_idx2


@pytest.mark.parametrize(
    "upsampling_rates",
    [
        [2, 1, 3],  # Different rates
        [1, 1, 1],  # Uniform (should match no upsampling)
        [0, 2, 1],  # Zero upsampling for first dataset
    ],
)
def test_various_upsampling_rates(multi_dataset, batch_size, upsampling_rates):
    """Test various combinations of upsampling rates."""
    sampler = MultiSampler(
        multi_dataset, batch_size, upsampling_rates=upsampling_rates, mode="test"
    )

    # Calculate expected length
    expected_length = sum(
        (len(dataset) // batch_size) * rate
        for dataset, rate in zip(multi_dataset.datasets, upsampling_rates)
    )
    assert len(sampler) == expected_length

    # Test iteration produces correct number of batches
    batches = list(sampler)
    assert len(batches) == expected_length

    # All batches should have correct size
    assert all(len(batch) == batch_size for batch in batches)
