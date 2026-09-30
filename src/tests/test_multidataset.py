import pytest
from torch.utils.data import Dataset
from typing import List

from baskerville.dataset import MultiDataset


# Simple dataset class for testing
class SimpleDataset(Dataset):
    def __init__(self, data: List[int]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@pytest.fixture
def sample_datasets():
    dataset1 = SimpleDataset([1, 2, 3])
    dataset2 = SimpleDataset([4, 5])
    dataset3 = SimpleDataset([6, 7, 8, 9])
    return [dataset1, dataset2, dataset3]


@pytest.fixture
def multi_dataset(sample_datasets):
    return MultiDataset(sample_datasets)


def test_initialization(multi_dataset):
    """Test that the dataset initializes correctly with cumulative lengths."""
    expected_lengths = [3, 5, 9]  # 3, 3+2, 3+2+4
    assert multi_dataset.cumulative_lengths == expected_lengths


def test_length(multi_dataset):
    """Test that the total length is correct."""
    assert len(multi_dataset) == 9


def test_getitem_first(multi_dataset):
    """Test accessing items from the first dataset."""
    for i in range(3):
        dataset_idx, item = multi_dataset[i]
        assert dataset_idx == 0
        assert item == i + 1


def test_getitem_second(multi_dataset):
    """Test accessing items from the second dataset."""
    for i in range(3, 5):
        dataset_idx, item = multi_dataset[i]
        assert dataset_idx == 1
        assert item == i + 1


def test_getitem_third(multi_dataset):
    """Test accessing items from the third dataset."""
    for i in range(5, 9):
        dataset_idx, item = multi_dataset[i]
        assert dataset_idx == 2
        assert item == i + 1


def test_oob_index(multi_dataset):
    """Test that accessing an out-of-bounds index raises IndexError."""
    with pytest.raises(IndexError):
        multi_dataset[9]


def test_single_dataset():
    """Test initialization with a single dataset."""
    dataset = SimpleDataset([1, 2, 3])
    single_dataset = MultiDataset([dataset])
    assert len(single_dataset) == 3
    dataset_idx, item = single_dataset[0]
    assert dataset_idx == 0
    assert item == 1


def test_dataset_boundaries(multi_dataset):
    """Test accessing items at dataset boundaries."""
    # Last item of first dataset
    dataset_idx, item = multi_dataset[2]
    assert dataset_idx == 0
    assert item == 3

    # First item of second dataset
    dataset_idx, item = multi_dataset[3]
    assert dataset_idx == 1
    assert item == 4


@pytest.mark.parametrize(
    "idx,expected_dataset,expected_item",
    [
        (0, 0, 1),  # First item overall
        (2, 0, 3),  # Last item of first dataset
        (3, 1, 4),  # First item of second dataset
        (4, 1, 5),  # Last item of second dataset
        (5, 2, 6),  # First item of third dataset
        (8, 2, 9),  # Last item overall
    ],
)
def test_specific_indices(multi_dataset, idx, expected_dataset, expected_item):
    """Test specific index cases using parametrize."""
    dataset_idx, item = multi_dataset[idx]
    assert dataset_idx == expected_dataset
    assert item == expected_item
