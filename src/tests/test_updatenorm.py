#!/usr/bin/env python
"""Test hound_updatenorm script."""

import os
import json
import tempfile
import shutil
import pytest
import torch
import torch.nn as nn

from baskerville.scripts import hound_updatenorm


class SimpleBNModel(nn.Module):
    """Simple model with batch norm for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x, head_index=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MockSeqNN:
    """Mock SeqNN wrapper for testing."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mix_dtype = torch.float32


def test_update_bn_stats():
    """Test that update_bn_stats function updates running statistics."""
    # Create simple model
    model = SimpleBNModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Wrap in MockSeqNN
    seqnn_model = MockSeqNN(model, device)

    # Store initial BN statistics
    initial_mean = model.bn.running_mean.clone()
    initial_var = model.bn.running_var.clone()

    # Create dummy data loader (mimicking MultiDataset format)
    batch_size = 2
    seq_len = 32
    num_batches = 6

    dummy_data = []
    for i in range(num_batches):
        di = torch.tensor([i % 2])  # Dataset index
        x = torch.randn(batch_size, 4, seq_len)
        y = torch.randn(batch_size, 5, seq_len // 2)  # Dummy targets
        dummy_data.append((di, (x, y)))

    class DummyDataLoader:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    data_loader = DummyDataLoader(dummy_data)

    # Update BN statistics
    hound_updatenorm.update_bn_stats(seqnn_model, data_loader, num_batches=num_batches)

    # Check that statistics have changed
    updated_mean = model.bn.running_mean
    updated_var = model.bn.running_var

    # Statistics should have changed (not equal to initial)
    assert not torch.allclose(initial_mean, updated_mean), (
        "Running mean should be updated"
    )
    assert not torch.allclose(initial_var, updated_var), (
        "Running variance should be updated"
    )

    # Statistics should be finite and reasonable
    assert torch.all(torch.isfinite(updated_mean)), "Running mean should be finite"
    assert torch.all(torch.isfinite(updated_var)), "Running variance should be finite"
    assert torch.all(updated_var > 0), "Running variance should be positive"
