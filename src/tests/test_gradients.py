"""
Test gradients() method for SeqNN models.
"""

import json
import os
import pytest
import torch
import numpy as np

from baskerville.seqnn import SeqNN


@pytest.fixture
def params():
    """Load parameters from params_sc3.json file."""
    params_file = os.path.join(os.path.dirname(__file__), "data", "params_sc3.json")
    with open(params_file) as f:
        params = json.load(f)
    return params["model"]


def test_gradients_basic(params):
    """Test basic functionality of gradients() method."""
    # Create model with sc3 parameters
    model = SeqNN(params)

    # Create test input without batch dimension (channels, seq_length)
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Test basic gradients computation
    gradients = model.gradients(x, hi=0)

    # Check output shape matches input
    assert gradients.shape == x.shape, (
        f"Expected shape {x.shape}, got {gradients.shape}"
    )

    # Check gradients are not all zeros
    assert not torch.allclose(gradients, torch.zeros_like(gradients)), (
        "Gradients should not be all zeros"
    )

    # Check gradients are finite
    assert torch.isfinite(gradients).all(), "Gradients should be finite"


def test_gradients_slicing(params):
    """Test gradients() method with spatial and task slicing."""
    # Create model with sc3 parameters
    model = SeqNN(params)

    # Create test input without batch dimension
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Get model output dimensions
    output_length = model.output_length()
    output_depth = model.output_depth(hi=0)

    # Test spatial slicing with integer slice
    spatial_slice = slice(10, min(20, output_length))
    grads_spatial = model.gradients(x, hi=0, spatial_slice=spatial_slice)
    assert grads_spatial.shape == x.shape

    # Test task slicing with integer slice
    task_slice = slice(0, min(2, output_depth))  # First 2 tasks or less
    grads_task = model.gradients(x, hi=0, task_slice=task_slice)
    assert grads_task.shape == x.shape

    # Test both spatial and task slicing
    grads_both = model.gradients(
        x, hi=0, spatial_slice=spatial_slice, task_slice=task_slice
    )
    assert grads_both.shape == x.shape

    # Test boolean mask slicing
    spatial_mask = torch.zeros(output_length, dtype=torch.bool)
    if output_length > 10:
        spatial_mask[5 : min(15, output_length)] = True
    else:
        spatial_mask[0] = True  # At least one position
    grads_mask = model.gradients(x, hi=0, spatial_slice=spatial_mask)
    assert grads_mask.shape == x.shape

    # Test task boolean mask
    task_mask = torch.zeros(output_depth, dtype=torch.bool)
    task_mask[0] = True  # Only first task
    grads_task_mask = model.gradients(x, hi=0, task_slice=task_mask)
    assert grads_task_mask.shape == x.shape


def test_gradients_ensemble(params):
    """Test gradients() method with ensemble augmentation."""
    # Create model with sc3 parameters (use smaller seq_length for faster testing)
    params = params.copy()
    params["seq_length"] = 1024  # Smaller for faster testing
    model = SeqNN(params)

    # Create test input without batch dimension
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Test without ensemble (baseline)
    model.ensemble_shifts = [0]
    model.ensemble_rc = False
    grads_no_ensemble = model.gradients(x, hi=0)

    # Test with shift ensemble
    model.ensemble_shifts = [0, 1, -1]
    model.ensemble_rc = False
    grads_shifts = model.gradients(x, hi=0)

    # Test with reverse complement ensemble
    model.ensemble_shifts = [0]
    model.ensemble_rc = True
    grads_rc = model.gradients(x, hi=0)

    # Test with both shifts and reverse complement
    model.ensemble_shifts = [0, 1, -1]
    model.ensemble_rc = True
    grads_full_ensemble = model.gradients(x, hi=0)

    # All should have the same shape
    assert grads_no_ensemble.shape == x.shape
    assert grads_shifts.shape == x.shape
    assert grads_rc.shape == x.shape
    assert grads_full_ensemble.shape == x.shape

    # Gradients should be different with different ensemble settings
    assert not torch.allclose(grads_no_ensemble, grads_shifts, atol=1e-6), (
        "Gradients should differ with different shift ensembles"
    )
    assert not torch.allclose(grads_no_ensemble, grads_rc, atol=1e-6), (
        "Gradients should differ with reverse complement ensemble"
    )


def test_gradients_consistency(params):
    """Test that gradients are consistent and reproducible."""
    # Create model with sc3 parameters
    params = params.copy()
    params["seq_length"] = 512  # Smaller for fast testing
    model = SeqNN(params)

    # Create test input
    seq_length = params["seq_length"]
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(4, seq_length, requires_grad=True)

    # Compute gradients twice
    grads1 = model.gradients(x, hi=0)
    grads2 = model.gradients(x, hi=0)

    # Should be identical (deterministic)
    assert torch.allclose(grads1, grads2), "Gradients should be deterministic"

    # Test that different inputs produce different gradients
    x2 = torch.randn(4, seq_length, requires_grad=True)
    grads_diff = model.gradients(x2, hi=0)

    assert not torch.allclose(grads1, grads_diff, atol=1e-6), (
        "Different inputs should produce different gradients"
    )


def test_gradients_magnitude(params):
    """Test that gradients have reasonable magnitudes."""
    # Create model with sc3 parameters
    params = params.copy()
    params["seq_length"] = 512  # Smaller for fast testing
    model = SeqNN(params)

    # Create test input
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Compute gradients
    gradients = model.gradients(x, hi=0)

    # Check gradient statistics
    grad_mean = gradients.mean().item()
    grad_std = gradients.std().item()
    grad_max = gradients.max().item()
    grad_min = gradients.min().item()

    # Gradients should have reasonable magnitude (not too large or too small)
    assert abs(grad_mean) < 10.0, f"Gradient mean too large: {grad_mean}"
    assert grad_std > 1e-8, f"Gradient std too small: {grad_std}"
    assert grad_std < 100.0, f"Gradient std too large: {grad_std}"
    assert abs(grad_max) < 1000.0, f"Gradient max too large: {grad_max}"
    assert abs(grad_min) < 1000.0, f"Gradient min too large (negative): {grad_min}"


def test_gradients_device_compatibility(params):
    """Test that gradients work on different devices."""
    # Create model with sc3 parameters
    params = params.copy()
    params["seq_length"] = 256  # Very small for fast testing
    model = SeqNN(params)

    # Create test input
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Test on CPU
    model.set_device("cpu")
    x_cpu = x.to("cpu")
    x_cpu.requires_grad_(True)
    grads_cpu = model.gradients(x_cpu, hi=0)
    assert grads_cpu.device.type == "cpu"

    # Test on CUDA if available
    if torch.cuda.is_available():
        model.set_device("cuda")
        x_cuda = x.to("cuda")
        x_cuda.requires_grad_(True)
        grads_cuda = model.gradients(x_cuda, hi=0)
        assert grads_cuda.device.type == "cuda"

        # Results should be similar between devices (within numerical precision)
        assert torch.allclose(grads_cpu, grads_cuda.cpu(), atol=1e-4), (
            "Gradients should be similar across devices"
        )


# Test with pre-trained model if available (optional fixture-based test)
def test_gradients_with_pretrained_model(model_dir):
    """Test gradients() method with a pre-trained model from fixtures."""
    try:
        # Load model parameters
        params_file = os.path.join(model_dir, "params.json")
        with open(params_file) as f:
            params = json.load(f)

        # Create model
        model = SeqNN(params["model"])

        # Restore trained weights
        model_file = os.path.join(model_dir, "model_best.pth")
        if os.path.exists(model_file):
            model.restore(model_file)

        # Create test input
        seq_length = params["model"]["seq_length"]
        x = torch.randn(4, seq_length, requires_grad=True)

        # Test gradients computation
        gradients = model.gradients(x, hi=0)

        # Check output shape matches input
        assert gradients.shape == x.shape, (
            f"Expected shape {x.shape}, got {gradients.shape}"
        )

        # Check gradients are finite
        assert torch.isfinite(gradients).all(), "Gradients should be finite"

    except (FileNotFoundError, KeyError):
        pytest.skip("Pre-trained model not available, skipping this test")


def test_gradients_untransform(params):
    """Test gradients() method with untransform functionality."""
    import pandas as pd

    # Create model with sc3 parameters
    params = params.copy()
    params["seq_length"] = 512
    model = SeqNN(params)

    # Create test input without batch dimension
    seq_length = params["seq_length"]
    x = torch.randn(4, seq_length, requires_grad=True)

    # Load real targets_df from test data
    targets_file = os.path.join(os.path.dirname(__file__), "data", "targets_sc3_ac.txt")
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)

    # Verify targets_df has required columns
    required_cols = ["scale", "sum_stat", "clip_soft"]
    for col in required_cols:
        assert col in targets_df.columns, f"Missing required column: {col}"

    # Get model output depth and ensure targets_df matches
    output_depth = model.output_depth(hi=0)
    print(f"Model output_depth: {output_depth}, targets_df length: {len(targets_df)}")

    # Assert that targets_df matches the model output depth exactly
    assert len(targets_df) == output_depth, (
        f"Targets file length ({len(targets_df)}) must match model output depth ({output_depth})"
    )

    # Test without untransform
    grads_normal = model.gradients(x, hi=0)
    print(f"Gradients shape: {grads_normal.shape}")

    # Get a prediction to check its shape before untransform
    with torch.no_grad():
        output = model(x.unsqueeze(0), hi=0)  # Add batch dim for prediction
        print(f"Prediction shape: {output.coverage.shape}")

    # Test with untransform
    grads_untransform = model.gradients(x, hi=0, untransform_targets_df=targets_df)

    # Check shapes are the same
    assert grads_normal.shape == grads_untransform.shape

    # Check gradients are different (due to untransform)
    assert not torch.allclose(grads_normal, grads_untransform, atol=1e-6), (
        "Gradients should be different with and without untransform"
    )

    # Check gradients are finite
    assert torch.isfinite(grads_untransform).all(), (
        "Untransformed gradients should be finite"
    )

    # Create subset of targets_df matching the task slice
    task_slice = [0]  # Use first task
    targets_df_subset = targets_df.iloc[task_slice].copy()
    grads_sliced = model.gradients(
        x, hi=0, task_slice=task_slice, untransform_targets_df=targets_df_subset
    )

    # Check shape is still correct
    assert grads_sliced.shape == x.shape

    # Check gradients are finite
    assert torch.isfinite(grads_sliced).all(), (
        "Sliced untransformed gradients should be finite"
    )
