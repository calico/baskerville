import torch
import torch.nn as nn
import numpy as np
import pytest
from baskerville.seqnn import SeqNN


def create_tiny_model():
    """Create a tiny SeqNN model for fast testing on CPU."""
    params = {
        "seq_length": 512,  # Very small sequence length
        "verbose": False,  # Disable verbose output
        "trunk": [
            {
                "name": "ConvDNA",
                "out_channels": 8,  # Very small number of channels
                "kernel_size": 3,
                "pool_size": 2,
            },
            {
                "name": "ConvBlock",
                "in_channels": 8,
                "out_channels": 4,  # Even smaller
                "kernel_size": 3,
                "pool_size": 2,
            },
        ],
        "head_data0": {
            "name": "Final",
            "in_channels": 4,
            "num_targets": 2,  # Just 2 targets
        },
    }

    model = SeqNN(params)
    # Force to CPU and eval mode
    model.device = "cpu"
    model.model.to("cpu")
    model.model.eval()

    return model


def legacy_gradients_implementation(
    model,
    x,
    hi=0,
    spatial_slice=None,
    task_slice=None,
    untransform_targets_df=None,
    log_transform=False,
):
    """Reference implementation of the legacy gradients function."""
    model.model.eval()

    # Verify slices are on device
    if spatial_slice is not None and isinstance(spatial_slice, torch.Tensor):
        spatial_slice = spatial_slice.to(model.device)
    if task_slice is not None and isinstance(task_slice, torch.Tensor):
        task_slice = task_slice.to(model.device)

    # Add batch dimension and ensure input requires gradients
    xb = x.unsqueeze(0).to(model.device)
    if not xb.requires_grad:
        xb.requires_grad_(True)

    # Compute ensemble predictions with gradients preserved
    output = model(xb, hi, keep_gradients=True)

    # Extract coverage and remove batch dimension
    yh = output.coverage.squeeze(0)

    # Slice bins first (legacy order)
    if spatial_slice is not None:
        yh = yh[:, spatial_slice]

    # Apply inverse transformations if requested
    if untransform_targets_df is not None:
        from baskerville.dataset import untransform_preds

        yh = untransform_preds(yh, untransform_targets_df)

    # Slice tasks
    if task_slice is not None:
        yh = yh[task_slice, :]

    # Aggregate across spatial and task dimensions
    prediction_agg = yh.sum(dim=1).mean(dim=0)
    if log_transform:
        # Apply log transformation if specified
        prediction_agg = torch.log(prediction_agg + 1e-6)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=prediction_agg,
        inputs=xb,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]

    # Remove batch dimension from gradients to match input shape
    return gradients.squeeze(0)


class TestGradientsLegacy:
    """Test that new gradients function maintains backward compatibility."""

    def test_gradients_no_slicing(self):
        """Test gradients with no spatial or task slicing."""
        model = create_tiny_model()

        # Create random input sequence
        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)  # [channels, seq_len]

        # Compute gradients with both methods
        legacy_grads = legacy_gradients_implementation(model, x)
        new_grads = model.gradients(x)

        # Should be nearly identical
        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_with_spatial_slice(self):
        """Test gradients with spatial slicing."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        # Create spatial slice (select middle portion)
        spatial_slice = slice(32, 96)  # Select bins 32-95

        legacy_grads = legacy_gradients_implementation(
            model, x, spatial_slice=spatial_slice
        )
        new_grads = model.gradients(x, spatial_slice=spatial_slice)

        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_with_task_slice(self):
        """Test gradients with task slicing."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        # Create task slice (select first target only)
        task_slice = slice(0, 1)

        legacy_grads = legacy_gradients_implementation(model, x, task_slice=task_slice)
        new_grads = model.gradients(x, task_slice=task_slice)

        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_with_log_transform(self):
        """Test gradients with log transformation."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        legacy_grads = legacy_gradients_implementation(model, x, log_transform=True)
        new_grads = model.gradients(x, log_transform=True)

        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_with_all_options(self):
        """Test gradients with all legacy options combined."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        spatial_slice = slice(16, 80)
        task_slice = slice(0, 2)  # Both targets

        legacy_grads = legacy_gradients_implementation(
            model,
            x,
            spatial_slice=spatial_slice,
            task_slice=task_slice,
            log_transform=True,
        )
        new_grads = model.gradients(
            x, spatial_slice=spatial_slice, task_slice=task_slice, log_transform=True
        )

        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_boolean_slices(self):
        """Test gradients with boolean slices."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        # Get output shape to create proper boolean slices
        with torch.no_grad():
            yh_shape = (
                model(x.unsqueeze(0)).coverage.squeeze(0).shape
            )  # [targets, bins]

        # Create boolean slices
        spatial_slice = torch.zeros(yh_shape[1], dtype=torch.bool)
        spatial_slice[10:50] = True  # Select bins 10-49

        task_slice = torch.tensor([True, False], dtype=torch.bool)  # First target only

        legacy_grads = legacy_gradients_implementation(
            model, x, spatial_slice=spatial_slice, task_slice=task_slice
        )
        new_grads = model.gradients(
            x, spatial_slice=spatial_slice, task_slice=task_slice
        )

        torch.testing.assert_close(legacy_grads, new_grads, rtol=1e-5, atol=1e-6)

    def test_gradients_shape_consistency(self):
        """Test that gradients have the same shape as input."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        gradients = model.gradients(x)

        assert gradients.shape == x.shape, f"Expected {x.shape}, got {gradients.shape}"
        assert gradients.requires_grad == False, "Gradients should not require grad"

    def test_custom_agg_fn_mimics_legacy_no_log(self):
        """Test that custom agg_fn can mimic legacy spatial+task slicing without log transform."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        # Define slices
        spatial_slice = slice(16, 80)
        task_slice = slice(0, 2)  # Both targets

        # Custom aggregation function that mimics legacy behavior
        def mimic_legacy_agg(yh, spatial_slice, task_slice):
            """Mimic legacy: spatial slice -> task slice -> sum(dim=1).mean(dim=0)"""
            # Apply spatial slice
            if spatial_slice is not None:
                yh = yh[:, spatial_slice]

            # Apply task slice
            if task_slice is not None:
                yh = yh[task_slice, :]

            # Aggregate across spatial and task dimensions (legacy logic)
            return yh.sum(dim=1).mean(dim=0)

        # Compute gradients with both methods
        legacy_grads = model.gradients(
            x, spatial_slice=spatial_slice, task_slice=task_slice, log_transform=False
        )

        custom_grads = model.gradients(
            x,
            agg_fn=mimic_legacy_agg,
            agg_fn_kwargs={"spatial_slice": spatial_slice, "task_slice": task_slice},
        )

        # Should be identical
        torch.testing.assert_close(legacy_grads, custom_grads, rtol=1e-5, atol=1e-6)

    def test_custom_agg_fn_mimics_legacy_with_log(self):
        """Test that custom agg_fn can mimic legacy spatial+task slicing with log transform."""
        model = create_tiny_model()

        torch.manual_seed(42)
        x = torch.randn(4, 512, requires_grad=False)

        # Define slices
        spatial_slice = slice(20, 60)
        task_slice = slice(1, 2)  # Second target only

        # Custom aggregation function that mimics legacy behavior with log
        def mimic_legacy_agg_with_log(yh, spatial_slice, task_slice):
            """Mimic legacy: spatial slice -> task slice -> sum(dim=1).mean(dim=0) -> log"""
            # Apply spatial slice
            if spatial_slice is not None:
                yh = yh[:, spatial_slice]

            # Apply task slice
            if task_slice is not None:
                yh = yh[task_slice, :]

            # Aggregate across spatial and task dimensions
            prediction_agg = yh.sum(dim=1).mean(dim=0)

            # Apply log transformation (same as legacy)
            return torch.log(prediction_agg + 1e-6)

        # Compute gradients with both methods
        legacy_grads = model.gradients(
            x, spatial_slice=spatial_slice, task_slice=task_slice, log_transform=True
        )

        custom_grads = model.gradients(
            x,
            agg_fn=mimic_legacy_agg_with_log,
            agg_fn_kwargs={"spatial_slice": spatial_slice, "task_slice": task_slice},
        )

        # Should be identical
        torch.testing.assert_close(legacy_grads, custom_grads, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    # Run a quick test
    test = TestGradientsLegacy()
    test.test_gradients_no_slicing()
    test.test_gradients_with_spatial_slice()
    test.test_gradients_with_task_slice()
    test.test_gradients_with_log_transform()
    test.test_gradients_with_all_options()
    test.test_gradients_boolean_slices()
    test.test_gradients_shape_consistency()
    test.test_custom_agg_fn_mimics_legacy_no_log()
    test.test_custom_agg_fn_mimics_legacy_with_log()
    print("All tests passed!")
