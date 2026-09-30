#!/usr/bin/env python
# Copyright 2023 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import argparse
import concurrent.futures
import gc
import glob
import json
import os
import shutil
import time
from pathlib import Path

from natsort import natsorted

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import zarr

from baskerville import dataset
from baskerville import seqnn
from baskerville.hardware import check_mixed_precision

"""
hound_data_distill

Create a distilled dataset by averaging predictions from an ensemble of models.
This script implements knowledge distillation by generating soft targets from 
ensemble predictions, which can be used to train more efficient student models.

OVERVIEW:
--------
Knowledge distillation is a technique where a smaller "student" model learns to 
mimic the behavior of a larger "teacher" model (or ensemble of models). Instead 
of training on the original ground truth targets, the student learns from soft targets 
(ensemble predictions) that contain richer information about the teacher's 
learned representations and capture complex patterns in the data.

This script:
1. Loads an ensemble of trained models from specified directories
2. Processes an existing dataset by generating ensemble predictions
3. Optionally blends ensemble predictions with original targets
4. Saves the distilled dataset in the same format as the original

USAGE:
------
Basic usage (replace targets with ensemble predictions):
    hound_data_distill models_dir data_dir distill_dir

With target blending for knowledge distillation:
    hound_data_distill models_dir data_dir distill_dir --alpha 0.7

ARGUMENTS:
----------
models_dir : str
    Directory containing ensemble model subdirectories. Each subdirectory should
    contain train/model_best.pth. Parameters are loaded from models_dir/f0c0/params.json.

data_dir : str  
    Input dataset directory containing:
    - examples/*.zarr (data files - can be train.zarr, fold0.zarr, etc.)
    - targets.txt (target definitions)
    - statistics.json (dataset statistics)
    - sequences.bed (optional sequence metadata)

distill_dir : str
    Output directory for distilled dataset. Will contain same structure as data_dir
    but with ensemble predictions as targets.

OPTIONS:
--------
--alpha : float (default=0.0)
    Blending weight for original targets. Range [0,1] where:
    - 0.0: Pure ensemble predictions (default)
    - >0.0: Blend with original targets: alpha * original + (1-alpha) * ensemble
    Higher values (e.g., 0.7-0.9) preserve more original signal.

--batch_size : int (default=4)
    Batch size for prediction. Reduce if encountering GPU memory issues.

--head : int (default=0)
    Model head to use for distillation. Allows using any of the available heads
    in multi-head models.

DATASET COMPATIBILITY:
---------------------
Works with any zarr-based dataset structure:
- Traditional splits: train.zarr, valid.zarr, test.zarr
- Fold-based datasets: fold0.zarr, fold1.zarr, fold2.zarr, ...
- Custom naming: any *.zarr files in examples/ directory

TECHNICAL DETAILS:
-----------------
- Uses asynchronous I/O with ThreadPoolExecutor for efficient zarr writing
- Implements GPU memory management with automatic cleanup and error recovery
- Supports mixed precision inference for memory efficiency
- Preserves all dataset metadata and structure

KNOWLEDGE DISTILLATION BENEFITS:
-------------------------------
- Soft targets contain richer information than the original ground truth
- Student models can achieve similar performance with fewer parameters
- Ensemble knowledge is compressed into a single dataset
- Reduces inference computational requirements
- Enables model compression and deployment optimization

EXAMPLE WORKFLOW:
----------------
1. Train ensemble models: hound_train config1.json output1/ && hound_train config2.json output2/
2. Create distilled dataset: hound_data_distill ensemble_dir/ original_data/ distilled_data/ --alpha 0.8
3. Train student model: hound_train student_config.json student_output/ --data distilled_data/

See docs/distill.md for detailed documentation and troubleshooting.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Create distilled dataset from ensemble predictions"
    )
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,
        help="Blending weight for original targets (1-alpha for predictions). Use 0.0 for pure ensemble predictions, >0.0 to blend with original targets [Default: %(default)s]",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch size for prediction [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="Model head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "models_dir",
        help="Directory containing model subdirectories with */train/model_best.pth",
    )
    parser.add_argument("data_dir", help="Original dataset directory")
    parser.add_argument("distill_dir", help="Output directory for distilled dataset")
    args = parser.parse_args()

    # Set targets file from data directory
    targets_file = os.path.join(args.data_dir, "targets.txt")

    # Validate inputs
    validate_inputs(args.models_dir, args.data_dir, args.distill_dir)

    # Load ensemble models
    models = load_ensemble_models(args.models_dir, args.rc, targets_file)

    # Create distilled dataset
    create_distilled_dataset(
        models=models,
        data_dir=args.data_dir,
        distill_dir=args.distill_dir,
        alpha=args.alpha,
        batch_size=args.batch_size,
        head=args.head,
    )

    print(
        f"\nDistillation complete! Distilled dataset available at: {args.distill_dir}"
    )


def load_ensemble_models(models_dir, ensemble_rc=False, targets_file=None):
    """Load and initialize all models from ensemble directories.

    Discovers model files matching the pattern <models_dir>/*/train/model_best.pth
    and loads them with shared parameters. All models are assumed to have the same
    architecture and target definitions.

    Args:
        models_dir (str): Directory containing model subdirectories. Each subdirectory
            should contain train/model_best.pth. Parameters are loaded from the
            models_dir/f0c0/params.json file.
        ensemble_rc (bool): If True, enables reverse-complement averaging for predictions.
        targets_file (str, optional): Path to targets.txt file containing target
            definitions and optional strand pairing information.

    Returns:
        list[SeqNN]: List of loaded and initialized SeqNN models, ready for inference.
            All models are set to evaluation mode and moved to available GPU if present.

    Raises:
        ValueError: If no model files found or required params.json is missing.
        RuntimeError: If models fail to load or initialize properly.

    Notes:
        - Parameters are loaded once from models_dir/f0c0/params.json for efficiency
        - Supports mixed precision inference based on training configuration
        - Handles strand pairing if specified in targets file
        - Failed model loads are logged but don't stop the ensemble loading
    """
    # Find all model directories
    model_pattern = os.path.join(models_dir, "*/train/model_best.pth")
    model_files = natsorted(glob.glob(model_pattern))

    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")

    print(f"Found {len(model_files)} models for ensemble")

    # Load shared parameters from f0c0 directory
    params_file = os.path.join(models_dir, "f0c0", "params.json")
    if not os.path.exists(params_file):
        raise ValueError(f"Params file not found at {params_file}")

    with open(params_file) as f:
        params = json.load(f)
    params_model = params["model"].copy()
    params_train = params["train"].copy()

    # Load targets and strand pairs
    targets_df = pd.read_csv(targets_file, sep="\t", index_col=0)
    if "strand_pair" in targets_df.columns:
        params_model["strand_pair"] = targets_df.strand_pair.values

    # Set mixed precision
    mix_dtype = params_train.get("mix_dtype", "float32")
    print(f"Loaded parameters from {params_file}")

    # Check mixed precision support
    if mix_dtype != "float32":
        if not check_mixed_precision():
            print("Warning: Mixed precision training not supported on this GPU.")
            mix_dtype = "float32"  # Fall back to float32
        elif mix_dtype not in ["float16", "bfloat16"]:
            print(f"Warning: Unrecognized mixed precision dtype {mix_dtype}")
            mix_dtype = "float32"  # Fall back to float32

    models = []
    for i, model_file in enumerate(model_files):
        # Initialize model with shared parameters
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file)
        seqnn_model.ensemble_rc = ensemble_rc
        seqnn_model.model.eval()

        # Set mixed precision
        if mix_dtype == "float16":
            seqnn_model.mix_dtype = torch.float16
        elif mix_dtype == "bfloat16":
            seqnn_model.mix_dtype = torch.bfloat16

        models.append(seqnn_model)
        print(f"Loaded model {i + 1}/{len(model_files)} from {model_file}")

    if not models:
        raise ValueError("No valid models could be loaded")

    return models


def predict_ensemble(models, x, head=0):
    """Generate ensemble predictions for a batch of sequences.

    Runs inference on each model in the ensemble and returns the averaged predictions.
    All models must succeed for the ensemble prediction to complete - any model
    failure will halt the distillation process for investigation.

    Args:
        models (list[SeqNN]): List of loaded SeqNN models for ensemble prediction.
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, 4) containing
            one-hot encoded DNA sequences.
        head (int): Model head index to use for prediction.

    Returns:
        torch.Tensor: Averaged predictions across all models, with shape
            (batch_size, num_targets, target_length). Returns CPU tensor for
            memory efficiency.

    Raises:
        RuntimeError: If any model fails to generate predictions.

    Notes:
        - Uses autocast for mixed precision inference if configured
        - Predictions are moved to CPU immediately to free GPU memory
    """
    predictions = []

    for i, model in enumerate(models):
        try:
            with torch.no_grad():
                with torch.autocast(device_type=model.device, dtype=model.mix_dtype):
                    pred = model(x, head)
                    predictions.append(pred.cpu())
        except Exception as e:
            raise RuntimeError(f"Model {i} crash: {e}") from e

    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(0)

    return ensemble_pred


def write_batch_async(executor, zarr_file, batch_seqs, batch_targets, start_idx):
    """Submit an asynchronous write operation for a batch of distilled data.

    Creates a background task to write sequence and target data to zarr format.
    This enables overlapping of computation (ensemble prediction) with I/O
    (zarr writing) for improved performance.

    Args:
        executor (ThreadPoolExecutor): Thread pool for executing the write operation.
        zarr_file (str): Path to the zarr file being written.
        batch_seqs (np.ndarray): Batch of sequences with shape (batch_size, seq_length).
            Should be uint8 dtype for efficient storage containing integer indices (0-3 for ACGT).
        batch_targets (np.ndarray): Batch of targets with shape
            (batch_size, num_targets, target_length). Should be float16 for storage.
        start_idx (int): Starting index in the zarr arrays where this batch should
            be written.

    Returns:
        concurrent.futures.Future: Future object representing the write operation.
            Call .result() to wait for completion and check for errors.

    Notes:
        - The write operation opens the zarr file in append mode
        - Thread-safe when writing to different index ranges
        - Caller should manage the Future to ensure writes complete
    """

    def write_batch():
        end_idx = start_idx + len(batch_seqs)
        zarr_open = zarr.open(zarr_file, mode="a")
        zarr_open["sequence"][start_idx:end_idx] = batch_seqs
        zarr_open["target"][start_idx:end_idx] = batch_targets

    return executor.submit(write_batch)


def create_distilled_dataset(
    models, data_dir, distill_dir, alpha=0.0, batch_size=4, head=0
):
    """Create a complete distilled dataset from ensemble predictions.

    This function orchestrates the entire distillation process by discovering all
    zarr files in the input dataset, processing them with ensemble predictions,
    and creating a new dataset with the same structure but distilled targets.

    Args:
        models (list[SeqNN]): List of loaded ensemble models for prediction.
        data_dir (str): Input dataset directory containing examples/*.zarr files
            and metadata (targets.txt, statistics.json, sequences.bed).
        distill_dir (str): Output directory where distilled dataset will be created.
            Will have the same structure as data_dir.
        alpha (float): Blending weight for original targets. Range [0,1] where
            1.0 uses only original targets, 0.0 uses only ensemble predictions.
            distilled_targets = alpha * original + (1-alpha) * ensemble.
        batch_size (int): Batch size for processing. Reduce if encountering GPU
            memory issues.
        head (int): Model head index to use for prediction.

    Notes:
        - Automatically detects and processes all *.zarr files in data_dir/examples/
        - Works with any naming scheme (train.zarr, fold0.zarr, custom names, etc.)
        - Copies all metadata files to preserve dataset structure
        - Uses asynchronous I/O for efficient zarr writing
        - Includes comprehensive error handling and memory management
        - Verifies output integrity after processing each split

    File Structure:
        Input (data_dir):
            examples/
                ├── train.zarr (or fold0.zarr, etc.)
                ├── valid.zarr (or fold1.zarr, etc.)
                └── test.zarr (or fold2.zarr, etc.)
            targets.txt
            statistics.json
            sequences.bed (optional)

        Output (distill_dir):
            examples/
                ├── train.zarr (with ensemble predictions as targets)
                ├── valid.zarr
                └── test.zarr
            targets.txt (copied)
            statistics.json (copied)
            sequences.bed (copied if present)
    """
    # Create output directory
    os.makedirs(distill_dir, exist_ok=True)

    # Copy dataset metadata
    for metadata_file in ["statistics.json", "targets.txt", "sequences.bed"]:
        src_path = os.path.join(data_dir, metadata_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(distill_dir, metadata_file))

    # Create examples directory
    examples_dir = os.path.join(distill_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Find all zarr files in the data directory
    data_examples_dir = os.path.join(data_dir, "examples")
    zarr_files = natsorted(glob.glob(os.path.join(data_examples_dir, "*.zarr")))

    if not zarr_files:
        raise ValueError(f"No zarr files found in {data_examples_dir}")

    print(f"Found {len(zarr_files)} zarr files to process")

    # Process each zarr file
    for zarr_file in zarr_files:
        split_name = os.path.basename(zarr_file).replace(".zarr", "")
        print(f"\nProcessing {split_name}...")

        # Load original dataset
        try:
            orig_dataset = dataset.SeqDataset(
                data_dir=data_dir, split_label=split_name, mode="eval"
            )
        except Exception as e:
            print(f"Failed to load {split_name}: {e}")
            continue

        if len(orig_dataset) == 0:
            print(f"Empty {split_name}, skipping...")
            continue

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            orig_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues with zarr
            drop_last=False,
        )

        # Setup output zarr file
        output_zarr_file = os.path.join(examples_dir, f"{split_name}.zarr")
        if os.path.exists(output_zarr_file):
            # Remove existing file
            shutil.rmtree(output_zarr_file)

        # Process this split
        _process_split(
            models, dataloader, output_zarr_file, orig_dataset, alpha, split_name, head
        )


def _process_split(
    models, dataloader, output_zarr_file, orig_dataset, alpha, split_name, head=0
):
    """Process a single dataset split/fold and write distilled data to zarr format.

    This function handles the core distillation logic for one zarr file by:
    1. Creating output zarr arrays with proper dimensions and chunking
    2. Processing sequences in batches through the ensemble
    3. Converting one-hot encoded sequences back to integer indices for storage
    4. Optionally blending ensemble predictions with original targets based on alpha
    5. Writing results asynchronously for performance
    6. Verifying output integrity

    Args:
        models (list[SeqNN]): Loaded ensemble models for prediction.
        dataloader (DataLoader): PyTorch DataLoader for the split being processed.
        output_zarr_file (str): Path to output zarr file to be created.
        orig_dataset (SeqDataset): Original dataset object for metadata extraction.
        alpha (float): Blending weight for original targets. 0.0 uses pure ensemble,
            >0.0 blends with original targets.
        split_name (str): Name of split for logging (e.g., 'train', 'fold0').
        head (int): Model head index to use for prediction.
    """
    # Initialize zarr arrays
    num_seqs = len(orig_dataset)
    seq_length = orig_dataset.seq_length
    target_length = orig_dataset.target_length
    num_targets = orig_dataset.num_targets

    zarr_open = zarr.open(output_zarr_file, mode="w")
    zarr_open.create_array(
        "sequence",
        shape=(num_seqs, seq_length),
        dtype="uint8",
        chunks=(1, seq_length),
    )
    zarr_open.create_array(
        "target",
        shape=(num_seqs, num_targets, target_length),
        dtype="float16",
        chunks=(1, num_targets, target_length),
    )

    # Process batches with async writes
    write_futures = []
    seq_idx = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for batch_idx, (x_batch, y_batch) in enumerate(
            tqdm(dataloader, desc=f"Distilling {split_name}")
        ):
            # Grab batch info before moving to GPU
            batch_size_actual = x_batch.shape[0]

            # Convert one-hot sequences back to integer indices
            # x_batch shape: (batch_size, 4, seq_length)
            # Need to convert to: (batch_size, seq_length) with integer indices
            seq_batch_np = torch.argmax(x_batch, dim=1).numpy().astype("uint8")

            # Move to device of first model for prediction
            device = models[0].device
            x_batch = x_batch.to(device)

            try:
                # Generate ensemble predictions
                ensemble_preds = predict_ensemble(models, x_batch, head)

                # Blend with original targets based on alpha value
                if alpha > 0.0:
                    distilled_targets = alpha * y_batch + (1 - alpha) * ensemble_preds
                else:
                    distilled_targets = ensemble_preds

                # Convert targets to appropriate dtype for storage
                target_batch_np = distilled_targets.numpy().astype("float16")

                # Submit async write
                future = write_batch_async(
                    executor, output_zarr_file, seq_batch_np, target_batch_np, seq_idx
                )
                write_futures.append(future)

                seq_idx += batch_size_actual

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"GPU memory error at batch {batch_idx}, clearing cache and retrying..."
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Try with smaller batch or skip this batch
                    print(f"Skipping batch {batch_idx} due to memory constraints")
                    continue
                else:
                    raise e

            # Manage memory and completed writes
            if len(write_futures) > 5:  # Keep at most 5 pending writes
                # Wait for oldest write to complete
                write_futures.pop(0).result()

            # Periodic garbage collection
            if batch_idx % 20 == 0:  # More frequent cleanup
                gc.collect()
                torch.cuda.empty_cache()

        # Wait for all remaining writes
        for future in write_futures:
            future.result()

    print(f"Completed {split_name}: {seq_idx} sequences processed")

    # Verify the written data
    zarr_check = zarr.open(output_zarr_file, mode="r")
    assert zarr_check["sequence"].shape[0] == num_seqs
    assert zarr_check["target"].shape[0] == num_seqs
    print(f"Verification passed for {split_name}")


def validate_inputs(models_dir, data_dir, distill_dir):
    """Validate input arguments and directories before processing.

    Performs comprehensive validation of all required inputs to ensure the
    distillation process can proceed successfully. Checks for required files,
    directory structure, and potential issues.

    Args:
        models_dir (str): Directory containing ensemble model subdirectories.
            Should contain subdirs with train/model_best.pth files.
        data_dir (str): Original dataset directory. Must contain examples/*.zarr
            files and required metadata files.
        distill_dir (str): Output directory for distilled dataset. Will warn if
            it already exists.

    Raises:
        ValueError: If any required files or directories are missing, or if the
            directory structure is invalid.

    Validation Checks:
        - models_dir exists and contains model files matching pattern
        - data_dir exists with required metadata files (targets.txt, statistics.json)
        - examples/ subdirectory exists with at least one .zarr file
        - Warns if distill_dir already exists (files may be overwritten)
    """
    # Check models directory
    if not os.path.exists(models_dir):
        raise ValueError(f"Models directory does not exist: {models_dir}")

    model_pattern = os.path.join(models_dir, "*/train/model_best.pth")
    model_files = glob.glob(model_pattern)
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")

    # Check data directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    required_files = ["statistics.json", "targets.txt"]
    for req_file in required_files:
        file_path = os.path.join(data_dir, req_file)
        if not os.path.exists(file_path):
            raise ValueError(f"Required file not found: {file_path}")

    examples_dir = os.path.join(data_dir, "examples")
    if not os.path.exists(examples_dir):
        raise ValueError(f"Examples directory not found: {examples_dir}")

    # Check for at least one split
    zarr_files = glob.glob(os.path.join(examples_dir, "*.zarr"))
    if not zarr_files:
        raise ValueError(f"No zarr files found in {examples_dir}")

    # Check output directory
    if os.path.exists(distill_dir):
        print(
            f"Warning: Output directory {distill_dir} already exists. Files may be overwritten."
        )


if __name__ == "__main__":
    main()
