# Update Batch Normalization Statistics

## Overview

The `hound_updatenorm` script updates the running statistics (running mean and running variance) of batch normalization layers in a trained model. This is useful when:

1. You've trained a model with a small batch size and want to update BN statistics with a larger effective batch size
2. You've fine-tuned a model and want to recalibrate BN statistics on new data
3. You need to update BN statistics after modifying the model architecture while preserving weights

## Usage

### Basic Usage

```bash
python -m baskerville.scripts.hound_updatenorm \
  params.json \
  model.pth \
  data_dir/
```

This will:

1. Load the model from `model.pth` with architecture defined in `params.json`
2. Run forward passes on training data from `data_dir/`
3. Update batch normalization running statistics
4. Backup the original model to `model_orig.pth`
5. Replace `model.pth` with the updated version

### Example with model_best.pth

```bash
python -m baskerville.scripts.hound_updatenorm \
  params.json \
  train_out/model_best.pth \
  data_dir/
```

This will backup `model_best.pth` to `model_best_orig.pth` and update `model_best.pth` in place.

### Multiple Data Directories

You can provide multiple data directories, similar to `hound_train`:

```bash
python -m baskerville.scripts.hound_updatenorm \
  params.json \
  model.pth \
  data_dir1/ data_dir2/ data_dir3/
```

### Options

| Option          | Description                                                                                             | Default          |
| --------------- | ------------------------------------------------------------------------------------------------------- | ---------------- |
| `--head`        | Model head(s) to use: None (dataset-specific heads), int (specific head), or -1 (concatenate all heads) | `None`           |
| `-w, --whole`   | Use all data (including validation split)                                                               | `False`          |
| `--num_batches` | Limit number of batches for BN update                                                                   | `None` (use all) |

**Note:** Batch size and number of workers are taken from the `params.json` file (`train.batch_size` and `train.num_workers`), not specified as command-line arguments.

### Examples

**Update BN statistics using only 100 batches:**

```bash
python -m baskerville.scripts.hound_updatenorm \
  --num_batches 100 \
  params.json \
  model.pth \
  data_dir/
```

**Update using all data (train + valid):**

```bash
python -m baskerville.scripts.hound_updatenorm \
  -w \
  params.json \
  train_out/model_best.pth \
  data_dir/
```

**Update with specific model head:**

```bash
python -m baskerville.scripts.hound_updatenorm \
  --head 0 \
  params.json \
  model.pth \
  data_dir/
```

**Update with all heads concatenated:**

```bash
python -m baskerville.scripts.hound_updatenorm \
  --head -1 \
  params.json \
  model.pth \
  data_dir/
```

## How It Works

1. **Load Model**: Restores the trained model weights from the checkpoint file
2. **Set to Train Mode**: Sets the model to training mode so batch norm layers will update their running statistics
3. **Forward Passes**: Runs forward passes on the training data without computing gradients
4. **Update Statistics**: Batch norm layers automatically update their running mean and running variance during forward passes
5. **Save Model**: Saves the model with updated batch norm statistics

## Technical Details

- The script disables gradient computation (`torch.no_grad()`) since we're only updating BN statistics, not training
- The model is set to `train()` mode to enable BN running statistics updates
- After updates are complete, the model is set back to `eval()` mode
- The saved model contains the same weights as the input model, but with updated BN running statistics
