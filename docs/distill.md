# Ensemble Distillation

This guide demonstrates how to create distilled datasets from ensemble models for knowledge distillation training.

## Overview

Ensemble distillation combines predictions from multiple trained models to create "soft targets" that provide richer supervision signals than hard labels. This technique often leads to improved model performance and faster convergence when training new models.

## Prepare Ensemble Models

First, ensure you have multiple trained models in separate directories, each containing a `train/model_best.pth` file:

```
ensemble_models/
├── model1/train/model_best.pth
├── model2/train/model_best.pth
├── model3/train/model_best.pth
└── ...
```

Each model directory should also contain the corresponding `params.json` file (either in the `train/` directory or the parent directory).

## Create Distilled Dataset

Generate the distilled dataset using:

```bash
hound_data_distill ensemble_models/ data_me distilled_data_me
```

This will:

- Load all models from `ensemble_models/*/train/model_best.pth`
- Process each sequence in the original dataset
- Generate ensemble predictions by averaging across all models
- Save the distilled dataset in the same format as the original

## Command Line Options

### Basic Options

- `--batch_size`: Adjust batch size based on GPU memory (default: 2)

### Target Blending

- `--alpha`: Weight for original targets when blending (default: 0.0)

The blending behavior is controlled by the `alpha` parameter:

- `alpha=0.0`: Pure ensemble predictions (default behavior)
- `alpha>0.0`: Blend with original targets

When `alpha > 0.0`, the final targets are computed as:

```
distilled_targets = alpha × original_targets + (1 - alpha) × ensemble_predictions
```

### Example with Options

```bash
hound_data_distill --alpha 0.7 --batch_size 8 ensemble_models/ data_me distilled_data_me
```

This creates distilled targets that are 70% original targets and 30% ensemble predictions, using a batch size of 8.

## Memory Management

The script includes several features to handle large datasets and models:

- **Asynchronous I/O**: Writes are performed in background threads
- **Memory monitoring**: Automatic garbage collection and GPU cache clearing
- **Error recovery**: Handles GPU out-of-memory errors gracefully
- **Batch processing**: Configurable batch sizes to fit available memory

## Train with Distilled Data

Use the distilled dataset like any other dataset for training:

```bash
hound_train -o distill_train_out params.json distilled_data_me
```

## Benefits of Knowledge Distillation

- **Improved Performance**: Soft targets provide more informative gradients
- **Faster Convergence**: Models often train faster with distilled targets
- **Model Compression**: Train smaller models that approach ensemble performance
- **Regularization**: Ensemble knowledge acts as implicit regularization

## Technical Details

### Model Loading

The script automatically discovers models using the pattern `models_dir/*/train/model_best.pth` and loads the corresponding parameters from `params.json` files.

### Prediction Pipeline

1. Load all ensemble models into GPU memory
2. Process dataset in batches
3. Generate predictions from each model
4. Average predictions across ensemble
5. Optionally blend with original targets
6. Write results asynchronously to Zarr format

### Error Handling

- **Input Validation**: Checks for required files and directories
- **GPU Memory**: Graceful handling of out-of-memory errors
- **Model Failures**: Continues if individual models fail
- **Write Verification**: Validates output after processing each split

## Troubleshooting

### GPU Memory Issues

If you encounter out-of-memory errors:

- Reduce `--batch_size`
- Use fewer ensemble models
- Enable gradient checkpointing in model training

### Slow Processing

To improve processing speed:

- Increase `--batch_size` if memory allows
- Use faster storage (SSD vs HDD)
- Ensure sufficient RAM for dataset caching

### Model Compatibility

Ensure all ensemble models:

- Use the same sequence length
- Have compatible target dimensions
- Use consistent strand pairing schemes
