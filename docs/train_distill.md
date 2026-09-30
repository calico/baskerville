# hound_train_distill

Train a student model using knowledge distillation with on-the-fly teacher predictions.

## Overview

`hound_train_distill` implements efficient knowledge distillation by computing teacher predictions dynamically during training, rather than pre-computing and storing them. This approach provides:

- **Memory efficiency**: No need to store pre-computed predictions
- **Dynamic augmentation**: Mutations applied to sequences before teacher prediction
- **Flexible ensembles**: Optional random subsampling of teachers per batch
- **Standard training flow**: Uses same Trainer infrastructure as regular training

## Key Differences from hound_train

| Feature        | hound_train                  | hound_train_distill             |
| -------------- | ---------------------------- | ------------------------------- |
| Dataset class  | `SeqDataset`                 | `SeqDatasetTeacher`             |
| Target source  | Pre-stored in data directory | On-the-fly from teacher models  |
| Teacher models | N/A                          | Loaded from models directory    |
| Memory usage   | Fixed (stored targets)       | Dynamic (computed targets)      |
| Augmentation   | Standard (shift, RC)         | Extended (shift, RC, mutations) |

## Usage

Combine all options for robust, efficient training:

```bash
python -m baskerville.scripts.hound_train_distill \
    student_params.json \
    teachers/cross_folds_out \
    data/hg38 \
    data/mm10 \
    -o student_out \
    --teacher_subset 2 \
    --snp_rate 0.05 \
    --rc \
    --head=-1
```

## Arguments

### Positional Arguments

- `params_file`: JSON file with student model parameters (architecture, training config)
- `teachers_dir`: Directory containing teacher models from `hound_train_folds`
  - Expected structure: `teachers_dir/f*c*/train/model_best.pth`
  - Parameters loaded from: `teachers_dir/f0c0/params.json`
- `data_dirs`: One or more data directories containing sequences
  - Targets file not required (teacher predictions used instead)
  - Must contain: `sequences.zarr`, `statistics.json`

### Optional Arguments

- `-o, --out_dir`: Output directory for student model (default: `train_out`)
- `--teacher_subset N`: Randomly sample N teachers per prediction for faster training (default: None, use all)
- `--snp_rate R`: Rate of random nucleotide mutations, 0.0-1.0 (default: 0.0)
- `--rc`: Average forward and reverse complement predictions from teachers (default: False)
- `--head`: Model head selection strategy for both teacher and student
  - `None` (default): Use dataset-specific heads (head index matches dataset index)
  - Integer: Force specific head for all sequences regardless of dataset
  - `-1`: Concatenate all heads

## Teacher Model Requirements

The teacher models directory should follow the structure created by `hound_train_folds`:

```
teachers_dir/
├── f0c0/
│   ├── params.json          # Shared parameters (loaded by distill script)
│   └── train/
│       └── model_best.pth   # Best model from fold 0, cross 0
├── f1c0/
│   └── train/
│       └── model_best.pth   # Best model from fold 1, cross 0
├── f2c0/
│   └── train/
│       └── model_best.pth   # Best model from fold 2, cross 0
...
```

All teachers must:

- Share the same architecture (defined in `f0c0/params.json`)
- Have compatible target definitions
- Be trained on the same sequence length and bin size

## Data Directory Requirements

Unlike `hound_train`, the data directories do **not** need pre-computed targets:

```
data_dir/
├── statistics.json      # Required: Dataset statistics
├── targets.txt          # Required: Target definitions
└── examples/
    ├── train.zarr/
    │   ├── sequence/    # Required: Input sequences
    │   └── target/      # NOT required (teachers compute targets)
    ├── valid.zarr/
    │   ├── sequence/
    │   └── target/
    └── test.zarr/
        ├── sequence/
        └── target/
```

The `targets.txt` file is still needed to:

- Define output tracks for the student model
- Provide strand pair information (if using RC averaging)
- Ensure compatibility between teachers and student

## Multi-Head and Multi-Dataset Training

When teacher models have multiple heads (e.g., for different species) and you provide multiple data directories, the `--head` parameter controls how both teacher predictions and student training use those heads:

### Dataset-Specific Heads (Default)

By default (omit `--head` or use `--head=None`):

```bash
python -m baskerville.scripts.hound_train_distill \
    student_params.json teachers_dir/ data/hg38 data/mm10
```

**Behavior:**

- **Teacher**: Uses head 0 for sequences from `data/hg38`, head 1 for sequences from `data/mm10`
- **Student**: Trains head 0 on human data, head 1 on mouse data
- **Use case**: Multi-species training where each dataset corresponds to a different species head

This is the recommended approach for multi-species distillation, as it preserves the species-specific learned features from the teacher models.

### Forced Single Head

Force a specific head for all sequences:

```bash
--head 0  # Force head 0 for all sequences
```

**Behavior:**

- **Teacher**: Uses only head 0 for predictions, regardless of which dataset the sequence comes from
- **Student**: Trains only head 0 on all sequences
- **Use case**: Single-species distillation, or focusing on one species-specific head

### All Heads Concatenated

Use all heads concatenated:

```bash
--head=-1  # Concatenate all heads
```

**Behavior:**

- **Teacher**: Concatenates predictions from all heads into a single output
- **Student**: Trains on concatenated multi-head predictions
- **Use case**: Pan-species or universal model training that learns from all heads simultaneously

With `--head=-1`, predictions from all heads are concatenated along the target dimension, providing a richer supervision signal that combines species-specific features.

## Training Process

1. **Load teachers**: Discovers all `model_best.pth` files in teachers directory
2. **Initialize datasets**: Creates `SeqDatasetTeacher` for train/validation splits
3. **Initialize student**: Creates student model from `params_file`
4. **Train**: Standard `Trainer.fit()` with on-the-fly teacher predictions

During each training step:

1. Load sequence from zarr
2. Apply standard augmentation (shift, optionally RC)
3. Apply SNP mutations if `snp_rate > 0`
4. Sample subset of teachers if `teacher_subset` specified
5. Compute predictions from selected teachers (using specified head(s))
6. Average predictions (with optional RC)
7. Use averaged predictions as soft targets for student

## Performance Considerations

- **Teacher subset**: Using `--teacher_subset 2` with 8 teachers provides ~4x speedup
- **SNP rate**: Recommend 0.01-0.05 for better generalization
- **Mixed precision**: Teachers use precision from their params.json (float16/bfloat16 for 2x speedup)

## Training Strategy

Distillation training uses the entire dataset (all sequences) without a validation split:

- Goal: Minimize training loss to match teacher predictions
- Teacher predictions serve as soft targets
- No early stopping based on validation loss
- Model selection typically based on final checkpoint or external evaluation

Benefits:

- Maximizes use of available data for learning from teachers
- Simpler training loop (no validation overhead)
- Focus on matching teacher distribution rather than generalization metrics

## Output Structure

```
out_dir/
├── params.json          # Copy of student params
├── model_best.pth       # Best model checkpoint
├── model_last.pth       # Last model checkpoint
└── loss.txt             # Training loss history
```

**Note:** Distillation training uses the entire dataset without validation split, focusing on minimizing training loss against teacher predictions.

## Example Workflow

### Step 1: Train Teacher Ensemble

```bash
# Train 4 teachers using cross-fold validation
python -m baskerville.scripts.hound_train_folds \
    teacher_params.json \
    data/hg38 \
    -o teachers/cross_folds_out \
    -c 1 \
    -f 4
```

This creates: `teachers/cross_folds_out/f[0-3]c0/train/model_best.pth`

### Step 2: Train Student with Distillation

```bash
# Train student using on-the-fly teacher predictions
python -m baskerville.scripts.hound_train_distill \
    student_params.json \
    teachers/cross_folds_out \
    data/hg38 \
    -o student_out \
    --teacher_subset 2 \
    --snp_rate 0.05
```

The trained student model will be saved to `student_out/model_best.pth`.

## On-the-fly vs Pre-computed Distillation

**On-the-fly (this script)**: No storage overhead, dynamic augmentation, single-step training. Requires more GPU memory.

**Pre-computed (hound_data_distill)**: Faster training, lower GPU memory. Requires disk storage and two-step process.

## See Also

- `hound_train_folds` - Train teacher ensemble with cross-fold validation
- `hound_data_distill` - Alternative with pre-computed predictions
- `hound_train` - Standard training without distillation
