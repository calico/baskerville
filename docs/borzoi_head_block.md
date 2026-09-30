# Borzoi Head Block Usage

## Overview

The `BorzoiHead` block provides access to the pretrained head of Borzoi (via the Gagneur lab's PyTorch implementation). This block allows you to use either the human or mouse head from any of the 4 available Borzoi model replicates, with optional flash attention support.

## Installation Requirements

Before using the `BorzoiHead` block, you must install the `borzoi_pytorch` package:

```bash
pip install borzoi-pytorch
```

For flash attention support (optional but recommended), also install:

```bash
pip install flash-attn
```

## Block Parameters

### `use_flash_attn` (bool, default: false)

- **Purpose**: Enable flash attention for improved memory efficiency and speed
- **Requirements**: Requires `flash-attn` package to be installed
- **Recommendation**: Set to `true` for better performance if flash-attn is available

### `replicate_index` (int, default: 0)

- **Purpose**: Select which of the 4 available Borzoi model replicates to use
- **Options**: 0, 1, 2, or 3

### `use_human` (bool, default: true)

- **Purpose**: Choose between human and mouse head architectures
- **Options**:
  - `true`: Use human head (7611 output channels)
  - `false`: Use mouse head (2608 output channels)

### `in_channels` (int, required: 1920)

- **Purpose**: Number of input channels (must match the pretrained model)
- **Required Value**: Must be exactly 1920 (will throw an error if different)
- **Note**: This matches the output of BorzoiTrunk

### `out_channels` (int, required: 7611 or 2608)

- **Purpose**: Number of output channels (targets)
- **Required values**:
  - 7611 for human head (`use_human: true`) - will throw an error if different
  - 2608 for mouse head (`use_human: false`) - will throw an error if different
- **Note**: Must exactly match the pretrained model architecture

## Example usage via params.json

### Human Head with Flash Attention, replicate 0

```json
{
  "train" : {
    ...
  },
  "model": {
    "seq_length": 524288,
    "trunk": [
      {
        "name": "BorzoiTrunk",
        "use_flash_attn": true,
        "replicate_index": 0,
        "pool_size": 32,
        "crop_size": 5120
      }
    ],
    "head_data0": {
      "name": "BorzoiHead",
      "use_flash_attn": true,
      "replicate_index": 0,
      "use_human": true,
      "in_channels": 1920,
      "out_channels": 7611
    }
  }
}
```

### Mouse Head without Flash Attention, replicate 2

```json
{
  "train" : {
    ...
  },
  "model": {
    "seq_length": 524288,
    "trunk": [
      {
        "name": "BorzoiTrunk",
        "use_flash_attn": false,
        "replicate_index": 2,
        "pool_size": 32,
        "crop_size": 5120
      }
    ],
    "head_data0": {
      "name": "BorzoiHead",
      "use_flash_attn": false,
      "replicate_index": 2,
      "use_human": false,
      "in_channels": 1920,
      "out_channels": 2608
    }
  }
}
```

### Multiple Heads (e.g., Human + Mouse)

```json
{
  "train" : {
    ...
  },
  "model": {
    "seq_length": 524288,
    "trunk": [
      {
        "name": "BorzoiTrunk",
        "use_flash_attn": true,
        "replicate_index": 0,
        "pool_size": 32,
        "crop_size": 5120
      }
    ],
    "head_human": {
      "name": "BorzoiHead",
      "use_flash_attn": true,
      "replicate_index": 0,
      "use_human": true,
      "in_channels": 1920,
      "out_channels": 7611
    },
    "head_mouse": {
      "name": "BorzoiHead",
      "use_flash_attn": true,
      "replicate_index": 0,
      "use_human": false,
      "in_channels": 1920,
      "out_channels": 2608
    }
  }
}
```

## Notes

- The block is excluded from PyTorch compilation to avoid potential CUDA memory issues with FlashAttention
