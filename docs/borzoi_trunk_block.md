# Borzoi Trunk Block Usage

## Overview

The `BorzoiTrunk` block provides access to the pretrained trunk of Borzoi (via the Gagneur lab's PyTorch implementation). User must specify replicate index of the Borzoi trunk, and optionally can specify usage of flash attention.

## Installation Requirements

Before using the `BorzoiTrunk` block, you must install the `borzoi_pytorch` package:

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

## Example usage via params.json

### Without Flash Attention, replicate 0

```json
{
  "train" : {
    ...
  }
  "model": {
    "seq_length": 524288,
    "trunk": [
      {
        "name": "BorzoiTrunk",
        "use_flash_attn": false,
        "replicate_index": 0,
        "pool_size" : 32,
        "crop_size": 5120,
      }
    ],
    "head_data0": {
      "name": "Final",
      "in_channels": 1920,
      "out_channels": ...
    }
  }
}
```

### With Flash Attention, replicate 0

```json
{
  "train" : {
    ...
  }
  "model": {
    "seq_length": 524288,
    "trunk": [
      {
        "name": "BorzoiTrunk",
        "use_flash_attn": false,
        "replicate_index": 0,
        "pool_size" : 32,
        "crop_size": 5120,
      }
    ],
    "head_data0": {
      "name": "Final",
      "in_channels": 1920,
      "out_channels": ...
    }
  }
}
```
