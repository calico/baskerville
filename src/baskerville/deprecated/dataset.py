import glob
import json

from natsort import natsorted
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import dok_matrix
import zarr

import torch
from torch.utils.data import Dataset, Sampler


class MultiBlockSampler(Sampler):
    """Sampler to ensure that batches are drawn in contiguous blocks from the same dataset."""

    def __init__(
        self,
        multi_dataset,
        batch_size,
        upsampling_rates=None,
        block_sizes=[1, 2, 4, 8],
        block_probs=[0.15, 0.30, 0.40, 0.15],
        mode="train",
    ):
        self.multi_dataset = multi_dataset
        self.batch_size = batch_size
        self.upsampling_rates = upsampling_rates
        if self.upsampling_rates is None:
            self.upsampling_rates = [1 for _ in range(len(self.multi_dataset.datasets))]
        self.block_sizes = block_sizes
        self.block_probs = block_probs
        self.mode = mode

    def __len__(self):
        total_size = 0

        # loop over block sizes
        for block_size, block_prob in zip(self.block_sizes, self.block_probs):
            curr_size = 0
            for dataset, upsampling_rate in zip(
                self.multi_dataset.datasets, self.upsampling_rates
            ):
                curr_size += (
                    len(dataset) // (self.batch_size * block_size)
                ) * upsampling_rate
            total_size += int(curr_size * block_prob) * block_size

        return total_size

    def __iter__(self):
        block_indices_union = []

        # loop over block sizes
        for block_size, block_prob in zip(self.block_sizes, self.block_probs):
            block_indices = []
            examples_start = 0
            for di, [dataset, upsampling_rate] in enumerate(
                zip(self.multi_dataset.datasets, self.upsampling_rates)
            ):
                # make example index list
                num_examples = len(dataset)
                examples_end = examples_start + num_examples

                # repeat (optionally)
                for _ in range(upsampling_rate):
                    examples_indices = list(range(examples_start, examples_end))

                    # shuffle examples
                    if self.mode == "train":
                        np.random.shuffle(examples_indices)

                    # truncate to batch size
                    overhang = num_examples % (self.batch_size * block_size)
                    num_examples_batch = num_examples - overhang
                    examples_indices = examples_indices[:num_examples_batch]

                    # form batch tuples
                    examples_blocks = [
                        examples_indices[i : i + self.batch_size * block_size]
                        for i in range(
                            0, len(examples_indices), self.batch_size * block_size
                        )
                    ]
                    block_indices.extend(examples_blocks)

                # update start
                examples_start = examples_end

            # shuffle dataset blocks of batches
            if self.mode == "train":
                np.random.shuffle(block_indices)

            for batch_i in range(int(len(block_indices) * block_prob)):
                block_indices_union.append(block_indices[batch_i])

        # shuffle (union) dataset blocks of batches
        if self.mode == "train":
            np.random.shuffle(block_indices_union)

        # flatten blocks into batches
        batch_indices = []

        for block_i in range(len(block_indices_union)):
            for i in range(0, len(block_indices_union[block_i]), self.batch_size):
                batch_indices.append(
                    block_indices_union[block_i][i : i + self.batch_size]
                )

        return iter(batch_indices)
