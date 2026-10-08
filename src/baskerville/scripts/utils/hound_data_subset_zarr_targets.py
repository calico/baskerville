import argparse
import os
import pandas as pd
import numpy as np
import tqdm
import torch
import json
import zarr

from baskerville import dataset

"""
Example usage:

    python subset_zarr.py \
        --fold_index 0 \
        --old_data_dir /path/to/old/data/dir \
        --new_data_dir /path/to/new/data/dir \
        --num_workers 4 \
        --old_indices /path/to/old/indices.txt \
        --test_mode

"""


class IndexedSeqDataset(dataset.SeqDataset):
    """Same as SeqDataset, but __getitem__ returns (idx, seq_indexes, targets)."""

    def __getitem__(self, idx):
        if self.zarr_data is None:
            self._open_zarr()

        # determine zarr indexes
        zi = np.searchsorted(self.zarr_cumseqs[1:], idx, side="right")
        si = idx - self.zarr_cumseqs[zi]

        # read example
        seq_indexes = self.zarr_data[zi]["sequence"][si]
        targets = self.zarr_data[zi]["target"][si]

        return idx, seq_indexes, targets


def subset_zarr_for_fold(
    fold_index,
    old_data_dir,
    new_data_dir,
    old_indices_file,
    num_workers=4,
    batch_size=32,
    test_mode=False,
):
    """
    Create subset zarr for a single fold.

    Args:
        fold_index (int): Fold index to process
        old_data_dir (str): Original data directory
        new_data_dir (str): New data directory to create
        old_indices_file (str): File containing target indices to keep
        num_workers (int): Number of workers for data loading
        batch_size (int): Batch size for processing
        test_mode (bool): If True, process only first 100 examples
    """
    fi = fold_index
    og_data_dir = old_data_dir

    # --- GET THE STATS ---

    # ** load the old indices **
    old_indices = np.loadtxt(old_indices_file).astype(int)

    # ** read in the stats from the OG dir **
    with open(f"{og_data_dir}/statistics.json") as stats_open:
        stats = json.load(stats_open)
    seq_length = stats["seq_length"]
    target_length = stats["target_length"]
    num_targets = len(old_indices)
    fold_seqs = stats[f"fold{fi}_seqs"]
    chunk_length = target_length

    # --- SETUP NEW ZARR FOLDER ---

    # ** get new examples dir **
    examples_dir = f"{new_data_dir}/examples"
    os.makedirs(examples_dir, exist_ok=True)

    # ** set up compressor **
    compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")

    # ** init the zarr **
    fold_zarr_file = f"{examples_dir}/fold{fi}.zarr"
    fold_zarr_root = zarr.open_group(fold_zarr_file, mode="a")
    if not os.path.isdir(f"{fold_zarr_file}/sequence"):
        fold_zarr_root.create_array(
            "sequence",
            shape=(fold_seqs, seq_length),
            chunks=(1, seq_length),
            dtype="uint8",
        )
    if not os.path.isdir(f"{fold_zarr_file}/target"):
        fold_zarr_root.create_array(
            "target",
            shape=(fold_seqs, num_targets, target_length),
            chunks=(1, num_targets, chunk_length),
            dtype="float16",
            compressors=compressors,
        )

    # ** keep the kept tracks' histograms (of strand sums, so pairs must stay) **
    old_root = zarr.open_group(f"{og_data_dir}/examples/fold{fi}.zarr", mode="r")
    if "target_hist" in old_root:
        old_targets = pd.read_csv(f"{og_data_dir}/targets.txt", sep="\t", index_col=0)
        pair = dataset.strand_pair_indices(old_targets)
        if not np.isin(pair[old_indices], old_indices).all():
            raise ValueError("old_indices split a strand pair; target_hist would be wrong")
        hist = old_root["target_hist"].oindex[old_indices, :]
        fold_zarr_root.create_array(
            "target_hist",
            shape=hist.shape,
            dtype=hist.dtype,
            chunks=(1, hist.shape[1]),
            overwrite=True,
        )[:] = hist

    # --- CREATE DATALOADER ---
    dataset_obj = IndexedSeqDataset(og_data_dir, split_label=f"fold{fi}", mode="eval")
    dataloader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
        drop_last=False,
    )

    # --- CREATE ZARR ---
    z = 0
    zarr_open = zarr.open(fold_zarr_file, mode="a")
    for ex in tqdm.tqdm(dataloader, desc=f"Processing fold {fi}"):
        # Handle batch data - ex[0] is sample indices, ex[1] is sequences, ex[2] is targets
        sample_indices = ex[0].numpy()  # Shape: (batch_size,)
        sequences = ex[1].numpy().astype("uint8")  # Shape: (batch_size, seq_length)
        targets = (
            ex[2][:, old_indices, :].numpy().astype("float16")
        )  # Shape: (batch_size, num_targets, target_length)

        # Write batch to zarr using vectorized indexing
        zarr_open["sequence"][sample_indices] = sequences
        zarr_open["target"][sample_indices] = targets

        # increment by batch size
        z += len(sample_indices)

        # break at 100 if we're working with test mode
        if test_mode and z >= 100:
            break


def main():
    """Main function for command-line usage."""

    # --- PARSE ARGS ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--old_data_dir", type=str, required=True)
    parser.add_argument("--new_data_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--old_indices", type=str, required=True)
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    # Call the subset function
    subset_zarr_for_fold(
        fold_index=args.fold_index,
        old_data_dir=args.old_data_dir,
        new_data_dir=args.new_data_dir,
        old_indices_file=args.old_indices,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
