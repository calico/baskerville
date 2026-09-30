import numpy as np
import os
import zarr


def test_data(data_me_dir):
    stats_json = f"{data_me_dir}/statistics.json"
    assert os.path.exists(stats_json)

    seqs_bed = f"{data_me_dir}/sequences.bed"
    num_seqs = sum(1 for _ in open(seqs_bed))
    assert num_seqs > 0

    test_zarr = f"{data_me_dir}/examples/test.zarr"
    test_zarr_open = zarr.open(test_zarr, mode="r")
    assert test_zarr_open["sequence"].shape[0] > 0

    assert test_zarr_open["target"].shape[0] > 0
    assert test_zarr_open["target"].shape[1] == 2
    target_sums = test_zarr_open["target"][:].sum(axis=(0, 2), dtype="float32")
    assert (target_sums > 0).all()
    assert np.isfinite(target_sums).all()
