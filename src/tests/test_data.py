import numpy as np
import os
from pathlib import Path

import pandas as pd
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


def test_data_fixtures_are_independent(data_me_dir, data_ac_dir):
    """Creating either target dataset must leave the other intact."""
    assert Path(data_me_dir).resolve() != Path(data_ac_dir).resolve()
    for data_dir, kind in [(data_me_dir, "me"), (data_ac_dir, "ac")]:
        expected = pd.read_csv(
            Path(__file__).parent / "data" / f"targets_sc3_{kind}.txt", sep="\t"
        )
        actual = pd.read_csv(Path(data_dir) / "targets.txt", sep="\t")
        assert actual.identifier.tolist() == expected.identifier.tolist()
