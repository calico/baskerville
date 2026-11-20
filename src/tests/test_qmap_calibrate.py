"""hound_qmap_calibrate end to end on the sc3 test data and static model."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from baskerville.scripts import hound_qmap_calibrate
from tests.conftest import DATA_DIR


def test_qmap_calibrate(tmp_path, data_me_dir, static_model_dir, capsys, monkeypatch):
    # one group of both tracks, so the qmap term has a group to act on
    data_dir = tmp_path / "data"
    shutil.copytree(data_me_dir, data_dir)
    targets = pd.read_csv(data_dir / "targets.txt", sep="\t", index_col=0)
    targets["group"] = "CHIP"
    targets.to_csv(data_dir / "targets.txt", sep="\t")

    params = json.loads((DATA_DIR / "params_sc3.json").read_text())
    params["train"] |= {"spec_group_min": 2, "qmap_form": "proj"}
    params["train"].update(extra_crop_bp=32, extra_crop_bins=1)
    measure = hound_qmap_calibrate.gradient_ratios

    def measure_crop(model, data, *args):
        item = data[0]
        assert item.sequence.shape[-1] == 16384 - 64
        assert item.coverage_targets.shape[-1] == data.target_length == 512 - 2
        return measure(model, data, *args)

    monkeypatch.setattr(hound_qmap_calibrate, "gradient_ratios", measure_crop)
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))

    lams = {}
    for push in [0.1, 0.2]:
        hound_qmap_calibrate.main(
            [
                "--batches",
                "4",
                "--push",
                str(push),
                str(params_file),
                f"{static_model_dir}/sc3_model.pth",
                str(data_dir),
            ]
        )
        out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert out["qmap_form"] == "proj"
        lams[push] = out["qmap_weight"]

    assert len(lams[0.1]) == 1
    assert np.isfinite(lams[0.1][0]) and lams[0.1][0] > 0
    np.testing.assert_allclose(lams[0.2], 2 * np.array(lams[0.1]), rtol=1e-3)

    for median in (0, np.nan, np.inf):
        monkeypatch.setattr(
            hound_qmap_calibrate, "gradient_ratios", lambda *args: np.array([median])
        )
        with pytest.raises(SystemExit):
            hound_qmap_calibrate.main(
                [str(params_file), f"{static_model_dir}/sc3_model.pth", str(data_dir)]
            )
        error = capsys.readouterr().err
        assert str(data_dir) in error and "finite and positive" in error
