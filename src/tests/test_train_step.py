import json
import pathlib

import numpy as np
import pytest
import torch

from baskerville.dataset import SeqDataset
from baskerville.seqnn import SeqNN
from baskerville.trainer import Trainer

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "params_file,data_fixture",
    [
        ("params_sc3.json", "data_me_dir"),
        ("params_sc3_gene.json", "data_gene_dir"),
        ("params_sc3_gene_only.json", "data_gene_dir"),
    ],
    ids=["coverage", "gene", "gene_only"],
)
def test_trainer_overfits_batch(request, tmp_path, params_file, data_fixture):
    """The real Trainer setup drives loss down on a repeated training batch."""
    data_dir = request.getfixturevalue(data_fixture)
    with open(DATA_DIR / params_file) as f:
        params = json.load(f)

    torch.manual_seed(0)
    np.random.seed(0)
    trainer = Trainer(
        params["train"],
        [SeqDataset(data_dir, "train", mode="train")],
        [SeqDataset(data_dir, "valid", mode="eval")],
        str(tmp_path),
    )
    trainer.model = SeqNN(params["model"]).model.to(trainer.device)
    trainer._setup_optimization()
    trainer._init_metrics()
    trainer.model.train()

    di_tensor, batch_data = next(iter(trainer.train_dataload))
    losses = []
    for _ in range(8):
        trainer.optimizer.zero_grad()
        loss = trainer._train_batch_cov(di_tensor, batch_data)
        trainer._optimizer_step(loss)
        losses.append(loss.item())

    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
