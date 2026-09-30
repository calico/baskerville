import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from baskerville.trainer import Trainer


def make_trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.params = {}
    trainer.weight_decay = 0.0
    trainer.weight_decay_rules = [
        {"regex": "weight", "weight_decay": 0.0, "lr_mult": 0.1}
    ]
    trainer.learning_rate = 1e-3
    trainer.learning_rate_min = 1e-4
    trainer.optimizer = "adamw"
    trainer.schedule = "cosine"
    trainer.warmup_steps = 2
    trainer.train_batches_max = 10
    trainer.train_epochs_max = 1
    trainer._make_optimizer()
    trainer._make_scheduler()
    return trainer


def step(trainer):
    trainer.optimizer.step()
    trainer.scheduler.step()
    return trainer.scheduler.get_last_lr()


def test_cosine_preserves_group_multiplier():
    trainer = make_trainer()
    expected_base = {2: 1e-3, 6: 5.5e-4, 10: 1e-4}
    for i in range(1, 11):
        scaled, base = step(trainer)
        assert scaled == pytest.approx(0.1 * base)
        if i in expected_base:
            assert base == pytest.approx(expected_base[i])


@pytest.mark.parametrize("legacy", [False, True])
def test_cosine_checkpoint_resume(tmp_path, legacy):
    trainer = make_trainer()
    if legacy:
        # Recreate the scheduler used before per-group cosine scaling.
        for group in trainer.optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        warmup = LinearLR(trainer.optimizer, start_factor=1e-6, total_iters=2)
        cosine = CosineAnnealingLR(trainer.optimizer, T_max=8, eta_min=1e-4)
        trainer.scheduler = SequentialLR(
            trainer.optimizer, [warmup, cosine], milestones=[2]
        )
    for _ in range(5):
        step(trainer)
    checkpoint = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "epoch": 1,
        "valid_best": 0.5,
        "unimproved": 0,
    }
    torch.save(checkpoint, tmp_path / "checkpoint.pth")
    resumed = make_trainer()
    resumed.out_dir = str(tmp_path)
    resumed.reset_unimproved = False
    assert resumed._load_checkpoint() == (1, 0.5, 0)
    assert step(resumed) == pytest.approx([5.5e-5, 5.5e-4])
