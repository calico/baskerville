import pytest
import torch

from baskerville.metrics import GenePoissonLoss
from baskerville.seqnn import SeqNNMod
from baskerville.trainer import Trainer, parse_loss


@pytest.fixture
def fixed_rng():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        yield


@pytest.mark.parametrize(
    "kind,di",
    [("coverage", 0), ("gene", 0), ("combined", 0), ("mixed", 0), ("mixed", 1)],
    ids=["coverage", "gene", "combined", "mixed-gene", "mixed-coverage"],
)
def test_optimizer_step_reduces_loss(fixed_rng, kind, di):
    cov_head = {"name": "Final", "in_channels": 8, "num_targets": 2}
    gene_head = {
        "name": "FinalGene",
        "in_channels": 8,
        "num_targets": 2,
        "aggregation": "sum",
    }
    heads_cov = None if kind == "gene" else [cov_head]
    heads_gene = None if kind == "coverage" else [gene_head]
    if kind == "mixed":
        heads_cov = [cov_head, cov_head]
        heads_gene = [gene_head, None]

    # Initialize only the trainer state needed for loss and optimization.
    trainer = Trainer.__new__(Trainer)
    trainer.model = (
        SeqNNMod(
            [
                {
                    "name": "ConvBlock",
                    "in_channels": 4,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "pool_size": 2,
                }
            ],
            heads_cov_def=heads_cov,
            heads_gene_def=heads_gene,
            seq_length=32,
        )
        .cpu()
        .float()
        .eval()
    )
    trainer.device = "cpu"
    trainer.loss_fn = parse_loss(
        "poisson_mn", total_weight=0.25, weight_range=10, weight_exp=8
    )
    trainer.loss_gene_fn = GenePoissonLoss()
    trainer.gene_weight = 2.0
    trainer.target_weights = [None, None]
    trainer.gene_target_weights = [None, None]
    trainer.params = {}
    trainer.optimizer = "adamw"
    trainer.learning_rate = 1e-3
    trainer.weight_decay = 0.0
    trainer.weight_decay_rules = []
    trainer.schedule = "constant"
    trainer.mix_dtype = torch.float32
    trainer.global_clipnorm = 1.0
    trainer._make_optimizer()
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda _: 1.0
    )

    x = torch.nn.functional.one_hot(torch.randint(4, (2, 32)), 4)
    x = x.permute(0, 2, 1).float()
    y = torch.rand(2, 2, 16) * 3
    yg = torch.rand(2, 2, 2) * 10
    gene_presence = torch.tensor([[True, True], [True, False]])
    gene_out_mask = torch.zeros(2, 2, 16, dtype=torch.bool)
    gene_out_mask[:, 0, 2:6] = True
    gene_out_mask[:, 1, 9:14] = True
    has_cov = heads_cov is not None
    has_gene = heads_gene is not None and heads_gene[di] is not None

    def loss():
        output = trainer.model(
            x, hi=di, gene_presence=gene_presence, gene_out_mask=gene_out_mask
        )
        assert output.has_coverage == has_cov
        assert output.has_gene == has_gene
        return trainer._compute_loss(
            output,
            y if has_cov else None,
            yg if has_gene else None,
            gene_presence,
            torch.tensor(1.0),
            di,
        )

    trainer.optimizer.zero_grad()
    before = loss()
    assert torch.isfinite(before)
    trainer._optimizer_step(before)
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    with torch.no_grad():
        after = loss()
    assert torch.isfinite(after)
    assert after < before.detach(), (
        f"Loss did not decrease: {before.item()} -> {after.item()}"
    )
