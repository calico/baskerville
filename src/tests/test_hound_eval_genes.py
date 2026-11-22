import shutil
import tempfile
from pathlib import Path

import pytest

from baskerville.scripts import hound_eval_genes

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"


@pytest.mark.skipif(
    shutil.which("bedtools") is None,
    reason="bedtools binary not installed (required by pybedtools .intersect())",
)
def test_hound_eval_genes_smoke(model_cov_with_gene_data_dir, data_gene_dir):
    """Smoke test for hound_eval_genes using yeast test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            f"{model_cov_with_gene_data_dir}/params.json",
            f"{model_cov_with_gene_data_dir}/model_best.pth",
            data_gene_dir,
            str(DATA_DIR / "sc3_genes.gtf"),
            "-o",
            tmp_dir,
            "--split",
            "test",
        ]
        hound_eval_genes.main(cmd)

        # Check expected output files exist
        out_path = Path(tmp_dir)
        assert (out_path / "genes.bed").exists()
        assert (out_path / "task_metrics.tsv").exists()
        assert (out_path / "gene_metrics.tsv").exists()
        assert (out_path / "gene_preds.tsv.gz").exists()
        assert (out_path / "gene_targets.tsv.gz").exists()


@pytest.mark.skipif(shutil.which("bedtools") is None, reason="requires bedtools")
@pytest.mark.parametrize("span", [False, True])
def test_gene_summaries_match_concatenated_bins(tmp_path, span):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    import torch

    from baskerville.dataset import untransform_preds
    from baskerville.gene_eval import GeneCoverage

    # Two disjoint exons in the same bin, partial coverage, shuffled BED rows.
    (tmp_path / "sequences.bed").write_text("chr1\t8\t24\ttest\nchr1\t0\t16\ttest\n")
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\tt\texon\t1\t1\t.\t+\t.\tgene_id "g";\n'
        'chr1\tt\texon\t3\t15\t.\t+\t.\tgene_id "g";\n'
        'chr1\tt\texon\t19\t20\t.\t+\t.\tgene_id "overlap1";\n'
        'chr1\tt\texon\t20\t22\t.\t+\t.\tgene_id "overlap2";\n'
    )

    class Data(SimpleNamespace):
        def __len__(self):
            return 2

    data = Data(
        data_dir=tmp_path,
        split_label="test",
        target_length=4,
        pool_width=4,
        data_crop_bp=0,
        extra_crop_bp=0,
        extra_crop_bins=0,
    )
    model = SimpleNamespace(
        output_crop_bp=lambda: 0,
        model=SimpleNamespace(output_stride=4),
        untransform_fn=untransform_preds,
    )
    meta = pd.DataFrame(
        {
            "identifier": ["exp", "constant"],
            "gene": [1, 1],
            "description": ["RNA:a", "RNA:b"],
            "sum_stat": ["sum_sqrt", "sum_sqrt"],
        }
    )
    args = SimpleNamespace(
        out_dir=tmp_path,
        genes_gtf=str(gtf),
        span=span,
        save_span=True,
        pseudo_qtl=0.5,
        seq_step=1,
    )
    scorer = GeneCoverage(data, model, meta, args)
    p = torch.tensor([[[1.01, 2, 3, 4], [1, 1, 1, 1]], [[4, 2, 1, 3], [1, 1, 1, 1]]])
    y = torch.tensor([[[3.02, 1, 2, 1], [1, 1, 1, 1]], [[3, 1, 3, 5], [1, 1, 1, 1]]])

    def reference(v):
        pieces = [v[0, :, :2], v[1]] if span else [v[0, :, :2], v[1, :, :1], v[1]]
        return untransform_preds(torch.cat(pieces, dim=-1).half().float().numpy(), meta)

    for row in (0, 1):
        scorer.update(row, p[row : row + 1], y[row : row + 1])
    pred, target = reference(p), reference(y)
    assert scorer.ids == ["g"]
    length = (15 if span else 14) / 4
    expected = [
        pred.mean(1) * length,
        target.mean(1) * length,
        [pearsonr(np.log2(pred[0] + 1), np.log2(target[0] + 1)).statistic, np.nan],
        target.var(1),
    ]
    np.testing.assert_allclose(scorer.results[:, 0], expected, rtol=1e-6)
    for name, expected_profile in (("preds", pred), ("targets", target)):
        profile = tmp_path / f"gene_within/g/{name}/g_{name}.npy"
        np.testing.assert_array_equal(
            np.load(profile), expected_profile.astype("float16")
        )
    scorer.finish()
    table = pd.read_csv(tmp_path / "gene_targets.tsv.gz", sep="\t", index_col=0)
    np.testing.assert_allclose(
        table.iloc[0], np.log2(2 * target.mean(1) * length + 1), atol=0.0005
    )
