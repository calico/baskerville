import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baskerville.dataset import untransform_preds
from baskerville.gene_eval import GeneCoverage
from baskerville.scripts import hound_eval

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"


@pytest.mark.skipif(
    shutil.which("bedtools") is None,
    reason="bedtools binary not installed (required by pybedtools .intersect())",
)
def test_hound_eval_gtf_smoke(model_cov_with_gene_data_dir, data_gene_dir):
    """Smoke test for hound_eval -g using yeast test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            f"{model_cov_with_gene_data_dir}/params.json",
            f"{model_cov_with_gene_data_dir}/model_best.pth",
            data_gene_dir,
            "-g",
            str(DATA_DIR / "sc3_genes.gtf"),
            "-o",
            tmp_dir,
            "--split",
            "test",
        ]
        hound_eval.main(cmd)

        # Check expected output files exist
        out_path = Path(tmp_dir)
        assert (out_path / "acc.txt").exists()
        assert (out_path / "genes.bed").exists()
        assert (out_path / "acc_covgene.txt").exists()
        assert (out_path / "covgene_metrics.tsv").exists()
        assert (out_path / "covgene_preds.tsv.gz").exists()
        assert (out_path / "covgene_targets.tsv.gz").exists()


needs_bedtools = pytest.mark.skipif(
    shutil.which("bedtools") is None, reason="requires bedtools"
)
META = pd.DataFrame(
    {
        "identifier": ["exp", "constant"],
        "gene": [1, 1],
        "description": ["RNA a", "RNA b"],
        "sum_stat": ["sum_sqrt", "sum_sqrt"],
    }
)


def _scorer(tmp_path, bed, gtf, shards, span=False):
    """GeneCoverage over 16 bp windows of four 4 bp bins."""
    (tmp_path / "sequences.bed").write_text(bed)
    (tmp_path / "genes.gtf").write_text(gtf)

    class Data(SimpleNamespace):
        def __len__(self):
            return 2

    data = Data(
        data_dir=tmp_path,
        split_label="*",
        zarr_files=[str(tmp_path / f"{shard}.zarr") for shard in shards],
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
    args = SimpleNamespace(
        out_dir=tmp_path,
        genes_gtf=str(tmp_path / "genes.gtf"),
        span=span,
        pseudo_qtl=0.5,
        seq_step=1,
    )
    return GeneCoverage(data, model, META, args)


P = torch.tensor([[[1.01, 2, 3, 4], [1, 1, 1, 1]], [[4, 2, 1, 3], [1, 1, 1, 1]]])
Y = torch.tensor([[[3.02, 1, 2, 1], [1, 1, 1, 1]], [[3, 1, 3, 5], [1, 1, 1, 1]]])


@needs_bedtools
@pytest.mark.parametrize("span", [False, True])
def test_gene_summaries_weight_bins_by_window_coverage(tmp_path, span):
    # Overlapping windows; two exons share a bin; rows follow shard order.
    scorer = _scorer(
        tmp_path,
        "chr1\t0\t16\tfold1\nchr1\t40\t56\ttest\nchr1\t8\t24\tfold0\n",
        'chr1\tt\texon\t1\t1\t.\t+\t.\tgene_id "g";\n'
        'chr1\tt\texon\t3\t15\t.\t+\t.\tgene_id "g";\n'
        'chr1\tt\texon\t19\t20\t.\t+\t.\tgene_id "overlap1";\n'
        'chr1\tt\texon\t20\t22\t.\t+\t.\tgene_id "overlap2";\n',
        ["fold0", "fold1"],
        span,
    )
    assert scorer.metadata.group.tolist() == ["*", "*"]  # dataset.target_groups
    for row in (0, 1):
        scorer.update(row, P[row : row + 1], Y[row : row + 1])

    # Gene bins: window 8-24 bins 0-1 and window 0-16 bins 0-3, once each,
    # weighted by 1/(windows covering the bin's exon bases).
    weights = np.array([0.5, 0.5, 1, 1, 0.5, 0.5])

    def reference(v):
        pieces = [v[0, :, :2], v[1]]
        return untransform_preds(torch.cat(pieces, dim=-1).numpy(), META)

    def moments(a, b):
        a, b = a - np.average(a, weights=weights), b - np.average(b, weights=weights)
        return [np.average(x * z, weights=weights) for x, z in ((a, a), (b, b), (a, b))]

    pred, target = reference(P), reference(Y)
    assert scorer.ids == ["g"]
    length = (15 if span else 14) / 4
    var_p, var_y, cov = moments(np.log2(pred[0] + 1), np.log2(target[0] + 1))
    target_mean = np.average(target, axis=1, weights=weights)
    expected = [
        np.average(pred, axis=1, weights=weights) * length,
        target_mean * length,
        [cov / np.sqrt(var_p * var_y), np.nan],
        [moments(target[i], target[i])[0] for i in range(2)],
    ]
    np.testing.assert_allclose([v[0] for v in scorer.summaries()], expected, rtol=1e-6)
    scorer.finish()
    table = pd.read_csv(tmp_path / "covgene_targets.tsv.gz", sep="\t", index_col=0)
    np.testing.assert_allclose(
        table.iloc[0], np.log2(2 * target_mean * length + 1), atol=0.0005
    )


@needs_bedtools
def test_window_without_exon_bases_leaves_summaries_unchanged(tmp_path):
    # Exons at bases 101 and 113. Window 114-130 covers the center (114) of the
    # bin holding base 113 but no exon base. Numeric chromosome, numbered shards.
    gtf = (
        '1\tt\texon\t102\t102\t.\t+\t.\tgene_id "h";\n'
        '1\tt\texon\t114\t114\t.\t+\t.\tgene_id "h";\n'
    )
    summaries = []
    for extra in ("1\t300\t316", "1\t114\t130"):
        bed = f"1\t100\t116\ttrain\n{extra}\ttrain\n"
        scorer = _scorer(tmp_path, bed, gtf, ["train2", "train3"])
        for row in (0, 1):
            scorer.update(row, P[row : row + 1], Y[row : row + 1])
        assert list(scorer.overlaps) == [0]
        summaries.append(scorer.summaries())
    for alone, beside in zip(*summaries):
        np.testing.assert_array_equal(alone, beside)
