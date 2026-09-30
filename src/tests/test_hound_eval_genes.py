import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

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
            "python",
            "-m",
            "baskerville.scripts.hound_eval_genes",
            f"{model_cov_with_gene_data_dir}/params.json",
            f"{model_cov_with_gene_data_dir}/model_best.pth",
            data_gene_dir,
            str(DATA_DIR / "sc3_genes.gtf"),
            "-o",
            tmp_dir,
            "--split",
            "test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        # Check expected output files exist
        out_path = Path(tmp_dir)
        assert (out_path / "genes.bed").exists()
        assert (out_path / "task_metrics.tsv").exists()
        assert (out_path / "gene_metrics.tsv").exists()
        assert (out_path / "gene_preds.tsv.gz").exists()
        assert (out_path / "gene_targets.tsv.gz").exists()
