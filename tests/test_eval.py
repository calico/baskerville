import os
import pytest
import shutil
import subprocess

import numpy as np
import pandas as pd

@pytest.fixture
def clean_data():
  # download data
  if not os.path.isfile('tests/data/eval/train/model_best.h5'):
    cmd = ['gsutil', '-m', 'cp', '-r', 'gs://westminster2/eval', 'tests/data/']
    subprocess.run(cmd, check=True)

  # clean previous evaluation directories
  out_dirs = ['tests/data/eval/eval_out']
  for out_dir in out_dirs:
    if os.path.exists(out_dir):
      shutil.rmtree(out_dir)


def test_eval(clean_data):
  cmd = ['scripts/hound_eval.py', 
         '-o', 'tests/data/eval/eval_out',
         '--rank',
         'tests/data/eval/params.json',
         'tests/data/eval/train/model_best.h5',
         'tests/data/eval/data']
  print(' '.join(cmd))
  subprocess.run(cmd, check=True)

  metrics_file = 'tests/data/eval/eval_out/acc.txt'
  assert(os.path.exists(metrics_file))
  metrics_df = pd.read_csv(metrics_file, sep='\t')

  mean_pearsonr = metrics_df.pearsonr.mean()
  assert(not np.isnan(mean_pearsonr))
  mean_spearmanr = metrics_df.spearmanr.mean()
  assert(not np.isnan(mean_spearmanr))
 