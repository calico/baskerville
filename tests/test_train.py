import os
import pytest
import shutil
import subprocess

@pytest.fixture
def clean_data():
  # download data
  if not os.path.isfile('tests/data/tiny/nets0.bed'):
    cmd = ['gsutil', '-m', 'cp', '-r', 'gs://westminster2/tiny', 'tests/data/']
    subprocess.run(cmd, check=True)

  # setup folds
  # TODO: update basenji_train_folds.py to baskerville version
  if not os.path.isdir('tests/data/train'):
    cmd = ['basenji_train_folds.py',
          '-o', 'tests/data/train',
          '--setup',
          'tests/data/params.json',
          'tests/data/tiny/hg38',
          'tests/data/tiny/mm10']
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

  # clean previous training directories
  out_dirs = ['tests/data/train1', 'tests/data/train2']
  for out_dir in out_dirs:
    if os.path.exists(out_dir):
      shutil.rmtree(out_dir)


def test_train(clean_data):
  cmd = ['scripts/hound_train.py', 
         '-o', 'tests/data/train1',
         'tests/data/params.json',
         'tests/data/train/f0c0/data0']
  print(' '.join(cmd))
  subprocess.run(cmd, check=True)

  # TODO: Check evaluation metrics numerical stability
  assert(os.path.exists('tests/data/train1/model_best.h5'))


def test_train2(clean_data):
  cmd = ['scripts/hound_train.py', 
         '-o', 'tests/data/train2',
         'tests/data/params.json',
         'tests/data/train/f0c0/data0',
         'tests/data/train/f0c0/data1']
  subprocess.run(cmd, check=True)

  # TODO: Check evaluation metrics numerical stability
  assert(os.path.exists('tests/data/train2/model0_best.h5'))
  assert(os.path.exists('tests/data/train2/model1_best.h5'))