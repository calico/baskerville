import json
import numpy as np
import pandas as pd
from baskerville import seqnn

model_file = "/home/yuanh/analysis/Borzoi_transfer/exp_10_10_23/hayflick/houlsby_se/se2/train/f0c0/train/model_best.h5"
targets_file = "/home/yuanh/programs/source/python_packages/baskerville/tests/data/transfer/targets.txt"
params_file = "model_se2.json"

# model params
with open(params_file) as params_open:
    params = json.load(params_open)
params_model = params["model"]
params_model["verbose"] = False

# set strand pairs
targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
if "strand_pair" in targets_df.columns:
    params_model["strand_pair"] = [np.array(targets_df.strand_pair)]
strand_pair = np.array(targets_df.strand_pair)

seqnn_model = seqnn.SeqNN(params_model)
seqnn_model.restore(model_file)

print("load model success!")