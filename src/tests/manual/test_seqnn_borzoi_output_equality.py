import torch
import numpy as np
import os
import shutil
import tqdm

from baskerville import seqnn
from baskerville import dataset
import borzoi_pytorch

"""
Note:
    Assumes access to GPU, and BORZOI_DATA_DIR pointing at the Borzoi training
    data (hg38/ and mm10/ hound_data directories).

Example correlations for the first 20 examples from the original Borzoi dataset:
    Human: Min corr: 0.9999968625404633; Mean corr: 0.9999975106827336
    Mouse: Min corr: 0.8977377320622217; Mean corr: 0.97564137298448
"""


# --- INIT ---
num_sequences = 20
tmp_dir = "~/tmp/flashzoi_test"

# --- MAKE TMP DIR ---
tmp_dir = os.path.expanduser(tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)

# --- INIT THE DATASETS ---
data_dir = os.environ["BORZOI_DATA_DIR"]
human_dataset = dataset.SeqDataset(f"{data_dir}/hg38", split_label="fold0", mode="eval")
mouse_dataset = dataset.SeqDataset(f"{data_dir}/mm10", split_label="fold0", mode="eval")
num_sequences = min(num_sequences, len(human_dataset), len(mouse_dataset))


# --- DEFINE FN's FOR MODEL PREDICTIONS ---
def get_model_preds(
    model,
    dataset,
    num_sequences,
    tmp_dir,
    data_label,
    model_label,
    is_flashzoi_mouse=False,
):

    # ** init model **
    model.eval()
    model.to("cuda")

    # ** get predictions **
    for i in range(0, num_sequences):
        x = dataset[i].sequence
        x = x.to("cuda").to(torch.bfloat16)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # get pred
                if (
                    is_flashzoi_mouse
                ):  # required for getting mouse prediction from Flashzoi...
                    y_pred = (
                        model(x.unsqueeze(0), is_human=False).squeeze(0).cpu().numpy()
                    )
                else:
                    y_pred = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()

                # write out
                with open(
                    f"{tmp_dir}/{data_label}_{model_label}_pred_{i}.npy", "wb"
                ) as f:
                    np.save(f, y_pred)


def get_seqnn_preds(dataset, model_params, num_sequences, tmp_dir, data_label):

    # ** init model **
    seqnn_model = seqnn.SeqNN(model_params)
    model = seqnn_model.model

    # ** get preds **
    get_model_preds(model, dataset, num_sequences, tmp_dir, data_label, "seqnn")


def get_borzoi_preds(dataset, use_human, num_sequences, tmp_dir, data_label):

    # ** init model **
    if use_human:
        model_string = f"johahi/borzoi-replicate-0"
    else:
        model_string = f"johahi/borzoi-replicate-0-mouse"
    model = borzoi_pytorch.Borzoi.from_pretrained(model_string)

    # ** get preds **
    get_model_preds(
        model,
        dataset,
        num_sequences,
        tmp_dir,
        data_label,
        "borzoi",
        is_flashzoi_mouse=not use_human,
    )


# --- HUMAN: GET SEQNN PREDS ---

# ** define model params **
model_params = {
    "seq_length": 524288,
    "trunk": [
        {
            "name": "BorzoiTrunk",
            "use_flash_attn": False,
            "replicate_index": 0,
        },
    ],
    "head": {
        "name": "BorzoiHead",
        "use_flash_attn": False,
        "replicate_index": 0,
        "use_human": True,
        "in_channels": 1920,
        "out_channels": 7611,  # 7611 for human, 2608 for mouse
    },
}

# ** get preds **
get_seqnn_preds(human_dataset, model_params, num_sequences, tmp_dir, "human")

# --- MOUSE: GET SEQNN PREDS ---


# ** define model params **
model_params = {
    "seq_length": 524288,
    "trunk": [
        {
            "name": "BorzoiTrunk",
            "use_flash_attn": False,
            "replicate_index": 0,
        },
    ],
    "head": {
        "name": "BorzoiHead",
        "use_flash_attn": False,
        "replicate_index": 0,
        "use_human": False,
        "in_channels": 1920,
        "out_channels": 2608,  # 7611 for human, 2608 for mouse
    },
}

# ** get preds **
get_seqnn_preds(mouse_dataset, model_params, num_sequences, tmp_dir, "mouse")

# --- HUMAN: GET BORZOI PREDS ---

get_borzoi_preds(
    human_dataset,
    use_human=True,
    num_sequences=num_sequences,
    tmp_dir=tmp_dir,
    data_label="human",
)


# --- MOUSE: GET BORZOI PREDS ---

get_borzoi_preds(
    mouse_dataset,
    use_human=False,
    num_sequences=num_sequences,
    tmp_dir=tmp_dir,
    data_label="mouse",
)

# --- ASSERT CLOSENESS ---

# ** human **
corrs = []
for i in tqdm.tqdm(range(0, num_sequences)):
    borzoi_path = f"{tmp_dir}/human_borzoi_pred_{i}.npy"
    seqnn_path = f"{tmp_dir}/human_seqnn_pred_{i}.npy"
    borzoi_pred = np.load(borzoi_path)
    seqnn_pred = np.load(seqnn_path)
    corrs.append(np.corrcoef(seqnn_pred.flatten(), borzoi_pred.flatten())[0, 1])
print(f"Min corr: {np.min(corrs)}; Mean corr: {np.mean(corrs)}")
assert np.min(corrs) >= 0.85

# ** mouse **
corrs = []
for i in tqdm.tqdm(range(0, num_sequences)):
    borzoi_path = f"{tmp_dir}/mouse_borzoi_pred_{i}.npy"
    seqnn_path = f"{tmp_dir}/mouse_seqnn_pred_{i}.npy"
    borzoi_pred = np.load(borzoi_path)
    seqnn_pred = np.load(seqnn_path)
    corrs.append(np.corrcoef(seqnn_pred.flatten(), borzoi_pred.flatten())[0, 1])
print(f"Min corr: {np.min(corrs)}; Mean corr: {np.mean(corrs)}")
assert np.min(corrs) >= 0.85

# --- CLEAN UP ---

shutil.rmtree(tmp_dir)
