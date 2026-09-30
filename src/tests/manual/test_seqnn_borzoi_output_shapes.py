import torch

from baskerville import seqnn

# --- DEFINE MODEL PARAMS ---
model_params = {
    "seq_length": 524288,
    "trunk": [
        {
            "name": "BorzoiTrunk",
            "use_flash_attn": False,
            "replicate_index": 0,
            "pool_size": 32,
            "crop_size": 5120,
        }
    ],
    "head_human": {
        "name": "BorzoiHead",
        "use_flash_attn": False,
        "replicate_index": 0,
        "use_human": True,
        "in_channels": 1920,
        "out_channels": 7611,
    },
    "head_mouse": {
        "name": "BorzoiHead",
        "use_flash_attn": False,
        "replicate_index": 0,
        "use_human": False,
        "in_channels": 1920,
        "out_channels": 2608,
    },
}

# --- GET MODEL ---
seqnn_model = seqnn.SeqNN(model_params)
model = seqnn_model.model
model.to("cuda")

# --- TEST SHAPES ---
x = torch.randn((1, 4, 524288)).to("cuda").to(torch.bfloat16)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yh = model(x, hi=0)
        assert [1, 7611, 6144] == list(yh.shape), "Human output shape incorrect"
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yh = model(x, hi=1)
        assert [1, 2608, 6144] == list(yh.shape), "Mouse output shape incorrect"

print("Passed output shape tests.")
