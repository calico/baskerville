pretrain_model='/home/yuanh/analysis/Borzoi_transfer/exp_10_10_23/westminster_no_gtex_trunk/trained_trunks/f0c0.h5'
data='/home/yuanh/analysis/Borzoi_transfer/exp_10_10_23/hayflick/houlsby/latent_8/train/f0c0/data0'

# test each script
# modify hound_transfer.py to exit after compile
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_full.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_linear.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_houlsby.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_lora.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_ia3.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_locon4.json ${data}
hound_transfer.py -o test --restore ${pretrain_model} --trunk ../data/transfer/transfer_json/params_se4.json ${data}
