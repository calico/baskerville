pretrain_model='/home/yuanh/analysis/Borzoi_transfer/westminster_no_gtex/train/train/f0c0/train/model0_best.h5'
data='/home/yuanh/analysis/Borzoi_transfer/exp_10_10_23/hayflick/houlsby/latent_8/train/f0c0/data0'

# test the trainable params match expectation
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_full.json ${data} #43,744,164
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_linear.json ${data} #52,292
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_houlsby.json ${data} #285,892
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_lora.json ${data} #216,132
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_ia3.json ${data} #72,772
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_locon4.json ${data} #270,404
hound_transfer.py -o test --restore ${pretrain_model} --skip_train ../data/transfer/json/borzoilite_se4.json ${data} #366,788
