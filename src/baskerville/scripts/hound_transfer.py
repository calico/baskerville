#!/usr/bin/env python
# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import argparse
import json
import os
import shutil
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer
from baskerville import layers
from baskerville.helpers import transfer_helper

"""
hound_transfer.py

Modified from hound_train.py.
Additional argument to allow for transfer learning from existing Hound model.
"""

def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-k",
        "--keras_fit",
        action="store_true",
        default=False,
        help="Train with Keras fit method [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Train with mixed precision [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %(default)s]",
    )    
    parser.add_argument(
        "--restore",
        default=None,
        help="Restore model and continue training [Default: %(default)s]",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        default=False,
        help="Restore only model trunk [Default: %(default)s]",
    )
    parser.add_argument(
        "--transfer_mode",
        default="full",
        help="transfer method. [full, linear, adapter]",
    )
    parser.add_argument(
        "--att_adapter",
        default=None,
        type=str,
        help="attention layer module [adapterHoulsby, lora, lora_full, ia3, locon]",
    )
    parser.add_argument(
        "--att_latent",
        type=int,
        default=8,
        help="attention adapter latent size.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="lora alpha.",
    )
    parser.add_argument(
        "--conv_select",
        default=None,
        type=int,
        help="# of conv layers to insert locon/se.",
    )
    parser.add_argument(
        "--conv_rank",
        type=int,
        default=4,
        help="locon/se rank.",
    )    
    parser.add_argument(
        "--locon_alpha",
        type=int,
        default=1,
        help="locon_alpha.",
    )    
    parser.add_argument(
        "--tfr_train",
        default=None,
        help="Training TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_eval",
        default=None,
        help="Evaluation TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    args = parser.parse_args()

    if args.keras_fit and len(args.data_dirs) > 1:
        print("Cannot use keras fit method with multi-genome training.")
        exit()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != "%s/params.json" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.json" % args.out_dir)

    if args.transfer_mode not in ['full','linear','sparse']:
        raise ValueError("transfer mode must be one of full, linear, sparse")
      
    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # read datasets
    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        # set strand pairs
        targets_df = pd.read_csv("%s/targets.txt" % data_dir, sep="\t", index_col=0)
        if "strand_pair" in targets_df.columns:
            strand_pairs.append(np.array(targets_df.strand_pair))

        # load train data
        train_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="train",
                batch_size=params_train["batch_size"],
                shuffle_buffer=params_train.get("shuffle_buffer", 128),
                mode="train",
                tfr_pattern=args.tfr_train,
            )
        )

        # load eval data
        eval_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                tfr_pattern=args.tfr_eval,
            )
        )
    
    params_model["strand_pair"] = strand_pairs

    if args.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
    
    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # one GPU

        # initialize model
        params_model['verbose']=False
        seqnn_model = seqnn.SeqNN(params_model)
    
        # restore
        if args.restore:
            seqnn_model.restore(args.restore, trunk=args.trunk)

        # head params
        print('params in new head: %d' %transfer_helper.param_count(seqnn_model.model.layers[-2]))

        ####################
        # transfer options #
        ####################
        if args.transfer_mode=='full':
            seqnn_model.model.trainable=True
        
        elif args.transfer_mode=='linear':
            seqnn_model.model_trunk.trainable=False

        ############
        # adapters #
        ############
        elif args.transfer_mode=='sparse':

            # attention adapter
            if args.att_adapter is not None:
                if args.att_adapter=='adapterHoulsby':
                    seqnn_model.model = transfer_helper.add_houlsby(seqnn_model.model, 
                                                                    strand_pairs[0], 
                                                                    latent_size=args.att_latent)
                elif args.att_adapter=='lora':
                    transfer_helper.add_lora(seqnn_model.model, 
                                             rank=args.att_latent, 
                                             alpha=args.lora_alpha,
                                             mode='default')
                    
                elif args.att_adapter=='lora_full':
                    transfer_helper.add_lora(seqnn_model.model, 
                                             rank=args.att_latent, 
                                             alpha=args.lora_alpha,
                                             mode='full')
                
                elif args.att_adapter=='ia3':
                    seqnn_model.model = transfer_helper.add_ia3(seqnn_model.model, 
                                                                strand_pairs[0])

                elif args.att_adapter=='locon': # lora on conv+att
                    seqnn_model.model = transfer_helper.add_locon(seqnn_model.model, 
                                                                  strand_pairs[0],
                                                                  conv_select=args.conv_select, 
                                                                  rank=args.conv_rank, 
                                                                  alpha=args.locon_alpha)

                elif args.att_adapter=='lora_conv': # lora on att, unfreeze_conv
                    transfer_helper.add_lora_conv(seqnn_model.model, conv_select=args.conv_select)

                elif args.att_adapter=='houlsby_se': # adapter on conv+att
                    seqnn_model.model = transfer_helper.add_houlsby_se(seqnn_model.model, 
                                                                       strand_pair=strand_pairs[0], 
                                                                       conv_select=args.conv_select,
                                                                       se_rank=args.conv_rank)
                    
        #################
        # final summary #
        #################
        seqnn_model.model.summary()

        if args.mixed_precision:
            # add additional activation to cast float16 output to float32
            seqnn_model.append_activation()
            # run with loss scaling
            seqnn_trainer = trainer.Trainer(
                params_train, train_data, eval_data, args.out_dir, args.log_dir, loss_scale=True
            )
        else:
            seqnn_trainer = trainer.Trainer(
                params_train, train_data, eval_data, args.out_dir, args.log_dir
            )
            
        # compile model
        seqnn_trainer.compile(seqnn_model)

        # train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model)
            else:
                seqnn_trainer.fit2(seqnn_model)

        #############################
        # post-training adjustments # 
        #############################
        if args.transfer_mode=='sparse':
            
            # for: adapterHoulsby and houlsby_se, overwrite json file
            if args.att_adapter=='adapterHoulsby':            
                transfer_helper.modify_json(input_json=args.params_file,
                                            output_json='%s/params.json'%args.out_dir,
                                            adapter=args.att_adapter,
                                            latent=args.att_latent)

            if args.att_adapter=='houlsby_se':
                transfer_helper.modify_json(input_json=args.params_file,
                                            output_json='%s/params.json'%args.out_dir,
                                            adapter=args.att_adapter,
                                            conv_select=args.conv_select,
                                            se_rank=args.conv_rank
                                            )
            
            # for lora, ia3, locon, save weight to: model_best.mergeW.h5
            if args.att_adapter in ['lora', 'lora_full', 'lora_conv']:
                seqnn_model.model.load_weights('%s/model_best.h5'%args.out_dir)
                transfer_helper.merge_lora(seqnn_model.model)
                seqnn_model.save('%s/model_best.mergeW.h5'%args.out_dir)
                transfer_helper.var_reorder('%s/model_best.mergeW.h5'%args.out_dir)
            
            if args.att_adapter=='ia3':
                # ia3 model
                ia3_model = seqnn_model.model
                ia3_model.load_weights('%s/model_best.h5'%args.out_dir)                
                # original model
                seqnn_model2 = seqnn.SeqNN(params_model)
                seqnn_model2.restore(args.restore, trunk=args.trunk)
                original_model = seqnn_model2.model
                # merge weights into original model
                transfer_helper.merge_ia3(original_model, ia3_model)
                original_model.save('%s/model_best.mergeW.h5'%args.out_dir)

            if args.att_adapter=='locon':
                # locon model
                locon_model = seqnn_model.model
                locon_model.load_weights('%s/model_best.h5'%args.out_dir)                
                # original model
                seqnn_model2 = seqnn.SeqNN(params_model)
                seqnn_model2.restore(args.restore, trunk=args.trunk)
                original_model = seqnn_model2.model
                # merge weights into original model
                transfer_helper.merge_locon(original_model, locon_model)
                original_model.save('%s/model_best.mergeW.h5'%args.out_dir)

    else:
        ########################################
        # multi GPU

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():

            if not args.keras_fit:
                # distribute data
                for di in range(len(args.data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)

            # initialize model
            seqnn_model = seqnn.SeqNN(params_model)

            # restore
            if args.restore:
                seqnn_model.restore(args.restore, args.trunk)

            # initialize trainer
            seqnn_trainer = trainer.Trainer(
                params_train,
                train_data,
                eval_data,
                args.out_dir,
                strategy,
                params_train["num_gpu"],
                args.keras_fit,
            )

            # compile model
            seqnn_trainer.compile(seqnn_model)

        # train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model)
            else:
                seqnn_trainer.fit2(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
