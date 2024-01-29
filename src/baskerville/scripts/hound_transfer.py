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
from baskerville import transfer_helper

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
        help="attention layer module [adapterHoulsby, lora, lora_full, ia3]",
    )
    parser.add_argument(
        "--att_latent",
        type=int,
        default=16,
        help="attention adapter latent size.",
    )    
    parser.add_argument(
        "--conv_adapter",
        default=None,
        type=str,
        help="conv layer module [conv, bn, conv_bn, squez_excit]",
    )

    parser.add_argument(
        "--se_ratio",
        type=int,
        default=16,
        help="se bottleneck ratio.",
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
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
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
                    if args.conv_adapter not in ['se', 'se_bn', 'se_all','se_all_bn']:
                        # when att_adapter=='Houlsby' and conv_adapter=='se', do nothing.
                        # see conv_adapter section.
                        seqnn_model.model = transfer_helper.add_houlsby(seqnn_model.model, 
                                                                        strand_pairs[0], 
                                                                        latent_size=args.att_latent)
                elif args.att_adapter=='lora':
                    transfer_helper.add_lora(seqnn_model.model, 
                                             rank=args.att_latent, 
                                             mode='default')
                    
                elif args.att_adapter=='lora_full':
                    transfer_helper.add_lora(seqnn_model.model, 
                                             rank=args.att_latent, 
                                             mode='full')
                
                elif args.att_adapter=='ia3':
                    transfer_helper.add_ia3(seqnn_model.model)

            # conv adapter
            # assume seqnn_model is appropriately frozen
            if args.conv_adapter is not None:
                if args.conv_adapter=='conv':
                    params_added = 0
                    for l in seqnn_model.model.layers:
                        if l.name.startswith(("conv1d","separable_conv1d")):
                            l.trainable=True
                            params_added += transfer_helper.param_count(l, type='trainable')
                    print('params added/unfrozen by conv: %d'%params_added)
                
                elif args.conv_adapter=='conv_bn':
                    params_added = 0
                    for l in seqnn_model.model.layers:
                        if l.name.startswith(("conv1d","separable_conv1d","batch_normalization")):
                            l.trainable=True
                            params_added += transfer_helper.param_count(l, type='trainable')
                    print('params added/unfrozen by conv_bn: %d'%params_added)

                elif args.conv_adapter=='bn':
                    params_added = 0
                    for l in seqnn_model.model.layers:
                        if l.name.startswith("batch_normalization"):
                            l.trainable=True
                            params_added += transfer_helper.param_count(l, type='trainable')
                    print('params added/unfrozen by bn: %d'%params_added)

                ##################
                # squeeze-excite #
                ##################
                elif args.conv_adapter in ['se','se_bn','se_all','se_all_bn']:
                    if args.att_adapter=='adapterHoulsby':
                        if args.conv_adapter=='se':
                            seqnn_model.model = transfer_helper.add_houlsby_se(seqnn_model.model, 
                                                                               strand_pair=strand_pairs[0], 
                                                                               houlsby_latent=args.att_latent,
                                                                               bottleneck_ratio=args.se_ratio, 
                                                                               insert_mode='pre_att',
                                                                               unfreeze_bn=False)
                        elif args.conv_adapter=='se_bn':
                            seqnn_model.model = transfer_helper.add_houlsby_se(seqnn_model.model, 
                                                                               strand_pair=strand_pairs[0], 
                                                                               houlsby_latent=args.att_latent,
                                                                               bottleneck_ratio=args.se_ratio, 
                                                                               insert_mode='pre_att',
                                                                               unfreeze_bn=True)
                        elif args.conv_adapter=='se_all':
                            seqnn_model.model = transfer_helper.add_houlsby_se(seqnn_model.model, 
                                                                               strand_pair=strand_pairs[0], 
                                                                               houlsby_latent=args.att_latent,
                                                                               bottleneck_ratio=args.se_ratio, 
                                                                               insert_mode='all',
                                                                               unfreeze_bn=False)
                        elif args.conv_adapter=='se_all_bn':
                            seqnn_model.model = transfer_helper.add_houlsby_se(seqnn_model.model, 
                                                                               strand_pair=strand_pairs[0], 
                                                                               houlsby_latent=args.att_latent,
                                                                               bottleneck_ratio=args.se_ratio, 
                                                                               insert_mode='all',
                                                                               unfreeze_bn=True)
                    else:
                        if args.conv_adapter=='se':
                            seqnn_model.model = transfer_helper.add_se(seqnn_model.model, 
                                                                       strand_pair=strand_pairs[0], 
                                                                       houlsby_latent=args.att_latent,
                                                                       bottleneck_ratio=args.se_ratio, 
                                                                       insert_mode='pre_att',
                                                                       unfreeze_bn=False)
                        elif args.conv_adapter=='se_bn':
                            seqnn_model.model = transfer_helper.add_se(seqnn_model.model, 
                                                                       strand_pair=strand_pairs[0], 
                                                                       houlsby_latent=args.att_latent,
                                                                       bottleneck_ratio=args.se_ratio, 
                                                                       insert_mode='pre_att',
                                                                       unfreeze_bn=True)
                        elif args.conv_adapter=='se_all':
                            seqnn_model.model = transfer_helper.add_se(seqnn_model.model, 
                                                                       strand_pair=strand_pairs[0], 
                                                                       houlsby_latent=args.att_latent,
                                                                       bottleneck_ratio=args.se_ratio, 
                                                                       insert_mode='all',
                                                                       unfreeze_bn=False)
                        elif args.conv_adapter=='se_all_bn':
                            seqnn_model.model = transfer_helper.add_se(seqnn_model.model, 
                                                                       strand_pair=strand_pairs[0], 
                                                                       houlsby_latent=args.att_latent,
                                                                       bottleneck_ratio=args.se_ratio, 
                                                                       insert_mode='pre_att',
                                                                       unfreeze_bn=True)
                    
        #################
        # final summary #
        #################
        seqnn_model.model.summary()
                
        # initialize trainer
        seqnn_trainer = trainer.Trainer(
            params_train, train_data, eval_data, args.out_dir
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
            
            # overwrite json file when needed
            # for: adapterHoulsby and squeeze-excite
            transfer_helper.modify_json(input_json=args.params_file,
                                        output_json='%s/params.json'%args.out_dir,
                                        adapter=args.att_adapter,
                                        latent=args.att_latent,
                                        conv=args.conv_adapter, 
                                        se_ratio=args.se_ratio)

            # merge weights when needed
            # for: lora and ia3
            # save weight to: model_best.mergeW.h5
            if args.att_adapter=='lora':
                seqnn_model.model.load_weights('%s/model_best.h5'%args.out_dir)
                transfer_helper.merge_lora(seqnn_model.model, mode='default')
                seqnn_model.save('%s/model_best.mergeW.h5'%args.out_dir)
                transfer_helper.var_reorder('%s/model_best.mergeW.h5'%args.out_dir)
            
            if args.att_adapter=='lora_full':
                seqnn_model.model.load_weights('%s/model_best.h5'%args.out_dir)
                transfer_helper.merge_lora(seqnn_model.model, mode='full')
                seqnn_model.save('%s/model_best.mergeW.h5'%args.out_dir)
                transfer_helper.var_reorder('%s/model_best.mergeW.h5'%args.out_dir)
    
            # merge ia3 weights to original, save weight to: model_best_mergeweight.h5
            if args.att_adapter=='ia3':
                seqnn_model.model.load_weights('%s/model_best.h5'%args.out_dir)
                transfer_helper.merge_ia3(seqnn_model.model)
                seqnn_model.save('%s/model_best.mergeW.h5'%args.out_dir)
                transfer_helper.var_reorder('%s/model_best.mergeW.h5'%args.out_dir)

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
