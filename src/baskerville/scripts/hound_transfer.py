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
        help="transfer method. [full, linear, adapterHoulsby, lora, lora_full, ia3]",
    )
    parser.add_argument(
        "--latent",
        type=int,
        default=16,
        help="adapter latent size.",
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
        seqnn_model = seqnn.SeqNN(params_model)
    
        # restore
        if args.restore:
            seqnn_model.restore(args.restore, trunk=args.trunk)

        # transfer learning strategies
        if args.transfer_mode=='full':
            seqnn_model.model.trainable=True
        
        elif args.transfer_mode=='batch_norm':
            seqnn_model.model_trunk.trainable=False
            for l in seqnn_model.model.layers:
                if l.name.startswith("batch_normalization"):
                    l.trainable=True
            seqnn_model.model.summary()
        
        elif args.transfer_mode=='linear':
            seqnn_model.model_trunk.trainable=False
            seqnn_model.model.summary()
        
        elif args.transfer_mode=='adapterHoulsby':
            seqnn_model.model_trunk.trainable=False
            strand_pair = strand_pairs[0]
            adapter_model = make_adapter_model(seqnn_model.model, strand_pair, args.latent)
            seqnn_model.model = adapter_model
            seqnn_model.models[0] = seqnn_model.model
            seqnn_model.model_trunk = None
            seqnn_model.model.summary()
        
        elif args.transfer_mode=='lora':
            add_lora(seqnn_model.model, rank=args.latent, mode='default')
            seqnn_model.model.summary()
        
        elif args.transfer_mode=='lora_full':
            add_lora(seqnn_model.model, rank=args.latent, mode='full')
            seqnn_model.model.summary()
            
        elif args.transfer_mode=='ia3':
            add_ia3(seqnn_model.model)
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

def make_adapter_model(input_model, strand_pair, latent_size=16):
    # take seqnn_model as input
    # output a new seqnn_model object
    # only the adapter, and layer_norm are trainable
    
    model = tf.keras.Model(inputs=input_model.input, 
                           outputs=input_model.layers[-2].output) # remove the switch_reverse layer
    
    # save current graph
    layer_parent_dict_old = {} # the parent layers of each layer in the old graph 
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in layer_parent_dict_old:
                layer_parent_dict_old.update({layer_name: [layer.name]})
            else:
                if layer.name not in layer_parent_dict_old[layer_name]:
                    layer_parent_dict_old[layer_name].append(layer.name)
    
    layer_output_dict_new = {} # the output tensor of each layer in the new graph
    layer_output_dict_new.update({model.layers[0].name: model.input})
    
    # remove switch_reverse
    to_fix = [i for i in layer_parent_dict_old if re.match('switch_reverse', i)]
    for i in to_fix:
        del layer_parent_dict_old[i]
    
    # Iterate over all layers after the input
    model_outputs = []
    reverse_bool = None
    
    for layer in model.layers[1:]:
    
        # parent layers
        parent_layers = layer_parent_dict_old[layer.name]
    
        # layer inputs
        layer_input = [layer_output_dict_new[parent] for parent in parent_layers]
        if len(layer_input) == 1: layer_input = layer_input[0]
    
        if re.match('stochastic_reverse_complement', layer.name):
            x, reverse_bool  = layer(layer_input)
        
        # insert adapter:
        elif re.match('add', layer.name):
            if any([re.match('dropout', i) for i in parent_layers]):
                print('adapter added before:%s'%layer.name)
                x = layers.AdapterHoulsby(latent_size=latent_size)(layer_input[1])
                x = layer([layer_input[0], x])
            else:
                x = layer(layer_input)
        
        else:
            x = layer(layer_input)
    
        # save the output tensor of every layer
        layer_output_dict_new.update({layer.name: x})
    
    final = layers.SwitchReverse(strand_pair)([layer_output_dict_new[model.layers[-1].name], reverse_bool])
    model_adapter = tf.keras.Model(inputs=model.inputs, outputs=final)
    
    # set layer_norm layers to trainable
    for l in model_adapter.layers:
        if re.match('layer_normalization', l.name): l.trainable = True

    return model_adapter

def add_lora(input_model, rank=8, alpha=16, mode='default'):
    ######################
    # inject lora layers #
    ######################
    # take seqnn.model as input
    # replace _q_layer, _v_layer in multihead_attention
    # optionally replace _k_layer, _embedding_layer
    if mode not in ['default','full']:
        raise ValueError("mode must be default or full")
    
    for layer in input_model.layers:
        if re.match('multihead_attention', layer.name):
            # default loRA
            layer._q_layer = layers.Lora(layer._q_layer, rank=rank, alpha=alpha, trainable=True)
            layer._v_layer = layers.Lora(layer._v_layer, rank=rank, alpha=alpha, trainable=True)
            # full loRA
            if mode=='full':
                layer._k_layer = layers.Lora(layer._k_layer, rank=rank, alpha=alpha, trainable=True)
                layer._embedding_layer = layers.Lora(layer._embedding_layer, rank=rank, alpha=alpha, trainable=True)
    
    input_model(input_model.input) # initialize new variables 

    #################
    # freeze params #
    #################
    # freeze all params but lora
    for layer in input_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())
        if len(lst_of_sublayers) == 1: 
            if layer.name in ["lora_a", "lora_b"]:
                layer.trainable = True
            else:
                layer.trainable = False

    ### bias terms need to be frozen separately 
    for layer in input_model.layers:
        if re.match('multihead_attention', layer.name):
            layer._r_w_bias = tf.Variable(layer._r_w_bias, trainable=False, name=layer._r_w_bias.name)
            layer._r_r_bias = tf.Variable(layer._r_r_bias, trainable=False, name=layer._r_r_bias.name)

    # set final head to be trainable
    input_model.layers[-2].trainable=True


def add_ia3(input_model):
    #####################
    # inject ia3 layers #
    #####################
    # take seqnn.model as input
    # replace _k_layer, _v_layer, _embedding_layer in multihead_attention
    for layer in input_model.layers:
        if re.match('multihead_attention', layer.name):
            layer._k_layer = layers.IA3(layer._k_layer, trainable=True)
            layer._v_layer = layers.IA3(layer._v_layer, trainable=True)
            layer._embedding_layer = layers.IA3(layer._embedding_layer, trainable=True)
    input_model(input_model.input) # instantiate model to initialize new variables

    #################
    # freeze params #
    #################
    # set ia3 to trainable
    for layer in input_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())
        if len(lst_of_sublayers) == 1: 
            if layer.name =='ia3':
                layer.trainable = True
            else:
                layer.trainable = False
            
    ### bias terms need to be frozen separately 
    for layer in input_model.layers:
        if re.match('multihead_attention', layer.name):
            layer._r_w_bias = tf.Variable(layer._r_w_bias, trainable=False, name=layer._r_w_bias.name)
            layer._r_r_bias = tf.Variable(layer._r_r_bias, trainable=False, name=layer._r_r_bias.name)

    # set final head to be trainable
    input_model.layers[-2].trainable=True

def param_count(model):
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    print('total params:%d' %(trainable + non_trainable))
    print('trainable params:%d' %trainable)
    print('non-trainable params:%d' %non_trainable)

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
