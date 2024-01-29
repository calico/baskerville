import argparse
import json
import os
import shutil
import re
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer
from baskerville import layers

def param_count(layer, type='all'):
    if type not in ['all','trainable','non_trainable']:
        raise ValueError("TYPE must be one of all, trainable, non_trainable")
    output = 0
    if type=='all':
        output = int(sum(tf.keras.backend.count_params(w) for w in layer.weights))
    elif type=='trainable':
        output = int(sum(tf.keras.backend.count_params(w) for w in layer.trainable_weights))
    else:
        output = int(sum(tf.keras.backend.count_params(w) for w in layer.non_trainable_weights))
    return output

def param_summary(model):
    trainable = param_count(model, type='trainable')
    non_trainable = param_count(model, type='non_trainable')
    print('total params:%d' %(trainable + non_trainable))
    print('trainable params:%d' %trainable)
    print('non-trainable params:%d' %non_trainable)

######################
# add houlsby layers #
######################
def add_houlsby(input_model, strand_pair, latent_size=16):
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

    # set trainable
    for l in model_adapter.layers[:-2]: # trunk
        if re.match('layer_normalization|adapter_houlsby', l.name): 
            l.trainable = True
        else:
            l.trainable = False

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_adapter.layers:
        if l.name.startswith("adapter_houlsby"): 
            params_added += param_count(l)
        elif l.name.startswith("layer_normalization"): 
            params_added += param_count(l, type='trainable')
    print('params added/unfrozen by adapter_houlsby: %d'%params_added)

    return model_adapter

###################
# add lora layers #
###################
def add_lora(input_model, rank=8, alpha=16, mode='default'):
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

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in input_model.layers:
        if re.match('multihead_attention', l.name):
            params_added += param_count(l._q_layer.down_layer)
            params_added += param_count(l._q_layer.up_layer)
            params_added += param_count(l._v_layer.down_layer)
            params_added += param_count(l._v_layer.up_layer)
            if mode=='full':
                params_added += param_count(l._k_layer.down_layer)
                params_added += param_count(l._k_layer.up_layer)
                params_added += param_count(l._embedding_layer.down_layer)
                params_added += param_count(l._embedding_layer.up_layer)
    
    print('params added/unfrozen by lora: %d'%params_added)

##################
# add ia3 layers #
##################
def add_ia3(input_model):
    # take seqnn.model as input
    # replace _k_layer, _v_layer, _embedding_layer in multihead_attention
    for layer in input_model.layers:
        if re.match('multihead_attention', layer.name):
            layer._k_layer = layers.IA3(layer._k_layer, trainable=True)
            layer._v_layer = layers.IA3(layer._v_layer, trainable=True)
            layer._embedding_layer = layers.IA3(layer._embedding_layer, trainable=True)
    input_model(input_model.input) # instantiate model to initialize new variables

    # freeze params:
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

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in input_model.layers:
        if re.match('multihead_attention', l.name):
            params_added += param_count(l._k_layer._ia3_layer)
            params_added += param_count(l._v_layer._ia3_layer)
            params_added += param_count(l._embedding_layer._ia3_layer)
    
    print('params added/unfrozen by ia3: %d'%params_added)

######################
# add squeeze excite #
######################
def add_se(input_model, strand_pair, bottleneck_ratio=8, insert_mode='pre_att', unfreeze_bn=False):
    # add squeeze-excitation blocks after conv
    # input_model should be properly frozen
    # pre_att: add se_block to pre-attention conv1d
    # all: add se_block to pre-attention conv1d and post-attention separable_conv1d
    
    if insert_mode not in ['pre_att','all']:
        raise ValueError("insert_mode must be pre_att or all")

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

        if layer.name.startswith("stochastic_reverse_complement"):
            x, reverse_bool  = layer(layer_input)
        
        # insert squeeze-excite layer:
        elif layer.name.startswith("conv1d"):
            se_layer = layers.SqueezeExcite(
                activation=None, # no activation before squeezing
                additive=False, # use sigmoid multiplicative scaling
                bottleneck_ratio=bottleneck_ratio, # bottleneck ratio
                use_bias=False, # ignore bias
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3), # near-zero weight initialization
                scale_fun='tanh'
            )
            x = layer(layer_input)
            x = x + se_layer(x)

        elif layer.name.startswith("separable_conv1d"):
            if insert_mode=='all':
                se_layer = layers.SqueezeExcite(
                    activation=None, # no activation before squeezing
                    additive=False, # use sigmoid multiplicative scaling
                    bottleneck_ratio=bottleneck_ratio, # bottleneck ratio
                    use_bias=False, # ignore bias
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3), # near-zero weight initialization
                    scale_fun='tanh'
                )
                x = layer(layer_input)
                x = x + se_layer(x)
            else:
                x = layer(layer_input)
        
        else:
            x = layer(layer_input)
    
        # save the output tensor of every layer
        layer_output_dict_new.update({layer.name: x})
    
    final = layers.SwitchReverse(strand_pair)([layer_output_dict_new[model.layers[-1].name], reverse_bool])
    model_final = tf.keras.Model(inputs=model.inputs, outputs=final)

    # unfreeze layers
    for l in model_final.layers: # set trunk
        if l.name.startswith("squeeze_excite"): l.trainable = True

    if unfreeze_bn:
        for l in model_final.layers:
            if l.name.startswith("batch_normalization"): l.trainable=True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_final.layers:
        if l.name.startswith("squeeze_excite"): 
            params_added += param_count(l)
        elif l.name.startswith("batch_normalization"): 
            if unfreeze_bn: params_added += param_count(l, type='trainable')
    print('params added/unfrozen by se_block: %d'%params_added)
    
    return model_final


def add_houlsby_se(input_model, strand_pair, houlsby_latent=8, bottleneck_ratio=8, insert_mode='pre_att', unfreeze_bn=False):
    # add squeeze-excitation blocks after conv
    # input_model should be properly frozen
    # pre_att: add se_block to pre-attention conv1d
    # all: add se_block to pre-attention conv1d and post-attention separable_conv1d
    
    if insert_mode not in ['pre_att','all']:
        raise ValueError("insert_mode must be pre_att or all")

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

        if layer.name.startswith("stochastic_reverse_complement"):
            x, reverse_bool  = layer(layer_input)
        
        # insert houlsby:
        elif re.match('add', layer.name):
            if any([re.match('dropout', i) for i in parent_layers]):
                print('adapter added before:%s'%layer.name)
                x = layers.AdapterHoulsby(latent_size=houlsby_latent)(layer_input[1])
                x = layer([layer_input[0], x])
            else:
                x = layer(layer_input)

        # insert squeeze-excite layer:
        elif layer.name.startswith("conv1d"):
            se_layer = layers.SqueezeExcite(
                activation=None, # no activation before squeezing
                additive=False, # use sigmoid multiplicative scaling
                bottleneck_ratio=bottleneck_ratio, # bottleneck ratio
                use_bias=False, # ignore bias
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3), # near-zero weight initialization
                scale_fun='tanh'
            )
            x = layer(layer_input)
            x = x + se_layer(x)

        elif layer.name.startswith("separable_conv1d"):
            if insert_mode=='all':
                se_layer = layers.SqueezeExcite(
                    activation=None, # no activation before squeezing
                    additive=False, # use sigmoid multiplicative scaling
                    bottleneck_ratio=bottleneck_ratio, # bottleneck ratio
                    use_bias=False, # ignore bias
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3), # near-zero weight initialization
                    scale_fun='tanh'
                )
                x = layer(layer_input)
                x = x + se_layer(x)
            else:
                x = layer(layer_input)
                
        else:
            x = layer(layer_input)
    
        # save the output tensor of every layer
        layer_output_dict_new.update({layer.name: x})
    
    final = layers.SwitchReverse(strand_pair)([layer_output_dict_new[model.layers[-1].name], reverse_bool])
    model_final = tf.keras.Model(inputs=model.inputs, outputs=final)

    # set trainable
    for l in model_final.layers[:-2]: # trunk
        if re.match('layer_normalization|adapter_houlsby', l.name): 
            l.trainable = True
        else:
            l.trainable = False

    for l in model_final.layers: # set trunk
        if l.name.startswith("squeeze_excite"): l.trainable = True

    if unfreeze_bn:
        for l in model_final.layers:
            if l.name.startswith("batch_normalization"): l.trainable=True

    # expected number of trainable params added/unfrozen:
    params_added = 0
    for l in model_final.layers:
        if l.name.startswith("squeeze_excite"): 
            params_added += param_count(l)
        elif l.name.startswith("batch_normalization"): 
            if unfreeze_bn: params_added += param_count(l, type='trainable')
        elif l.name.startswith("adapter_houlsby"): 
            params_added += param_count(l)
        elif l.name.startswith("layer_normalization"): 
            params_added += param_count(l, type='trainable')
    print('params added/unfrozen by se_block: %d'%params_added)
    
    return model_final

###############
# modify json #
###############
# houlsby and squeeze-excite
def modify_json(input_json, output_json, adapter='adapterHoulsby', latent=None, conv=None, se_ratio=None):

    with open(input_json) as params_open:
        params = json.load(params_open)

    # houlsby #
    if adapter=='adapterHoulsby':
        params["model"]["trunk"][2]['adapter']= 'houlsby'
        params["model"]["trunk"][2]['latent']= latent

    # squeeze-excite #
    if conv=='se_all' or conv=='se_all_bn':
        for i in [0, 1, 3, 4]:
            params['model']['trunk'][i]['transfer_se']=True
            params['model']['trunk'][i]['se_ratio']=se_ratio
    
    elif conv=='se' or conv=='se_bn':
        for i in [0, 1]:
            params['model']['trunk'][i]['transfer_se']=True
            params['model']['trunk'][i]['se_ratio']=se_ratio

    else:
        pass
        
    ### output
    with open(output_json, 'w') as params_open:
        json.dump(params, params_open, indent=4)


######################
# merge lora weights #
######################
def merge_lora_layer(lora_layer):
    down_weights = lora_layer.down_layer.kernel
    up_weights = lora_layer.up_layer.kernel
    increment_weights = tf.einsum("ab,bc->ac", down_weights, up_weights) * lora_layer.scale
    lora_layer.original_layer.kernel.assign_add(increment_weights)
    return lora_layer.original_layer

def merge_lora(input_model, mode='default'):
    for layer in input_model.layers:
        if 'multihead_attention' in layer.name:
            # default loRA
            layer._q_layer = merge_lora_layer(layer._q_layer)
            layer._v_layer = merge_lora_layer(layer._v_layer)
            if mode=='full':
                layer._k_layer = merge_lora_layer(layer._k_layer)
                layer._embedding_layer = merge_lora_layer(layer._embedding_layer)
    input_model(input_model.input)

# correct weights.h5 weight order
def var_reorder(weight_h5):
    # assumes weight_h5 model saved with seqnn_model.save()
    # [i.name for i in model.layers[30].weights] to check for multihead_attention layer weights order.
    # model.load_weights() load weights sequencially, assuming layer weights are in the right order.
    # When inserting lora/ia3, multihead_attention layer weights order changed.
    # multihead_attention layer weights order is saved inside f['model_weights']['multihead_attention'].attrs
    # After saving the weight_merged model, we need to go into the weights.h5, and change the attrs in multihead attention.
    var_init_order = ['r_w_bias:0:0',
                      'r_r_bias:0:0', 
                      'q_layer/kernel:0', 
                      'k_layer/kernel:0',
                      'v_layer/kernel:0',
                      'embedding_layer/kernel:0',
                      'embedding_layer/bias:0',
                      'r_k_layer/kernel:0']

    f = h5py.File(weight_h5, 'r+')
    layers = [i for i in list(f['model_weights'].keys()) if 'multihead_attention' in i]
    for l_name in layers:
        new_name_order = [l_name+'/'+i for i in var_init_order]
        f['model_weights'][l_name].attrs.modify(name='weight_names', value=new_name_order)
    f.close()

#####################
# merge ia3 weights #
#####################
def merge_ia3_layer(ia3_layer, type='kv'):
    scaler = ia3_layer._ia3_layer.kernel[0]
    ia3_layer.original_layer.kernel.assign(ia3_layer.original_layer.kernel * scaler)
    if type=='embedding':
        ia3_layer.original_layer.bias.assign(ia3_layer.original_layer.bias * scaler)
    return ia3_layer.original_layer

def merge_ia3(input_model):
    for layer in input_model.layers:
        if 'multihead_attention' in layer.name:
            layer._k_layer = merge_ia3_layer(layer._k_layer, type='kv')
            layer._v_layer = merge_ia3_layer(layer._v_layer, type='kv')
            layer._embedding_layer = merge_ia3_layer(layer._embedding_layer, type='embedding')
    input_model(input_model.input)

