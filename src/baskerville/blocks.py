# Copyright 2019 Calico LLC
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
import pdb
import numpy as np
import tensorflow as tf

from baskerville import layers


############################################################
# Convolution
############################################################
def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    activation="relu",
    activation_end=None,
    stride=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    pool_size=1,
    pool_type="max",
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    residual=False,
    kernel_initializer="he_normal",
    padding="same",
):
    """Construct a single convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        Conv1D stride
      dilation_rate: Conv1D dilation rate
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      residual:      Residual connection boolean
      pool_size:     Max pool width
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum
      norm_gamma:    BatchNorm gamma (defaults according to residual)

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = layers.activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=(norm_type is None),
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # normalize
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma
        )(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(
            current
        )

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = layers.activate(current, activation_end)

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
        else:
            current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def conv_dna(
    inputs,
    filters=None,
    kernel_size=15,
    activation="relu",
    stride=1,
    l2_scale=0,
    residual=False,
    dropout=0,
    dropout_residual=0,
    pool_size=1,
    pool_type="max",
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    use_bias=None,
    se=False,
    conv_type="standard",
    kernel_initializer="he_normal",
    padding="same",
):
    """Construct a single convolution block, assumed to be operating on DNA.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        Conv1D stride
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      pool_size:     Max pool width
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # need option to define for older models
    if use_bias is None:
        use_bias = norm_type is None and not residual

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # squeeze-excite
    if se:
        current = squeeze_excite(current)

    if residual:
        # residual conv block
        rcurrent = conv_nac(
            current,
            activation=activation,
            l2_scale=l2_scale,
            dropout=dropout_residual,
            conv_type=conv_type,
            norm_type=norm_type,
            se=se,
            bn_momentum=bn_momentum,
            kernel_initializer=kernel_initializer,
        )

        # residual add
        rcurrent = layers.Scale()(rcurrent)
        current = tf.keras.layers.Add()([current, rcurrent])

    else:
        # normalize
        if norm_type == "batch-sync":
            current = tf.keras.layers.BatchNormalization(
                momentum=bn_momentum, synchronized=True
            )(current)
        elif norm_type == "batch":
            current = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current)
        elif norm_type == "layer":
            current = tf.keras.layers.LayerNormalization()(current)

        # activation
        current = layers.activate(current, activation)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
        else:
            current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def conv_nac(
    inputs,
    filters=None,
    kernel_size=1,
    activation="relu",
    stride=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    residual=False,
    pool_size=1,
    pool_type="max",
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    kernel_initializer="he_normal",
    padding="same",
    se=False,
):
    """Construct a single convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        Conv1D stride
      dilation_rate: Conv1D dilation rate
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      residual:      Residual connection boolean
      pool_size:     Max pool width
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # normalize
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization()(current)

    # activation
    current = layers.activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=True,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # squeeze-excite
    if se:
        current = squeeze_excite(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
        else:
            current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def conv_next(
    inputs,
    filters=None,
    kernel_size=7,
    activation="relu",
    dense_expansion=2.0,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    residual=False,
    pool_size=1,
    pool_type="max",
    kernel_initializer="he_normal",
    padding="same",
    norm_type=None,
    bn_momentum=0.99,
):
    """Construct a single convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      dilation_rate: Conv1D dilation rate
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      residual:      Residual connection boolean
      pool_size:     Max pool width
      bn_momentum:   BatchNorm momentum

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    if filters is None:
        filters = inputs.shape[-1]

    # flow through variable current
    current = inputs

    # convolution
    current = tf.keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=True,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # normalize
    current = tf.keras.layers.LayerNormalization(epsilon=1e-5)(current)

    # dense expansion
    expansion_filters = int(dense_expansion) * filters
    current = tf.keras.layers.Dense(
        units=expansion_filters,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # activation
    current = layers.activate(current, activation)

    # dense contraction
    current = tf.keras.layers.Dense(
        units=filters,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # Pool
    if pool_size > 1:
        if pool_type == "softmax":
            current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
        else:
            current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def unet_conv(
    inputs,
    unet_repr,
    activation="relu",
    stride=2,
    l2_scale=0,
    dropout=0,
    norm_type=None,
    bn_momentum=0.99,
    kernel_size=1,
    kernel_initializer="he_normal",
    upsample_conv=False,
):
    """Construct a feature pyramid network block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        UpSample stride
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum
      upsample_conv: Conv1D the upsampled input path

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # variables
    current1 = inputs
    current2 = unet_repr

    # normalize
    if norm_type == "batch-sync":
        current1 = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current1)
        current2 = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current2)
    elif norm_type == "batch":
        current1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current1)
        current2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current2)
    elif norm_type == "layer":
        current1 = tf.keras.layers.LayerNormalization()(current1)
        current2 = tf.keras.layers.LayerNormalization()(current2)

    # activate
    current1 = layers.activate(current1, activation)
    current2 = layers.activate(current2, activation)

    filters = inputs.shape[-1]

    # dense
    if upsample_conv:
        current1 = tf.keras.layers.Dense(
            units=filters,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
            kernel_initializer=kernel_initializer,
        )(current1)
    current2 = tf.keras.layers.Dense(
        units=filters,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current2)

    # upsample
    current1 = tf.keras.layers.UpSampling1D(size=stride)(current1)

    # add
    current = tf.keras.layers.Add()([current1, current2])

    # normalize?
    # activate?

    # convolution
    current = tf.keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    return current


def unet_concat(
    inputs,
    unet_repr,
    activation="relu",
    stride=2,
    l2_scale=0,
    dropout=0,
    norm_type=None,
    bn_momentum=0.99,
    kernel_size=1,
    kernel_initializer="he_normal",
):
    """Construct a single transposed convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        UpSample stride
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum

    Returns:
      [batch_size, stride*seq_length, features] output sequence
    """

    # variables
    current1 = inputs
    current2 = unet_repr

    # normalize
    if norm_type == "batch-sync":
        current1 = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current1)
        current2 = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current2)
    elif norm_type == "batch":
        current1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current1)
        current2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current2)
    elif norm_type == "layer":
        current1 = tf.keras.layers.LayerNormalization()(current1)
        current2 = tf.keras.layers.LayerNormalization()(current2)

    # upsample
    current1 = tf.keras.layers.UpSampling1D(size=stride)(current1)

    # concatenate
    current = tf.keras.layers.Concatenate()([current2, current1])

    # activate
    current = layers.activate(current, activation)

    # dense
    mid_units = int(1.5 * unet_repr.shape[-1])
    current = tf.keras.layers.Dense(
        units=mid_units,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # activate
    current = layers.activate(current, activation)

    # dense
    current = tf.keras.layers.Conv1D(
        filters=unet_repr.shape[-1],
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # residual
    current = tf.keras.layers.Add()([unet_repr, current])

    return current


def tconv_nac(
    inputs,
    filters=None,
    kernel_size=1,
    activation="relu",
    stride=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    kernel_initializer="he_normal",
    padding="same",
):
    """Construct a single transposed convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      stride:        UpSample stride
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      norm_type:     Apply batch or layer normalization
      bn_momentum:   BatchNorm momentum

    Returns:
      [batch_size, stride*seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    if filters is None:
        filters = inputs.shape[-1]

    # normalize
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization()(current)

    # activation
    current = layers.activate(current, activation)

    # convolution
    current = tf.keras.layers.Conv1DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    return current


def conv_block_2d(
    inputs,
    filters=128,
    activation="relu",
    conv_type="standard",
    kernel_size=1,
    stride=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    pool_size=1,
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma="ones",
    kernel_initializer="he_normal",
    symmetric=False,
):
    """Construct a single 2D convolution block."""

    # flow through variable current
    current = inputs

    # activation
    current = layers.activate(current, activation)

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv2D
    else:
        conv_layer = tf.keras.layers.Conv2D

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=(norm_type is None),
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # normalize
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma
        )(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(
            current
        )

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(
            current
        )

    # symmetric
    if symmetric:
        current = layers.Symmetrize2D()(current)

    return current


############################################################
# Towers
############################################################
def conv_tower_v1(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
    """Construct a reducing convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      repeat:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    for ri in range(repeat):
        # convolution
        current = conv_block(current, filters=int(np.round(rep_filters)), **kwargs)

        # update filters
        rep_filters *= filters_mult

    return current


def conv_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    divisible_by=1,
    repeat=1,
    reprs=[],
    **kwargs,
):
    """Construct a reducing convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_end:   End Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      divisible_by:  Round filters to be divisible by (eg a power of two)
      repeat:        Tower repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # convolution
        current = conv_block(current, filters=_round(rep_filters), **kwargs)

        # save representation
        reprs.append(current)

        # update filters
        rep_filters *= filters_mult

    return current


def conv_tower_nac(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    divisible_by=1,
    repeat=1,
    reprs=[],
    **kwargs,
):
    """Construct a reducing convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_end:   End Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      divisible_by:  Round filters to be divisible by (eg a power of two)
      repeat:        Tower repetitions
      reprs:         Append representations.

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # convolution
        current = conv_nac(current, filters=_round(rep_filters), **kwargs)

        # save representation
        reprs.append(current)

        # update filters
        rep_filters *= filters_mult

    return current


def res_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    kernel_size=1,
    dropout=0,
    pool_size=2,
    pool_type="max",
    divisible_by=1,
    repeat=1,
    num_convs=2,
    reprs=[],
    **kwargs,
):
    """Construct a reducing convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_end:   End Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      kernel_size:   Conv1D kernel_size
      dropout:       Dropout on subsequent convolution blocks.
      pool_size:     Pool width.
      repeat:        Residual block repetitions
      num_convs:     Conv blocks per residual layer

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        rep_filters_int = _round(rep_filters)

        # initial
        current = conv_nac(
            current, filters=rep_filters_int, kernel_size=kernel_size, **kwargs
        )
        current0 = current

        # subsequent
        for ci in range(1, num_convs):
            # bg = 'ones' if ci < num_convs-1 else 'zeros'
            current = conv_nac(current, filters=rep_filters_int, **kwargs)

        # dropout
        if dropout > 0:
            current = tf.keras.layers.Dropout(rate=dropout)(current)

        # residual add
        if num_convs > 1:
            current = layers.Scale()(current)
            current = tf.keras.layers.Add()([current0, current])

        # save representation
        reprs.append(current)

        # pool
        if pool_size > 1:
            if pool_type == "softmax":
                current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
            else:
                current = tf.keras.layers.MaxPool1D(
                    pool_size=pool_size, padding="same"
                )(current)

        # update filters
        rep_filters *= filters_mult

    return current


def convnext_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    kernel_size=1,
    dropout=0,
    pool_size=2,
    pool_type="max",
    divisible_by=1,
    repeat=1,
    num_convs=2,
    reprs=[],
    **kwargs,
):
    """Abc.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_end:   End Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      kernel_size:   Conv1D kernel_size
      dropout:       Dropout on subsequent convolution blocks.
      pool_size:     Pool width.
      repeat:        Residual block repetitions
      num_convs:     Conv blocks per residual layer

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        rep_filters_int = _round(rep_filters)

        # initial
        current = conv_next(
            current,
            filters=rep_filters_int,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs,
        )
        current0 = current

        # subsequent
        for ci in range(1, num_convs):
            # bg = 'ones' if ci < num_convs-1 else 'zeros'
            current = conv_next(
                current,
                filters=rep_filters_int,
                kernel_size=kernel_size,
                dropout=dropout,
                **kwargs,
            )

        # residual add
        if num_convs > 1:
            current = layers.Scale()(current)
            current = tf.keras.layers.Add()([current0, current])

        # pool
        if pool_size > 1:
            if pool_type == "softmax":
                current = layers.SoftmaxPool1D(pool_size=pool_size)(current)
            else:
                current = tf.keras.layers.MaxPool1D(
                    pool_size=pool_size, padding="same"
                )(current)

        # save representation
        reprs.append(current)

        # update filters
        rep_filters *= filters_mult

    return current


############################################################
# Attention
############################################################
def transformer(
    inputs,
    key_size=None,
    heads=1,
    out_size=None,
    activation="relu",
    dense_expansion=2.0,
    content_position_bias=True,
    dropout=0.25,
    attention_dropout=0.05,
    position_dropout=0.01,
    l2_scale=0,
    mha_l2_scale=0,
    num_position_features=None,
    qkv_width=1,
    mha_initializer="he_normal",
    kernel_initializer="he_normal",
    seqlen_train=None,
    **kwargs,
):
    """Construct a transformer block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      key_size:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """
    if out_size is None:
        out_size = inputs.shape[-1]
        assert out_size % heads == 0
        value_size = out_size // heads

    # layer norm
    current = tf.keras.layers.LayerNormalization()(inputs)

    # multi-head attention
    current = layers.MultiheadAttention(
        value_size=value_size,
        key_size=key_size,
        heads=heads,
        num_position_features=num_position_features,
        attention_dropout_rate=attention_dropout,
        positional_dropout_rate=position_dropout,
        content_position_bias=content_position_bias,
        initializer=mha_initializer,
        l2_scale=mha_l2_scale,
        qkv_width=qkv_width,
        seqlen_train=seqlen_train,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # residual
    current = tf.keras.layers.Add()([inputs, current])

    if dense_expansion == 0:
        final = current
    else:
        final = transformer_dense(
            current, out_size, dense_expansion, l2_scale, dropout, kernel_initializer
        )

    return final


def transformer_split(
    inputs,
    splits=2,
    key_size=None,
    heads=1,
    out_size=None,
    activation="relu",
    dense_expansion=2.0,
    content_position_bias=True,
    dropout=0.25,
    attention_dropout=0.05,
    position_dropout=0.01,
    l2_scale=0,
    mha_l2_scale=0,
    num_position_features=None,
    qkv_width=1,
    mha_initializer="he_normal",
    kernel_initializer="he_normal",
    **kwargs,
):
    """Construct a transformer block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      key_size:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """
    if out_size is None:
        out_size = inputs.shape[-1]
        assert out_size % heads == 0
        value_size = out_size // heads

    # layer norm
    current = tf.keras.layers.LayerNormalization()(inputs)

    # multi-head attention
    mha = layers.MultiheadAttention(
        value_size=value_size,
        key_size=key_size,
        heads=heads,
        num_position_features=num_position_features,
        attention_dropout_rate=attention_dropout,
        positional_dropout_rate=position_dropout,
        content_position_bias=content_position_bias,
        initializer=mha_initializer,
        l2_scale=mha_l2_scale,
        qkv_width=qkv_width,
    )

    _, seq_len, seq_depth = inputs.shape
    seq_len2 = seq_len // 2
    seq_len4 = seq_len // 4

    if splits == 2:
        # split in two length-wise
        current = tf.keras.layers.Reshape((2, seq_len2, seq_depth))(current)

        # MHA left/right
        current_left = mha(current[:, 0, :, :])
        current_right = mha(current[:, 1, :, :])

        current_list = [current_left, current_right]

    elif splits == 3:
        # split in four length-wise
        current = tf.keras.layers.Reshape((4, seq_len4, seq_depth))(current)

        # transformer left/right
        current_left = mha(current[:, 0, :, :])
        current_right = mha(current[:, 3, :, :])

        # transformer center
        current_center = tf.keras.layers.Reshape((seq_len2, seq_depth))(
            current[:, 1:3, :, :]
        )
        current_center = mha(current_center)

        current_list = [current_left, current_center, current_right]

    else:
        print("transformer_split not implemented for splits > 3", sys.stderr)
        exit(1)

    # concat along position axis
    current = tf.keras.layers.Concatenate(axis=1)(current_list)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # residual
    current = tf.keras.layers.Add()([inputs, current])

    if dense_expansion == 0:
        final = current
    else:
        final = transformer_dense(
            current, out_size, dense_expansion, l2_scale, dropout, kernel_initializer
        )

    return final


def transformer_dense(
    inputs, out_size, dense_expansion, l2_scale, dropout, kernel_initializer
):
    """Transformer block dense portion."""
    # layer norm
    current = tf.keras.layers.LayerNormalization()(inputs)

    # dense
    expansion_filters = int(dense_expansion * out_size)
    current = tf.keras.layers.Dense(
        units=expansion_filters,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # activation
    current = layers.activate(current, "relu")

    # dense
    current = tf.keras.layers.Dense(
        units=out_size,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        kernel_initializer=kernel_initializer,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # residual
    final = tf.keras.layers.Add()([inputs, current])

    return final


def transformer2(
    inputs,
    key_size=None,
    heads=1,
    out_size=None,
    activation="relu",
    num_position_features=None,
    attention_dropout=0.05,
    position_dropout=0.01,
    dropout=0.25,
    dense_expansion=2.0,
    qkv_width=1,
    **kwargs,
):
    """Construct a transformer block, with length-wise pooling before
       returning to full length.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      key_size:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """
    if out_size is None:
        out_size = inputs.shape[-1]
        assert out_size % heads == 0
        value_size = out_size // heads

    # convolution to decrease length
    current = conv_nac(
        inputs,
        filters=min(4 * key_size, inputs.shape[-1]),
        kernel_size=3,
        pool_size=2,
        **kwargs,
    )

    # layer norm
    current = tf.keras.layers.LayerNormalization()(current)

    # multi-head attention
    current = layers.MultiheadAttention(
        value_size=value_size,
        key_size=key_size,
        heads=heads,
        num_position_features=num_position_features,
        attention_dropout_rate=attention_dropout,
        positional_dropout_rate=position_dropout,
        transpose_stride=2,
        qkv_width=qkv_width,
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(dropout)(current)

    # concatenate and transform
    current = tf.keras.layers.Concatenate()([inputs, current])
    current = tf.keras.layers.Dense(out_size)(current)

    # residual
    current = tf.keras.layers.Add()([inputs, current])

    if dense_expansion == 0:
        final = current
    else:
        current_mha = current

        # layer norm
        current = tf.keras.layers.LayerNormalization()(current)

        # dense
        expansion_filters = int(dense_expansion * out_size)
        current = tf.keras.layers.Dense(expansion_filters)(current)

        # dropout
        if dropout > 0:
            current = tf.keras.layers.Dropout(dropout)(current)

        # activation
        current = layers.activate(current, "relu")

        # dense
        current = tf.keras.layers.Dense(out_size)(current)

        # dropout
        if dropout > 0:
            current = tf.keras.layers.Dropout(dropout)(current)

        # residual
        final = tf.keras.layers.Add()([current_mha, current])

    return final


def swin_transformer(inputs, **kwargs):
    current = inputs
    current = transformer_split(current, splits=2, **kwargs)
    current = transformer_split(current, splits=3, **kwargs)
    return current


def transformer_tower(inputs, repeat=2, block_type="transformer", **kwargs):
    """Construct a tower of repeated transformer blocks.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      repeat:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    if block_type == "lambda":
        transformer_block = transformer_lambda
    elif block_type == "swin":
        transformer_block = swin_transformer
    elif block_type == "transformer2":
        transformer_block = transformer2
    else:
        transformer_block = transformer

    current = inputs
    for ri in range(repeat):
        current = transformer_block(current, **kwargs)
    return current


def squeeze_excite(
    inputs,
    activation="relu",
    bottleneck_ratio=8,
    additive=False,
    norm_type=None,
    bn_momentum=0.9,
    kernel_initializer="glorot_uniform",
    use_bias=True,
    scale_fun="sigmoid",
    **kwargs,
):
    return layers.SqueezeExcite(
        activation=activation,
        additive=additive,
        bottleneck_ratio=bottleneck_ratio,
        norm_type=norm_type,
        bn_momentum=bn_momentum,
        kernel_initializer=kernel_initializer,
        scale_fun=scale_fun,
        use_bias=use_bias,
    )(inputs)


def wheeze_excite(inputs, pool_size, **kwargs):
    return layers.WheezeExcite(pool_size)(inputs)


def global_context(inputs, **kwargs):
    return layers.GlobalContext()(inputs)


############################################################
# Dilated Towers
############################################################


def dilated_dense(
    inputs,
    filters,
    kernel_size=3,
    rate_mult=2,
    conv_type="standard",
    dropout=0,
    repeat=1,
    **kwargs,
):
    """Construct a residual dilated dense block.

    Args:

    Returns:
    """

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_block(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            conv_type=conv_type,
            **kwargs,
        )

        # dense concat
        current = tf.keras.layers.Concatenate()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult

    return current


def dilated_residual(
    inputs,
    filters,
    kernel_size=3,
    rate_mult=2,
    dropout=0,
    repeat=1,
    conv_type="standard",
    norm_type=None,
    round=False,
    **kwargs,
):
    """Construct a residual dilated convolution block.

    Args:

    Returns:
    """

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_block(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            conv_type=conv_type,
            norm_type=norm_type,
            norm_gamma="ones",
            **kwargs,
        )

        # return
        current = conv_block(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_type=norm_type,
            norm_gamma="zeros",
            **kwargs,
        )

        # InitZero
        if norm_type is None:
            current = layers.Scale()(current)

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult
        if round:
            dilation_rate = np.round(dilation_rate)

    return current


def dilated_residual_nac(
    inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs
):
    """Construct a residual dilated convolution block.

    Args:

    Returns:
    """

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_nac(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            **kwargs,
        )

        # return
        current = conv_nac(
            current, filters=rep_input.shape[-1], dropout=dropout, **kwargs
        )

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult

    return current


def dilated_residual_2d(
    inputs,
    filters,
    kernel_size=3,
    rate_mult=2,
    dropout=0,
    repeat=1,
    symmetric=True,
    **kwargs,
):
    """Construct a residual dilated convolution block."""

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_block_2d(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            norm_gamma="ones",
            **kwargs,
        )

        # return
        current = conv_block_2d(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_gamma="zeros",
            **kwargs,
        )

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # enforce symmetry
        if symmetric:
            current = layers.Symmetrize2D()(current)

        # update dilation rate
        dilation_rate *= rate_mult

    return current


############################################################
# Center ops
############################################################


def center_average(inputs, center, **kwargs):
    current = layers.CenterAverage(center)(inputs)
    return current


def center_slice(inputs, center, **kwargs):
    current = layers.CenterSlice(center)(inputs)
    return current


############################################################
# 2D
############################################################


def concat_dist_2d(inputs, **kwargs):
    current = layers.ConcatDist2D()(inputs)
    return current


def concat_position(inputs, transform="abs", power=1, **kwargs):
    current = layers.ConcatPosition(transform, power)(inputs)
    return current


def cropping_2d(inputs, cropping, **kwargs):
    current = tf.keras.layers.Cropping2D(cropping)(inputs)
    return current


def one_to_two(inputs, operation="mean", **kwargs):
    current = layers.OneToTwo(operation)(inputs)
    return current


def symmetrize_2d(inputs, **kwargs):
    return layers.Symmetrize2D()(inputs)


def upper_tri(inputs, diagonal_offset=2, **kwargs):
    current = layers.UpperTri(diagonal_offset)(inputs)
    return current


############################################################
# Factorization
############################################################


def factor_inverse(inputs, components_file, **kwargs):
    current = layers.FactorInverse(components_file)(inputs)
    return current


############################################################
# Dense
############################################################
def dense_block(
    inputs,
    units=None,
    activation="relu",
    activation_end=None,
    flatten=False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    kernel_initializer="he_normal",
    **kwargs,
):
    """Construct a single convolution block.

    Args:
      inputs:         [batch_size, seq_length, features] input sequence
      units:          Conv1D filters
      activation:     relu/gelu/etc
      activation_end: Compute activation after the other operations
      flatten:        Flatten across positional axis
      dropout:        Dropout rate probability
      l2_scale:       L2 regularization weight.
      l1_scale:       L1 regularization weight.
      residual:       Residual connection boolean
      batch_norm:     Apply batch normalization
      bn_momentum:    BatchNorm momentum
      norm_gamma:       BatchNorm gamma (defaults according to residual)

    Returns:
      [batch_size, seq_length(?), features] output sequence
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # activation
    current = layers.activate(current, activation)

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=(norm_type is None),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    # normalize
    if norm_gamma is None:
        norm_gamma = "zeros" if residual else "ones"
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma
        )(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(
            current
        )

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = layers.activate(current, activation_end)

    return current


def dense_nac(
    inputs,
    units=None,
    activation="relu",
    flatten=False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    norm_type=None,
    bn_momentum=0.99,
    norm_gamma=None,
    kernel_initializer="he_normal",
    **kwargs,
):
    """Construct a single convolution block.

    Args:
      inputs:         [batch_size, seq_length, features] input sequence
      units:          Conv1D filters
      activation:     relu/gelu/etc
      activation_end: Compute activation after the other operations
      flatten:        Flatten across positional axis
      dropout:        Dropout rate probability
      l2_scale:       L2 regularization weight.
      l1_scale:       L1 regularization weight.
      residual:       Residual connection boolean
      batch_norm:     Apply batch normalization
      bn_momentum:    BatchNorm momentum
      norm_gamma:       BatchNorm gamma (defaults according to residual)

    Returns:
      [batch_size, seq_length(?), features] output sequence
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # normalize
    if norm_gamma is None:
        norm_gamma = "zeros" if residual else "ones"
    if norm_type == "batch-sync":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma, synchronized=True
        )(current)
    elif norm_type == "batch":
        current = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, gamma_initializer=norm_gamma
        )(current)
    elif norm_type == "layer":
        current = tf.keras.layers.LayerNormalization(gamma_initializer=norm_gamma)(
            current
        )

    # activation
    current = layers.activate(current, activation)

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    return current


def final(
    inputs,
    units,
    activation="linear",
    flatten=False,
    kernel_initializer="he_normal",
    l2_scale=0,
    l1_scale=0,
    **kwargs,
):
    """Final simple transformation before comparison to targets.

    Args:
      inputs:         [batch_size, seq_length, features] input sequence
      units:          Dense units
      activation:     relu/gelu/etc
      flatten:        Flatten positional axis.
      l2_scale:       L2 regularization weight.
      l1_scale:       L1 regularization weight.

    Returns:
      [batch_size, seq_length(?), units] output sequence

    """
    current = inputs

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    return current


############################################################
# Dictionary
############################################################
name_func = {
    "center_slice": center_slice,
    "center_average": center_average,
    "concat_dist_2d": concat_dist_2d,
    "concat_position": concat_position,
    "conv_block": conv_block,
    "conv_dna": conv_dna,
    "conv_nac": conv_nac,
    "conv_next": conv_next,
    "conv_block_2d": conv_block_2d,
    "conv_tower": conv_tower,
    "conv_tower_nac": conv_tower_nac,
    "convnext_tower": convnext_tower,
    "cropping_2d": cropping_2d,
    "dense_block": dense_block,
    "dense_nac": dense_nac,
    "dilated_residual": dilated_residual,
    "dilated_residual_nac": dilated_residual_nac,
    "dilated_residual_2d": dilated_residual_2d,
    "dilated_dense": dilated_dense,
    "factor_inverse": factor_inverse,
    "final": final,
    "global_context": global_context,
    "one_to_two": one_to_two,
    "symmetrize_2d": symmetrize_2d,
    "squeeze_excite": squeeze_excite,
    "swin_transformer": swin_transformer,
    "res_tower": res_tower,
    "tconv_nac": tconv_nac,
    "transformer": transformer,
    "transformer_tower": transformer_tower,
    "unet_conv": unet_conv,
    "unet_concat": unet_concat,
    "upper_tri": upper_tri,
    "wheeze_excite": wheeze_excite,
}

keras_func = {
    "Conv1D": tf.keras.layers.Conv1D,
    "Cropping1D": tf.keras.layers.Cropping1D,
    "Cropping2D": tf.keras.layers.Cropping2D,
    "Dense": tf.keras.layers.Dense,
    "Flatten": tf.keras.layers.Flatten,
}
