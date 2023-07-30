import logging
from functools import partial
from typing import List

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from .utils import ensure_tf_type, is_numpy


def calculate_permute_values(n_dims: int, to_channel_first: bool) -> List[int]:
    if to_channel_first:
        return [n_dims - 1] + list(range(1, n_dims - 1))
    else:
        return list(range(2, n_dims)) + [1]


def permute_wrap_conv_if_constant(partial_func, conv_input, is_constant, conv_channels):
    if is_constant:
        input_shape = tf.shape(conv_input)
        permuted = keras.layers.Permute(calculate_permute_values(len(input_shape), to_channel_first=False))(conv_input)
        conv_res = partial_func(data_format="channels_last")(permuted)
        result = keras.layers.Permute(calculate_permute_values(len(input_shape), to_channel_first=True))(conv_res)
    else:
        data_fmt = keras.backend.image_data_format()
        conv = partial_func(data_format=data_fmt)
        if conv_input.shape[-1] is None:
            conv.build((None, conv_channels, *conv_input.shape[2:]))
        result = conv(conv_input)
    return result


def convert_conv(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.conv')

    if len(node.input) == 3:
        logger.debug('Conv with bias')
        # Has bias
        has_bias = True
        W = layers[node.input[1]]
        bias = layers[node.input[2]]

    elif len(node.input) == 2:
        logger.debug('Conv without bias')
        has_bias = False
        W = layers[node.input[1]]
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    n_groups = params['group'] if 'group' in params else 1
    dilation = params['dilations'][0] if 'dilations' in params else 1
    pads = params['pads'] if 'pads' in params else [0, 0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1, 1]

    if len(W.shape) == 5:  # 3D conv
        logger.debug('3D convolution')
        if pads[0] > 0 or pads[1] > 0 or pads[2] > 0:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = keras_name + '_pad'
            padding_layer = keras.layers.ZeroPadding3D(
                padding=(pads[0], pads[1], pads[2]),
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
        out_channels, channels_per_group, dimension, height, width = W.shape
        W = W.transpose(2, 3, 4, 1, 0)

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]
        conv_args = {"filters": out_channels,
                     "kernel_size": (dimension, height, width),
                     "strides": (strides[0], strides[1], strides[2]),
                     "padding": 'valid',
                     "weights": weights,
                     "use_bias": has_bias,
                     "activation": None,
                     "dilation_rate": dilation,
                     "name": keras_name,
                     "groups": n_groups}
        partial_conv = partial(keras.layers.Conv3D, **conv_args)
        layers[node_name] = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2])

    elif len(W.shape) == 4:  # 2D conv
        logger.debug('2D convolution')

        padding = None
        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            padding = (pads[0], pads[1])
        elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
            padding = ((pads[0], pads[2]), (pads[1], pads[3]))

        if padding:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = keras_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=padding,
                name=padding_name,
                data_format='channels_first'
            )
            layers[padding_name] = input_0 = padding_layer(input_0)

        W = W.transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        conv_args = {"filters": out_channels,
                     "kernel_size": (height, width),
                     "strides": (strides[0], strides[1]),
                     "padding": 'valid',
                     "weights": weights,
                     "use_bias": has_bias,
                     "activation": None,
                     "dilation_rate": dilation,
                     "name": keras_name,
                     "groups": n_groups}
        partial_conv = partial(keras.layers.Conv2D, **conv_args)
        layers[node_name] = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2])
    else:
        # 1D conv
        W = W.transpose(2, 1, 0)
        width, channels, n_filters = W.shape
        print(width, channels, n_filters, has_bias)

        weights = [W]
        conv_args = {"filters": n_filters,
                     "kernel_size": (width),
                     "strides": (strides[0]),
                     "weights": weights,
                     "use_bias": False,
                     "activation": None,
                     "dilation_rate": dilation,
                     "name": keras_name,
                     "groups": n_groups}

        padding = None
        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            padding = (pads[0], pads[1])

        if padding:
            # find the dimension to pad and use the exact padding values
            input_shape = np.asarray(keras.backend.int_shape(input_0))
            partitioned_dim = np.argwhere(input_shape == channels * n_groups)[0][0]
            padding_dim = 2 if partitioned_dim == 1 else 1
            tf_padding = np.zeros((2, len(input_shape))).astype(int)
            tf_padding[:, padding_dim] = [padding[0], padding[1]]
            input_0 = tf.pad(input_0, tf.constant(list(tf_padding.transpose())))
        else:
            conv_args['padding'] = 'valid'
        partial_conv = partial(keras.layers.Conv1D, **conv_args)
        res = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2])
        if has_bias:
            res_shape = np.asarray(keras.backend.int_shape(res))
            bias_dim = np.argwhere(res_shape == bias.shape)[0][0]
            expanded_dims = [dim for dim in range(len(res_shape)) if dim != bias_dim]
            res = res + np.expand_dims(bias, expanded_dims)

        layers[node_name] = res


def convert_convtranspose(node, params, layers,
                          lambda_func, node_name, keras_name):
    """
    Convert transposed convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.convtranpose')

    if len(node.input) == 3:
        logger.debug('ConvTranspose with bias')
        # Has bias
        has_bias = True
        W = layers[node.input[1]]
        bias = layers[node.input[2]]

    elif len(node.input) == 2:
        logger.debug('ConvTranspose without bias')
        has_bias = False
        W = layers[node.input[1]]
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    n_groups = params['group'] if 'group' in params else 1
    dilation = params['dilations'][0] if 'dilations' in params else 1
    pads = params['pads'] if 'pads' in params else [0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1]

    if len(W.shape) == 5:  # 3D conv
        W = W.transpose(2, 3, 4, 1, 0)
        height, width, depth, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if n_groups > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with groups != 1')

        if dilation > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')

        conv = keras.layers.Conv3DTranspose(
            filters=n_filters,
            kernel_size=(height, width, depth),
            strides=strides,
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            name=keras_name
        )

        if 'output_shape' in params and 'pads' not in params:
            logger.debug('!!!!! Paddings will be calculated automatically !!!!!')
            pads = [strides[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][0],
                    strides[1] * (int(input_0.shape[3]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][1]]

        layers[node_name] = input_0 = conv(input_0)

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        # input_0.set_shape(input_0.shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if pads[0] > 0:
            logger.debug('Add cropping layer for output padding')
            assert (len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=keras_name + '_crop'
            )
            layers[node_name] = crop(input_0)

    elif len(W.shape) == 4:  # 2D conv
        W = W.transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if n_groups > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with groups != 1')

        if dilation > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=strides,
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            name=keras_name
        )

        if 'output_shape' in params and 'pads' not in params:
            logger.debug('!!!!! Paddings will be calculated automatically !!!!!')
            pads = [strides[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][0],
                    strides[1] * (int(input_0.shape[3]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][1]]

        layers[node_name] = input_0 = conv(input_0)

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        # input_0.set_shape(input_0.shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if pads[0] > 0:
            logger.debug('Add cropping layer for output padding')
            assert (len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=keras_name + '_crop'
            )
            layers[node_name] = crop(input_0)
    else:
        raise AttributeError('Layer is not supported for now')
