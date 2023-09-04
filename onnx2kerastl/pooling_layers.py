import keras
import logging
from .utils import ensure_tf_type
import numpy as np
import string
import random
import tensorflow as tf

def convert_maxpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert MaxPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.maxpool')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]
    pad = 'valid'

    if all([shape % 2 == 1 for shape in kernel_shape]) and \
       all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
       all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = keras_name + '_pad'
        if len(kernel_shape) == 2:
            padding = None

            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))

            if padding is not None:
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=padding,
                    name=padding_name
                )
                layers[padding_name] = input_0 = padding_layer(input_0)
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling = keras.layers.MaxPooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.MaxPooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    ceil_mode = params.get('ceil_mode', False)
    if ceil_mode:
        if pad == 'valid':
            output_shape = ((np.array(input_0.shape[-len(kernel_shape):]) - np.array(kernel_shape)) / np.array(
                stride_shape)) + 1
        else:
            output_shape = np.floor((np.array(input_0.shape[-len(kernel_shape):]) - 1) / np.array(stride_shape)) + 1
        if not np.array([output_shape[i].is_integer() for i in range(len(output_shape))]).all():
            padding = [0 if output_shape[i].is_integer() else stride_shape[i] for i in range(len(kernel_shape))]
            rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
            if len(kernel_shape) == 2:
                layers[node_name + "_pre_" + rand_string] = keras.layers.ZeroPadding2D(
                    ((0, padding[0]), (0, padding[1])))(input_0)
            else:
                layers[node_name + "_pre_" + rand_string] = keras.layers.ZeroPadding3D(
                    ((0, padding[0]), (0, padding[1]), (0, padding[2])))(input_0)
            input_0 = layers[node_name + "_pre_" + rand_string]
    layers[node_name] = pooling(input_0)


def convert_avgpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert AvgPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.avgpool')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]

    if not any(pads):
        pad = 'valid'

    elif all([shape % 2 == 1 for shape in kernel_shape]) and \
       all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
       all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        pad = 'valid'
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = keras_name + '_pad'
        if len(kernel_shape) == 2:
            padding_layer = keras.layers.ZeroPadding2D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        layers[padding_name] = input_0 = padding_layer(input_0)

    if len(kernel_shape) == 2:
        pooling = keras.layers.AveragePooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    elif len(kernel_shape) == 1:
        pooling = keras.layers.AveragePooling1D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.AveragePooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    layers[node_name] = pooling(input_0)


def convert_global_avg_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    tensor_dim = len(input_0.shape)
    if tensor_dim == 3:
        global_pool = keras.layers.GlobalAveragePooling1D(data_format='channels_first', name=keras_name)
    elif tensor_dim == 4:
        global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=keras_name)
    elif tensor_dim == 5:
        global_pool = keras.layers.GlobalAveragePooling3D(data_format='channels_first', name=keras_name)
    else:
        raise NotImplementedError("Global average pooling of dims < 3 or dims > 5 is not supported")
    input_0 = global_pool(input_0)
    new_shape = input_0.shape.as_list()
    new_shape = new_shape[1:]
    new_shape.extend([1]*(tensor_dim-2))
    reshape_layer = keras.layers.Reshape(new_shape)
    input_0 = reshape_layer(input_0)

    layers[node_name] = input_0


def convert_topk(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert topk layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    axis = params.get('axis', -1)
    largest = bool(params.get('largest', 1))
    to_sort = bool(params.get('sorted', 1))
    x = layers[node.input[0]]
    k = layers[node.input[1]][0]
    if not largest:
        in_tensor = -x
    else:
        in_tensor = x

    def target_layer(in_tensor,k=k, to_sort=to_sort, axis=axis):
        rank = len(in_tensor.shape)
        if axis >= rank-1 or axis == -1:
            permuted = in_tensor
        else:
            ord_permute = np.arange(rank)
            ord_permute[axis] = rank-1
            ord_permute[-1] = axis
            permuted = tf.transpose(in_tensor, ord_permute)
        topk_res = tf.math.top_k(permuted, k=k, sorted=to_sort)
        values_pre_permute = topk_res[0]
        indices_pre_permute = topk_res[1]
        topk_concat = tf.stack([values_pre_permute, tf.cast(indices_pre_permute, tf.float32)])
        if axis >= rank - 1 or axis == -1:
            out = topk_concat
        else:
            ord_permute = [0] + (ord_permute+1).tolist()
            out = tf.transpose(topk_concat, ord_permute)
        return out

    lambda_layer = keras.layers.Lambda(target_layer)
    result = lambda_layer(in_tensor)
    values = result[0]
    indices = tf.cast(result[1], tf.int32)
    if not largest:
        out_tensor = -values
    else:
        out_tensor = values
    layers[keras_name[0]] = out_tensor
    layers[keras_name[1]] = indices
