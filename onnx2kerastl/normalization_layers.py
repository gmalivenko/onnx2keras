import logging

import keras
import numpy as np
import tensorflow as tf

from .utils import ensure_tf_type


def convert_batchnorm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert BatchNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.batchnorm2d')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    if len(node.input) == 5:
        weights = [
            layers[node.input[1]],
            layers[node.input[2]],
            layers[node.input[3]],
            layers[node.input[4]]
        ]
    elif len(node.input) == 3:
        weights = [
            layers[node.input[1]],
            layers[node.input[2]]
        ]
    else:
        raise AttributeError('Unknown arguments for batch norm')

    eps = params['epsilon'] if 'epsilon' in params else 1e-05  # default epsilon
    momentum = params['momentum'] if 'momentum' in params else 0.9  # default momentum

    if isinstance(keras_name, list):
        keras_name = keras_name[0]

    if len(weights) == 2:
        logger.debug('Batch normalization without running averages')
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=weights,
            name=keras_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=weights,
            name=keras_name
        )

    layers[node_name] = bn(input_0)


def convert_instancenorm(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert InstanceNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    # based on https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
    logger = logging.getLogger('onnx2keras.instancenorm2d')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    if len(node.input) == 3:
        scale = layers[node.input[1]]
        bias = layers[node.input[2]]
    else:
        raise AttributeError('Unknown arguments for instance norm')

    epsilon = params['epsilon']
    dims_x = len(input_0.shape)
    axis = list(range(2, dims_x))
    var = tf.math.reduce_variance(input_0, axis=axis, keepdims=True, name=None)
    mean = tf.math.reduce_mean(input_0, axis=axis, keepdims=True, name=None)
    dim_ones = (1,) * (dims_x - 2)
    scale = np.reshape(scale, (-1, *dim_ones))
    bias = np.reshape(bias, (-1, *dim_ones))
    layers[node_name] = (input_0 - mean) * scale / tf.sqrt(var + epsilon) + bias


def convert_dropout(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Dropout layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.dropout')

    # In ONNX Dropout returns dropout mask as well.
    if isinstance(keras_name, list) and len(keras_name) > 1:
        keras_name = keras_name[0]

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    ratio = params['ratio'] if 'ratio' in params else 0.0
    lambda_layer = keras.layers.Dropout(ratio, name=keras_name)
    layers[node_name] = lambda_layer(input_0)


def convert_lrn(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert LRN layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.LRN')
    logger.debug('LRN can\'t be tested with PyTorch exporter, so the support is experimental.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, depth_radius=params['size'], bias=params['bias'], alpha=params['alpha'], beta=params['beta']):
        import tensorflow as tf
        from keras import backend as K
        data_format = 'NCHW' if K.image_data_format() == 'channels_first' else 'NHWC'

        if data_format == 'NCHW':
            x = tf.transpose(x, [0, 2, 3, 1])

        layer = tf.nn.local_response_normalization(
            x, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta
        )

        if data_format == 'NCHW':
            layer = tf.transpose(x, [0, 3, 1, 2])

        return layer

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer
