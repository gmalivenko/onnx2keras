import logging

import keras
import numpy as np
import tensorflow as tf

from .utils import ensure_tf_type
from .tfops_funcs import tf_math_reduce_mean, tf_math_reduce_variance, tf_sqrt, tf_rank, tf_concat, tf_reshape


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
            name=f"{params['cleaned_name']}_bn"
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=weights,
            name=f"{params['cleaned_name']}_bn"
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
    var = tf_math_reduce_variance(input_0, axis=axis, keepdims=True, name=None, tf_name=f"{params['cleaned_name']}_var")
    mean = tf_math_reduce_mean(input_0, axis=axis, keepdims=True, name=None, tf_name=f"{params['cleaned_name']}_mean")
    dim_ones = (1,) * (dims_x - 2)
    scale = np.reshape(scale, (-1, *dim_ones))
    bias = np.reshape(bias, (-1, *dim_ones))
    layers[node_name] = (input_0 - mean) * scale / tf_sqrt(var + epsilon, tf_name=f"{params['cleaned_name']}_sqrt")\
                        + bias


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
    lambda_layer = keras.layers.Dropout(ratio, name=f"{params['cleaned_name']}_dropout")
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

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_lrn")
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_layernorm(node, params, layers, lambda_func, node_name, keras_name):
    axis = params.get('axis', -1)
    epsilon = params.get('epsilon', 1e-05)
    stash_type = params.get('stash_type')
    if stash_type is not None:
        raise Exception("LayerNorm stash_type attribute is not implemented")
    input_x = layers[node.input[0]]
    weight = layers[node.input[1]]
    if len(node.input) > 2:
        bias = layers[node.input[2]]
    else:
        bias = None
    center = True if bias is not None else False
    layer_norm = tf.keras.layers.LayerNormalization(
        axis=axis,
        epsilon=epsilon,
        center=center,
        name=f"{params['cleaned_name']}_LayerNorm"
    )
    input_shape = input_x.shape.as_list()
    if input_shape[axis] is None:
        # reshape input such that the axis dim would be non-None (set by weights)
        tf_input_shape = tf.shape(input_x)
        if axis < 0:
            axis = tf_rank(input_x, tf_name=f"{params['cleaned_name']}_LayerNorm_rank")._inferred_value[0] + axis
        tf_new_shape = tf_concat([tf_input_shape[:axis], [weight.shape[0]], tf_input_shape[axis+1:]], axis=-1,
                                 tf_name=f"{params['cleaned_name']}_LayerNorm_new_shape")
        input_x = tf_reshape(input_x, tf_new_shape, tf_name=f"{params['cleaned_name']}_LayerNorm_reshape_none_axis")
    layer_norm.build(input_x.shape)
    if center:
        layer_norm.set_weights([weight, bias])
    else:
        layer_norm.set_weights([weight])
    layers[node_name] = layer_norm(input_x)
