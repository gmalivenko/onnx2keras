import keras.layers
import logging
from .utils import ensure_tf_type, ensure_numpy_type


def convert_batchnorm(node, params, layers, node_name):
    """
    Convert BatchNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:batchnorm2d')

    input_0 = ensure_tf_type(layers[node.input[0]])

    if len(node.input) == 5:
        weights = [
            ensure_numpy_type(layers[node.input[1]]),
            ensure_numpy_type(layers[node.input[2]]),
            ensure_numpy_type(layers[node.input[3]]),
            ensure_numpy_type(layers[node.input[4]])
        ]
    elif len(node.input) == 3:
        weights = [
            ensure_numpy_type(layers[node.input[1]]),
            ensure_numpy_type(layers[node.input[2]])
        ]
    else:
        raise AttributeError('Unknown arguments for batch norm')

    eps = params['epsilon'] if 'epsilon' in params else 1e-05  # default epsilon
    momentum = params['momentum'] if 'momentum' in params else 0.9  # default momentum

    if len(weights) == 2:
        logger.debug('Batch normalization without running averages')
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=weights,
            name=node_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=weights,
            name=node_name
        )

    layers[node_name] = bn(input_0)


def convert_instancenorm(node, params, layers, node_name):
    """
    Convert InstanceNorm2d layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:instancenorm2d')

    input_0 = ensure_tf_type(layers[node.input[0]])

    if len(node.input) == 3:
        gamma = ensure_numpy_type(layers[node.input[1]])
        beta = ensure_numpy_type(layers[node.input[2]])
    else:
        raise AttributeError('Unknown arguments for batch norm')

    def target_layer(x, epsilon=params['epsilon'], gamma=gamma, beta=beta):
        import tensorflow as tf
        from keras import backend as K
        data_format = 'NCHW' if K.image_data_format() == 'channels_first' else 'NHWC'

        layer = tf.contrib.layers.instance_norm(
            x,
            param_initializers={'beta': tf.constant_initializer(beta), 'gamma': tf.constant_initializer(gamma)},
            epsilon=epsilon, data_format=data_format,
            trainable=False
        )
        return layer

    lambda_layer = keras.layers.Lambda(target_layer, name=node_name)
    layers[node_name] = lambda_layer(input_0)


def convert_dropout(node, params, layers, node_name):
    """
    Convert Dropout layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:dropout')

    input_0 = ensure_tf_type(layers[node.input[0]])

    ratio = params['ratio'] if 'ratio' in params else 0.0
    lambda_layer = keras.layers.Dropout(ratio, name=node_name)
    layers[node_name] = lambda_layer(input_0)
