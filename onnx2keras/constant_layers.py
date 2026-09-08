import numpy as np


def convert_constant(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    layers[node_name] = params['value']


def convert_constant_of_shape(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ConstantOfShape layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    shape = layers[node.input[0]]

    # `value` is a one-element tensor giving both the fill value and the output dtype.
    value = params['value'] if 'value' in params else np.zeros(1, dtype=np.float32)
    value = np.asarray(value)

    layers[node_name] = np.full(tuple(np.int64(shape)), value.ravel()[0], dtype=value.dtype)
