import numpy as np
import tensorflow as tf


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
    value = params.get('value')
    if value is None:
        raise NotImplementedError("ConstantOfShape should have a value param")
    layers[node_name] = np.ones(layers[node.input[0]], dtype=np.int32)*params['value']


def convert_one_hot(node, params, layers, lambda_func, node_name, keras_name):
    axis = params.get('axis', -1)
    layers[node_name] = tf.one_hot(indices=tf.cast(layers[node.input[0]],
                                                   tf.int64),
                                   depth=int(layers[node.input[1]]), off_value=layers[node.input[2]][0],
                                   on_value=layers[node.input[2]][1],
                                   axis=axis
                                   )

