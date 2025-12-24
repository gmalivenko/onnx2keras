import numpy as np
import tensorflow as tf
from .utils import is_numpy
from .tfops_funcs import tf_cast, tf_one_hot
from keras import backend as K

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

    input_0 = layers[node.input[0]]

    if not is_numpy(input_0) and not isinstance(input_0, list) and K.is_keras_tensor(input_0):
        # Boolean case
        if value.dtype == np.bool_:
            layers[node_name] = tf.fill(layers[node.input[0]], tf.constant(value.item(), dtype=tf.bool))
        else:
            # Non-boolean case
            layers[node_name] = tf.ones(layers[node.input[0]], dtype=tf.as_dtype(value.dtype)) * value
    else:
        # Handle numpy inputs or non-Keras tensors
        if value.dtype == np.bool_:
            layers[node_name] = np.full(layers[node.input[0]], value.item(), dtype=bool)
        else:
            layers[node_name] = np.ones(layers[node.input[0]], dtype=value.dtype) * value



def convert_one_hot(node, params, layers, lambda_func, node_name, keras_name):
    axis = params.get('axis', -1)
    layers[node_name] = tf_one_hot(indices=tf_cast(layers[node.input[0]],
                                                   tf.int64,
                                                   tf_name=f"{params['cleaned_name']}_onehot_cast"),
                                   depth=int(layers[node.input[1]]),
                                   off_value=layers[node.input[2]][0],
                                   on_value=layers[node.input[2]][1],
                                   axis=axis,
                                   tf_name=f"{params['cleaned_name']}_onehot"
                                   )

