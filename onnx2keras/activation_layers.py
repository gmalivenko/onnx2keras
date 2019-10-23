from tensorflow import keras
from .utils import ensure_tf_type


def convert_relu(node, params, layers, node_name, keras_name):
    """
    Convert ReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    relu = keras.layers.Activation('relu', name=keras_name)
    layers[node_name] = relu(input_0)


def convert_lrelu(node, params, layers, node_name, keras_name):
    """
    Convert LeakyReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    leakyrelu = \
        keras.layers.LeakyReLU(alpha=params['alpha'], name=keras_name)
    layers[node_name] = leakyrelu(input_0)


def convert_sigmoid(node, params, layers, node_name, keras_name):
    """
    Convert Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    sigmoid = keras.layers.Activation('sigmoid', name=keras_name)
    layers[node_name] = sigmoid(input_0)


def convert_tanh(node, params, layers, node_name, keras_name):
    """
    Convert Tanh activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    tanh = keras.layers.Activation('tanh', name=keras_name)
    layers[node_name] = tanh(input_0)


def convert_selu(node, params, layers, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    selu = keras.layers.Activation('selu', name=keras_name)
    layers[node_name] = selu(input_0)


def convert_softmax(node, params, layers, node_name, keras_name):
    """
    Convert softmax activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    softmax = keras.layers.Activation('softmax', name=keras_name)
    layers[node_name] = softmax(input_0)
