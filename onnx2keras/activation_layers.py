import keras.layers


def convert_relu(node, params, layers, node_name):
    """
    Convert ReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_name = node.input[0]

    relu = keras.layers.Activation('relu', name=node_name)
    layers[node_name] = relu(layers[input_name])


def convert_lrelu(node, params, layers, node_name):
    """
    Convert LeakyReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_name = node.input[0]

    leakyrelu = \
        keras.layers.LeakyReLU(alpha=params['alpha'], name=node_name)
    layers[node_name] = leakyrelu(layers[input_name])


def convert_sigmoid(node, params, layers, node_name):
    """
    Convert Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_name = node.input[0]

    sigmoid = keras.layers.Activation('sigmoid', name=node_name)
    layers[node_name] = sigmoid(layers[input_name])


def convert_tanh(node, params, layers, node_name):
    """
    Convert Tanh activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_name = node.input[0]

    tanh = keras.layers.Activation('tanh', name=node_name)
    layers[node_name] = tanh(layers[input_name])


def convert_selu(node, params, layers, node_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_name = node.input[0]

    selu = keras.layers.Activation('selu', name=node_name)
    layers[node_name] = selu(layers[input_name])
