from tensorflow import keras
import logging
from .utils import is_numpy, ensure_tf_type


def convert_elementwise_div(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert element-wise division
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.div')

    if len(node.input) != 2:
        raise AttributeError('Number of inputs is not equal 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    try:
        logger.debug('Divide numpy arrays.')
        layers[node_name] = layers[node.input[0]] / layers[node.input[1]]
    except (IndexError, ValueError):
        logger.debug('Convert inputs to Keras/TF layers if needed.')

        def target_layer(x):
            import tensorflow as tf
            layer = tf.divide(
                x[0],
                x[1]
            )
            return layer

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer([input_0, input_1])
        lambda_func[keras_name] = target_layer


def convert_elementwise_add(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert element-wise add.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.add')

    if len(node.input) != 2:
        raise AttributeError('Number of inputs is not equal 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    input_0_is_np = is_numpy(input_0)
    input_1_is_np = is_numpy(input_1)
    try:
        if not input_0_is_np and not input_1_is_np:
            add = keras.layers.Add(name=keras_name)
            layers[node_name] = add([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Add. Fallback to TF lambda.')
        layers[node_name] = input_0 + input_1


def convert_elementwise_mul(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert element-wise mul.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.mul')

    if len(node.input) != 2:
        raise AttributeError('Number of inputs is not equal 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    input_0_is_np = is_numpy(input_0)
    input_1_is_np = is_numpy(input_1)

    try:
        if not input_0_is_np and not input_1_is_np:
            mul = keras.layers.Multiply(name=keras_name)
            layers[node_name] = mul([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Multiply. Fallback to TF lambda.')
        layers[node_name] = input_0 * input_1


def convert_elementwise_sub(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert element-wise sub.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.sub')

    if len(node.input) != 2:
        raise AttributeError('Number of inputs is not equal 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    input_0_is_np = is_numpy(input_0)
    input_1_is_np = is_numpy(input_1)

    try:
        if not input_0_is_np and not input_1_is_np:
            sub = keras.layers.Subtract(name=keras_name)
            layers[node_name] = sub([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Subtract. Fallback to TF lambda.')
        layers[node_name] = input_0 - input_1


def convert_min(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Min layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) < 2:
        assert AttributeError('Less than 2 inputs for min layer.')

    inputs = list()
    for i, inp in enumerate(node.input):
        input_ = ensure_tf_type(layers[inp], layers[list(layers)[0]], name="%s_const%i" % (keras_name, i + 1))
        inputs.append(input_)
    layers[node_name] = keras.layers.Minimum(name=keras_name)(inputs)


def convert_max(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Max layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) < 2:
        assert AttributeError('Less than 2 inputs for max layer.')

    inputs = list()
    for i, inp in enumerate(node.input):
        input_ = ensure_tf_type(layers[inp], layers[list(layers)[0]], name="%s_const%i" % (keras_name, i + 1))
        inputs.append(input_)
    layers[node_name] = keras.layers.Maximum(name=keras_name)(inputs)


def convert_mean(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Mean layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    :TODO: Test if this supports multidirectional (i.e., Numpy-style) broadcasting as required
    """
    if len(node.input) < 2:
        assert AttributeError('Less than 2 inputs for mean layer.')

    inputs = list()
    for i, inp in enumerate(node.input):
        input_ = ensure_tf_type(layers[inp], layers[list(layers)[0]], name="%s_const%i" % (keras_name, i + 1))
        inputs.append(input_)
    layers[node_name] = keras.layers.Average(name=keras_name)(inputs)
