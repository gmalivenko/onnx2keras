import numpy as np
import keras
import logging
from .utils import is_numpy, ensure_tf_type
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


def _is_integer_type(dtype) -> bool:
    return dtype in (tf.int32, tf.int64, tf.int16, tf.int8, np.int32, np.int64, np.int16, np.int8)


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
        div = input_0 / input_1
        if _is_integer_type(input_0.dtype) and _is_integer_type(input_1.dtype):
            div = tf.cast(div, input_0.dtype)
        if hasattr(div, 'numpy'):
            div = div.numpy()
        layers[node_name] = div

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
    input_0_is_non_keras = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_non_keras = is_numpy(input_1) or isinstance(input_1, EagerTensor)
    try:
        if not input_0_is_non_keras and not input_1_is_non_keras:
            to_add = input_1
            if input_0.shape != input_1.shape and input_0.shape[:-1] == input_1.shape:
                to_add = tf.repeat(tf.expand_dims(input_1, axis=-1), input_0.shape[-1], axis=-1)

            layers[node_name] = keras.layers.Add(name=keras_name)([input_0, to_add])
        else:
            raise ValueError('Operands are different.')
    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Add. Fallback to TF lambda.')
        if input_0_is_non_keras:
            layers[node_name] = input_1 + input_0
        else:
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
    input_0_is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_constant = is_numpy(input_1) or isinstance(input_1, EagerTensor)
    try:
        if not input_0_is_constant and not input_1_is_constant:
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
    input_0_is_np = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_np = is_numpy(input_1) or isinstance(input_1, EagerTensor)

    try:
        if not input_0_is_np and not input_1_is_np:
            sub = keras.layers.Subtract(name=keras_name)
            layers[node_name] = sub([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Subtract. Fallback to TF lambda.')
        if input_0_is_np and not input_1_is_np:  # constant - tensor does not parse well
            layers[node_name] = - (input_1 - input_0)
        else:
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


def convert_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.equal(layers[node.input[0]], layers[node.input[1]])


def convert_where(node, params, layers, lambda_func, node_name, keras_name):
    if layers[node.input[0]].dtype != tf.bool:
        casted = tf.cast(layers[node.input[0]], tf.bool)
    else:
        casted = layers[node.input[0]]
    if layers[node.input[1]].dtype == np.int64 and is_numpy(layers[node.input[1]]):
        #serialization doesn't work well for first argument if it is np array of type int64
        layers[node_name] = tf.where(tf.logical_not(casted), layers[node.input[2]], layers[node.input[1]])
    else:
        layers[node_name] = tf.where(casted, layers[node.input[1]], layers[node.input[2]])


def convert_scatter_nd(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ScatterND layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    :TODO: Test if this supports multidirectional (i.e., Numpy-style) broadcasting as required
    """
    if len(node.input) < 3:
        assert AttributeError('Less than 3 inputs')

    data = ensure_tf_type(layers[node.input[0]])
    indices = ensure_tf_type(layers[node.input[1]])
    updates = ensure_tf_type(layers[node.input[2]])
    layers[node_name] = tf.tensor_scatter_nd_update(data, indices, updates)


def convert_round(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf.round(layers[node.input[0]])


def convert_mod(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    if params.get('fmod') == 1:
        sign = tf.sign(layers[node.input[0]])
        input_0 = tf.abs(layers[node.input[0]])
        input_1 = tf.abs(layers[node.input[1]])
        layers[node_name] = tf.math.mod(input_0, input_1)*sign
    else:
        layers[node_name] = tf.math.mod(input_0, input_1)


def convert_bitshift(node, params, layers, lambda_func, node_name, keras_name):
    direction = params.get("direction").decode()
    if direction == "LEFT":
        shifter_pointer = tf.bitwise.left_shift
    elif direction == "RIGHT":
        shifter_pointer = tf.bitwise.right_shift
    else:
        raise AttributeError("Onnx2Kerras cannot convert the BitShift operator"
                             " since the 'direction' attribute was missing")
    layers[node_name] = shifter_pointer(tf.cast(layers[node.input[0]], tf.uint64), tf.cast(layers[node.input[1]], tf.uint64))

