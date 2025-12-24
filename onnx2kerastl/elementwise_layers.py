import numpy as np
import keras
import logging

from .utils import is_numpy, ensure_tf_type
from .tfops_funcs import tf_tensor_scatter_nd_update, tf_maximum, tf_minimum, tf_cast, tf_expand_dims, tf_repeat,\
    tf_equal, tf_where, tf_round, tf_sign, tf_abs, tf_math_mod, tf_bitwise_left_shift, tf_bitwise_right_shift,\
    tf_logical_not, tf_add
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
            div = tf_cast(div, input_0.dtype, tf_name=f"{params['cleaned_name']}_div_cast")
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

        lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_div")
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
        raise AttributeError('Number of inputs is not equal to 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    input_0_is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_constant = is_numpy(input_1) or isinstance(input_1, EagerTensor)

    try:
        if not input_0_is_constant and not input_1_is_constant:
            # Both inputs are variables
            if len(input_0.shape) != len(input_1.shape):
                # Use TensorFlow add to handle shape differences
                layers[node_name] = tf_add(input_0, input_1, tf_name=f"{params['cleaned_name']}_add")
            else:
                # Use Keras Add layer
                layers[node_name] = keras.layers.Add(name=f"{params['cleaned_name']}_add")([input_0, input_1])
        else:
            raise ValueError('Operands are different.')
    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Add. Fallback to Lambda layer.')

        if input_0_is_constant and not input_1_is_constant:
            # input_0 is constant, input_1 is variable
            constant_value = np.asarray(tf.cast(input_0, dtype=input_1.dtype))
            variable_input = input_1

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x + const_val,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x + constant_value,
                    name=params['cleaned_name']
                )(variable_input)

        elif not input_0_is_constant and input_1_is_constant:
            # input_0 is variable, input_1 is constant
            constant_value = np.asarray(tf.cast(input_1, dtype=input_0.dtype))
            variable_input = input_0

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x + const_val,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x + constant_value,
                    name=params['cleaned_name']
                )(variable_input)
        else:
            # Both inputs are constants; compute the result now
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
        raise AttributeError('Number of inputs is not equal to 2 for element-wise layer')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    input_0_is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_constant = is_numpy(input_1) or isinstance(input_1, EagerTensor)

    try:
        if not input_0_is_constant and not input_1_is_constant:
            mul = keras.layers.Multiply(name=f"{params['cleaned_name']}_mul")
            layers[node_name] = mul([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Multiply. Fallback to Lambda layer.')

        if input_0_is_constant and not input_1_is_constant:
            # input_0 is constant, input_1 is variable
            constant_value = np.asarray(tf.cast(input_0, dtype=input_1.dtype))
            variable_input = input_1

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x * const_val,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Cannot avoid embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x * constant_value,
                    name=params['cleaned_name']
                )(variable_input)

        elif not input_0_is_constant and input_1_is_constant:
            # input_0 is variable, input_1 is constant
            constant_value = np.asarray(tf.cast(input_1, dtype=input_0.dtype))
            variable_input = input_0

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x * const_val,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Cannot avoid embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x * constant_value,
                    name=params['cleaned_name']
                )(variable_input)
        else:
            # Both inputs are constants; compute the result now
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
    
    input_0_is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    input_1_is_constant = is_numpy(input_1) or isinstance(input_1, EagerTensor)

    try:
        if not input_0_is_constant and not input_1_is_constant:
            sub = keras.layers.Subtract(name=f"{params['cleaned_name']}_sub")
            layers[node_name] = sub([input_0, input_1])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.warning('Failed to use keras.layers.Subtract. Fallback to Lambda layer.')

        if input_0_is_constant and not input_1_is_constant:
            # input_0 is constant, input_1 is variable: constant - variable
            constant_value = np.asarray(tf.cast(input_0, dtype=input_1.dtype))
            variable_input = input_1

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: const_val - x,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Cannot avoid embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: constant_value - x,
                    name=params['cleaned_name']
                )(variable_input)

        elif not input_0_is_constant and input_1_is_constant:
            # input_0 is variable, input_1 is constant: variable - constant
            constant_value = np.asarray(tf.cast(input_1, dtype=input_0.dtype))
            variable_input = input_0

            if np.all(constant_value == constant_value.flat[0]):
                # Constant tensor has the same value throughout
                const_val = constant_value.flat[0]
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x - const_val,
                    name=params['cleaned_name']
                )(variable_input)
            else:
                # Cannot avoid embedding the constant tensor
                layers[node_name] = keras.layers.Lambda(
                    lambda x: x - constant_value,
                    name=params['cleaned_name']
                )(variable_input)
        else:
            # Both inputs are constants; compute the result now
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
        raise AttributeError('Less than 2 inputs for min layer.')

    inputs = [
        ensure_tf_type(layers[inp], name="%s_const%i" % (keras_name, i + 1))
        for i, inp in enumerate(node.input)
    ]

    # Broadcast the inputs to the same shape
    input1, input2 = inputs
    # Applying the minimum operation
    min_output = tf_minimum(input1, input2, tf_name=f"{params['cleaned_name']}_min")

    layers[node_name] = min_output


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
        raise AttributeError('Less than 2 inputs for max layer.')

    inputs = [
        ensure_tf_type(layers[inp], name="%s_const%i" % (keras_name, i + 1))
        for i, inp in enumerate(node.input)
    ]

    # Broadcast the inputs to the same shape
    input1, input2 = inputs
    # Applying the maximum operation
    max_output = tf_maximum(input1, input2, tf_name=f"{params['cleaned_name']}_maximum")

    layers[node_name] = max_output


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
    layers[node_name] = keras.layers.Average(name=f"{params['cleaned_name']}_mean")(inputs)


def convert_equal(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_equal(layers[node.input[0]], layers[node.input[1]],
                                 tf_name=f"{params['cleaned_name']}_equal")


def convert_where(node, params, layers, lambda_func, node_name, keras_name):
    if layers[node.input[0]].dtype != tf.bool:
        casted = tf_cast(layers[node.input[0]], tf.bool, tf_name=f"{params['cleaned_name']}_cast")
    else:
        casted = layers[node.input[0]]
    if layers[node.input[1]].dtype == np.int64 and is_numpy(layers[node.input[1]]):
        # serialization doesn't work well for first argument if it is np array of type int64
        layers[node_name] = tf_where(tf_logical_not(casted,
                                                    tf_name=f"{params['cleaned_name']}_not"
                                                    ),
                                     layers[node.input[2]],
                                     layers[node.input[1]],
                                     tf_name=f"{params['cleaned_name']}_where_1")
    else:
        try:
            layers[node_name] = tf_where(casted, layers[node.input[1]], layers[node.input[2]],
                                         tf_name=f"{params['cleaned_name']}_where_2")
        except Exception as e:
            print(1)


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
    layers[node_name] = tf_tensor_scatter_nd_update(data, indices, updates,
                                                    tf_name=f"{params['cleaned_name']}_scatter_nd")


def convert_round(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_round(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_round")


def convert_mod(node, params, layers, lambda_func, node_name, keras_name):
    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]
    if params.get('fmod') == 1:
        sign = tf_sign(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_mod_sign")
        input_0 = tf_abs(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_abs_0")
        input_1 = tf_abs(layers[node.input[1]], tf_name=f"{params['cleaned_name']}_abs_1")
        layers[node_name] = tf_math_mod(input_0, input_1, tf_name=f"{params['cleaned_name']}_mod") * sign
    else:
        layers[node_name] = tf_math_mod(input_0, input_1, tf_name=f"{params['cleaned_name']}_mod")


def convert_bitshift(node, params, layers, lambda_func, node_name, keras_name):
    direction = params.get("direction").decode()
    if direction == "LEFT":
        shifter_pointer = tf_bitwise_left_shift
    elif direction == "RIGHT":
        shifter_pointer = tf_bitwise_right_shift
    else:
        raise AttributeError("Onnx2Kerras cannot convert the BitShift operator"
                             " since the 'direction' attribute was missing")
    layers[node_name] = shifter_pointer(tf_cast(layers[node.input[0]], tf.uint64,
                                                tf_name=f"{params['cleaned_name']}_bitshift_cast_0"),
                                        tf_cast(layers[node.input[1]], tf.uint64,
                                                tf_name=f"{params['cleaned_name']}_bitshift_cast_1"),
                                        tf_name=f"{params['cleaned_name']}_bitshift")
