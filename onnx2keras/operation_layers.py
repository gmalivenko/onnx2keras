from tensorflow import keras
from tensorflow.keras import backend as K
import logging
from .utils import is_numpy, ensure_tf_type, ensure_numpy_type
import numpy as np

# Handle python 2.7 import error
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def convert_clip(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert clip layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:clip')
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for clip layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    if params['min'] == 0:
        logger.debug("Using ReLU({0}) instead of clip".format(params['max']))
        layer = keras.layers.ReLU(max_value=params['max'], name=keras_name)
    else:
        def target_layer(x, vmin=params['min'], vmax=params['max']):
            import tensorflow as tf
            return tf.clip_by_value(x, vmin, vmax)
        layer = keras.layers.Lambda(target_layer, name=keras_name)
        lambda_func[keras_name] = target_layer

    layers[node_name] = layer(input_0)


def convert_log(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Log layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for log layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        import tensorflow.keras.backend as K
        return K.log(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_exp(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Exp layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for log layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        import tensorflow.keras.backend as K
        return K.exp(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_reduce_sum(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce sum.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce sum layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    axis = params['axes']

    def target_layer(x, axis=axis):
        import tensorflow.keras.backend as K
        return K.sum(x, keepdims=True, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


def convert_reduce_mean(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reduce mean.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce mean layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, axis=params['axes'], keepdims=params['keepdims']):
        import tensorflow.keras.backend as K
        return K.mean(x, keepdims=(keepdims == 1), axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)
    lambda_func[keras_name] = target_layer


def convert_pow(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Pow layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 2:
        assert AttributeError('More than 2 inputs for pow layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    power = ensure_numpy_type(layers[node.input[1]])

    def target_layer(x, a=power):
        import tensorflow.keras.backend as K
        return K.pow(x, a)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_sqrt(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Sqrt layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for sqrt layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        import tensorflow.keras.backend as K
        return K.sqrt(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_split(node, params, layers, lambda_func, node_name, keras_names):
    """
    Convert Split layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for split layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_names[0])
    splits = params["split"]
    axis = params.get("axis", 0)
    if not isinstance(splits, Iterable):
        # This might not work if `split` is a tensor.
        chunk_size = K.int_size(input_0)[axis] // splits
        splits = (chunk_size,) * splits

    cur = 0
    for i, split in enumerate(splits):
        node_name = params['_outputs'][i]

        def target_layer(x, axis=axis, start_i=cur, end_i=cur+split):
            slices = [slice(None, None)] * len(K.int_shape(x))
            slices[axis] = slice(start_i, end_i)
            return x[tuple(slices)]

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_names[i])
        layers[node_name] = lambda_layer(input_0)
        lambda_func[keras_names[i]] = target_layer
        cur += split


def convert_cast(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Cast layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:cast')

    if len(node.input) != 1:
        assert AttributeError('More than 1 input for cast layer.')

    if is_numpy(layers[node.input[0]]):
        logger.debug('Cast numpy array')

        cast_map = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            9: np.bool,
            10: np.float16,
            11: np.double,
        }

        layers[node_name] = cast_map[params['to']](node.input[0])
    else:
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

        def target_layer(x, dtype=params['to']):
            import tensorflow as tf
            cast_map = {
                1: tf.float32,
                2: tf.uint8,
                3: tf.int8,
                5: tf.int16,
                6: tf.int32,
                7: tf.int64,
                9: tf.bool,
                10: tf.float16,
                11: tf.double,
            }
            return tf.cast(x, cast_map[dtype])

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(input_0)
        lambda_func[keras_name] = target_layer


def convert_floor(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Floor layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for floor layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x):
        # Floor is absent in keras.backend
        import tensorflow as tf
        return tf.floor(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_identity(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Identity layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for itentity layer.')

    layers[node_name] = layers[node.input[0]]


def convert_argmax(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ArgMax layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for argmax layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get("axis", -1)

    def target_layer(x, axis=axis):
        import tensorflow as tf
        return tf.argmax(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer


def convert_reduce_l2(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ReduceL2 layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for reduce_l2 layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get("axes", [-1])

    def target_layer(x, axis=axis):
        import tensorflow as tf
        return tf.norm(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
    lambda_func[keras_name] = target_layer
