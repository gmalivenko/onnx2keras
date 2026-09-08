from tensorflow import keras
import numpy as np
import logging
from .utils import ensure_numpy_type


def convert_upsample(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert upsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.upsample')
    logger.warning('!!! EXPERIMENTAL SUPPORT (upsample) !!!')

    if "scales" in params:
        # for opset version - 7
        if len(node.input) != 1:
            raise AttributeError('Unsupported number of inputs')
        scale = np.uint8(params['scales'][-2:])
    else:
        # for opset version - 9+
        # Upsample since opset version 9 uses input[1] as 'scales' instead of attributes.
        scale = np.uint8(layers[node.input[1]][-2:])

    if params['mode'].decode('utf-8') != 'nearest':
        logger.error('Cannot convert non-nearest upsampling.')
        raise AssertionError('Cannot convert non-nearest upsampling')

    upsampling = keras.layers.UpSampling2D(
        size=scale, name=keras_name
    )

    layers[node_name] = upsampling(layers[node.input[0]])


def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert resize. Since opset 10 `Resize` replaces the deprecated `Upsample` operation.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.resize')
    logger.warning('!!! EXPERIMENTAL SUPPORT (resize) !!!')

    mode = params.get('mode', b'nearest')
    if isinstance(mode, bytes):
        mode = mode.decode('utf-8')

    if mode != 'nearest':
        logger.error('Cannot convert non-nearest resize.')
        raise AssertionError('Cannot convert non-nearest resize (mode: %s)' % mode)

    input_0 = layers[node.input[0]]

    # Inputs are (X, roi, scales, sizes); `roi` and one of `scales` / `sizes` are optional.
    scales = None
    if len(node.input) > 2 and node.input[2]:
        scales = ensure_numpy_type(layers[node.input[2]])

    if scales is None or len(scales) == 0:
        if len(node.input) <= 3 or not node.input[3]:
            raise AttributeError('Resize without `scales` or `sizes` is not supported')

        sizes = ensure_numpy_type(layers[node.input[3]])
        input_shape = input_0.shape
        if any(input_shape[i] is None for i in (-2, -1)):
            raise AttributeError('Cannot convert Resize with `sizes` for a dynamic input shape')
        scales = np.array([sizes[-2] / int(input_shape[-2]), sizes[-1] / int(input_shape[-1])])

    scale = np.array(scales[-2:])
    if not np.array_equal(scale, scale.astype(np.int64)):
        raise AttributeError('Cannot convert Resize with non-integer scales: %s' % scale)

    upsampling = keras.layers.UpSampling2D(
        size=scale.astype(np.int64), name=keras_name
    )

    layers[node_name] = upsampling(input_0)
