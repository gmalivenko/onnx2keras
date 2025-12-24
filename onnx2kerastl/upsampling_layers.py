import keras
import numpy as np
import logging


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

    interpolation_mode = params['mode'].decode('utf-8')
    if interpolation_mode == 'nearest':
        interpolation = "nearest"
    elif interpolation_mode in ['bilinear', 'linear']:
        interpolation = "bilinear"
    elif interpolation_mode in "cubic":
        interpolation = "bicubic"
    else:
        logger.error(f'Cannot convert upsampling. interpolation mode: {interpolation_mode} is not supported')
        raise AssertionError(f'Cannot convert upsampling. interpolation mode: {interpolation_mode} is not supported')

    upsampling = keras.layers.UpSampling2D(size=scale, name=f"{params['cleaned_name']}_upsample_2d",
                                           interpolation=interpolation)

    layers[node_name] = upsampling(layers[node.input[0]])
