import keras.layers
import logging
from .utils import ensure_tf_type


def convert_maxpool(node, params, layers, node_name):
    """
    Convert MaxPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:maxpool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])

    height, width = params['kernel_shape']
    stride_height, stride_width = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0]
    padding_h, padding_w, _, _ = pads

    pad = 'valid' 

    if height % 2 == 1 and width % 2 == 1 and \
            height // 2 == padding_h and width // 2 == padding_w and \
            stride_height == 1 and stride_width == 1:
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = node_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = input_0 = padding_layer(input_0)

    pooling = keras.layers.MaxPooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=pad,
        name=node_name,
        data_format='channels_first'
    )

    layers[node_name] = pooling(input_0)


def convert_avgpool(node, params, layers, node_name):
    """
    Convert AvgPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:avgpool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])

    height, width = params['kernel_shape']
    stride_height, stride_width = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0]
    padding_h, padding_w, _, _ = pads

    pad = 'valid'

    if height % 2 == 1 and width % 2 == 1 and \
            height // 2 == padding_h and width // 2 == padding_w and \
            stride_height == 1 and stride_width == 1:
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = node_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = input_0 = padding_layer(input_0)

    pooling = keras.layers.AveragePooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=pad,
        name=node_name,
        data_format='channels_first'
    )

    layers[node_name] = pooling(input_0)


def convert_global_avg_pool(node, params, layers, node_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:global_avg_pool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])

    global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=node_name)
    input_0 = global_pool(input_0)

    def target_layer(x):
        import keras
        return keras.backend.expand_dims(x)

    logger.debug('Now expand dimensions twice.')
    lambda_layer1 = keras.layers.Lambda(target_layer, name=node_name + '_EXPAND1')
    lambda_layer2 = keras.layers.Lambda(target_layer, name=node_name + '_EXPAND2')
    input_0 = lambda_layer1(input_0)  # double expand dims
    layers[node_name] = lambda_layer2(input_0)