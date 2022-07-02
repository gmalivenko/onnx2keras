from tensorflow import keras
import logging
from .utils import ensure_tf_type


def convert_maxpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert MaxPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.maxpool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]
    pad = 'valid'

    if all([shape % 2 == 1 for shape in kernel_shape]) and \
       all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
       all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = keras_name + '_pad'
        if len(kernel_shape) == 2:
            padding = None

            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))

            if padding is not None:
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=padding,
                    name=padding_name
                )
                layers[padding_name] = input_0 = padding_layer(input_0)
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling = keras.layers.MaxPooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.MaxPooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )

    layers[node_name] = pooling(input_0)


def convert_avgpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert AvgPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.avgpool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]
    pad = 'valid'

    if all([shape % 2 == 1 for shape in kernel_shape]) and \
       all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
       all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = keras_name + '_pad'
        if len(kernel_shape) == 2:
            padding_layer = keras.layers.ZeroPadding2D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        layers[padding_name] = input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling = keras.layers.AveragePooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.AveragePooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=keras_name,
            data_format='channels_first'
        )
    layers[node_name] = pooling(input_0)


def convert_global_avg_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.global_avg_pool')

    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    try:
        global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', keepdims=True, name=keras_name)
        output_0 = global_pool(input_0)
    except TypeError:
        # in TensorFlow < 2.6 keyword argument `keepdims` not understood: raises TypeError: ('Keyword argument not understood:', 'keepdims')
        # reproduce behavior of keepdims=True by adding extra layer
        logger.debug("Argument `keepdims` not understood. Add Reshape layer to expand dims.")

        global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=keras_name)
        output_0 = global_pool(input_0)
        reshape = keras.layers.Reshape(output_0.shape[1:] + (1, 1), name=keras_name + '_expand')
        output_0 = reshape(output_0)

    layers[node_name] = output_0
