from tensorflow import keras
import logging
from .utils import ensure_tf_type, ensure_numpy_type


def convert_conv(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:conv')

    if len(node.input) == 3:
        logger.debug('Conv with bias')
        # Has bias
        has_bias = True
        W = ensure_numpy_type(layers[node.input[1]])
        bias = ensure_numpy_type(layers[node.input[2]])

    elif len(node.input) == 2:
        logger.debug('Conv without bias')
        has_bias = False
        W = ensure_numpy_type(layers[node.input[1]])
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    n_groups = params['group'] if 'group' in params else 1
    dilation = params['dilations'][0] if 'dilations' in params else 1
    pads = params['pads'] if 'pads' in params else [0, 0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1, 1]

    if len(W.shape) == 5:  # 3D conv
        logger.debug('3D convolution')
        if pads[0] > 0 or pads[1] > 0 or pads[2] > 0:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = keras_name + '_pad'
            padding_layer = keras.layers.ZeroPadding3D(
                padding=(pads[0], pads[1], pads[2]),
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
        out_channels, channels_per_group, dimension, height, width = W.shape
        W = W.transpose(2, 3, 4, 1, 0)

        if n_groups != 1:
            raise NotImplementedError("Not Implemented")
        else:
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.Conv3D(
                filters=out_channels,
                kernel_size=(dimension, height, width),
                strides=(strides[0], strides[1], strides[2]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=keras_name
            )
            layers[node_name] = conv(input_0)

    elif len(W.shape) == 4:  # 2D conv
        logger.debug('2D convolution')

        padding = None
        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            padding = (pads[0], pads[1])
        elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
            padding = ((pads[0], pads[2]), (pads[1], pads[3]))

        if padding:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = keras_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=padding,
                name=padding_name,
                data_format='channels_first'
            )
            layers[padding_name] = input_0 = padding_layer(input_0)

        W = W.transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        in_channels = channels_per_group * n_groups

        if n_groups == in_channels and n_groups != 1:
            logger.debug('Number of groups is equal to input channels, use DepthWise convolution')
            W = W.transpose(0, 1, 3, 2)
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.DepthwiseConv2D(
                kernel_size=(height, width),
                strides=(strides[0], strides[1]),
                padding='valid',
                use_bias=has_bias,
                activation=None,
                depth_multiplier=1,
                weights=weights,
                dilation_rate=dilation,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=keras_name
            )
            layers[node_name] = conv(input_0)

        elif n_groups != 1:
            logger.debug('Number of groups more than 1, but less than number of in_channel, use group convolution')

            # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
            def target_layer(x, groups=n_groups, stride_y=strides[0], stride_x=strides[1]):
                import tensorflow as tf
                from tensorflow.keras import backend as K
                data_format = 'NCHW' if K.image_data_format() == 'channels_first' else 'NHWC'

                if data_format == 'NCHW':
                    x = tf.transpose(x, [0, 2, 3, 1])

                def convolve_lambda_biased(i, k, b):
                    import tensorflow as tf
                    conv = tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], dilations=[1, dilation, dilation, 1], padding='VALID', data_format='NHWC')
                    return tf.nn.bias_add(conv, b,  data_format='NHWC')

                def convolve_lambda(i, k):
                    import tensorflow as tf
                    return tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], dilations=[1, dilation, dilation, 1], padding='VALID', data_format='NHWC')

                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
                if has_bias:
                    bias_groups = tf.split(axis=0, num_or_size_splits=groups, value=bias)
                    output_groups = [convolve_lambda_biased(i, k, b) for i, k, b in
                                     zip(input_groups, weight_groups, bias_groups)]
                else:
                    output_groups = [convolve_lambda(i, k) for i, k in zip(input_groups, weight_groups)]

                layer = tf.concat(axis=3, values=output_groups)
                if data_format == 'NCHW':
                    layer = tf.transpose(layer, [0, 3, 1, 2])

                return layer

            lambda_layer = keras.layers.Lambda(target_layer)
            layers[node_name] = lambda_layer(input_0)

        else:
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(strides[0], strides[1]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation,
                bias_initializer='zeros', kernel_initializer='zeros',
                name=keras_name
            )

            layers[node_name] = conv(input_0)
    else:
        # 1D conv
        W = W.transpose(2, 1, 0)
        width, channels, n_filters = W.shape
        print(width, channels, n_filters, has_bias)

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        def target_layer(x, w=weights, stride=strides[0]):
            import tensorflow as tf
            w = tf.convert_to_tensor(w[0])
            x = tf.transpose(x, [0, 2, 1])
            x = tf.nn.conv1d(x, w, stride=stride, padding='SAME', data_format='NWC')
            return tf.transpose(x, [0, 2, 1])

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        lambda_layer[keras_name] = target_layer
        layers[node_name] = lambda_layer(input_0)

        # padding_name = keras_name + '_pad'
        # padding_layer = keras.layers.ZeroPadding1D(
        #     padding=(pads[0]),
        #     name=padding_name
        # )
        # print(input_0)
        # layers[node_name] = padding_layer(input_0)
        # input_0.set_shape(input_0._keras_shape)
        # print(input_0._keras_shape)
        # print(input_0, n_filters, width)
        # conv = keras.layers.Conv1D(
        #     filters=n_filters,
        #     kernel_size=width,
        #     strides=strides[0],
        #     padding='valid',
        #     weights=weights,
        #     use_bias=has_bias,
        #     activation=None,
        #     dilation_rate=dilation,
        #     name=keras_name
        # )
        # layers[node_name] = conv(input_0)


def convert_convtranspose(node, params, layers,
                          lambda_func, node_name, keras_name):
    """
    Convert transposed convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:convtranpose')

    if len(node.input) == 3:
        logger.debug('ConvTranspose with bias')
        # Has bias
        has_bias = True
        W = ensure_numpy_type(layers[node.input[1]])
        bias = ensure_numpy_type(layers[node.input[2]])

    elif len(node.input) == 2:
        logger.debug('ConvTranspose without bias')
        has_bias = False
        W = ensure_numpy_type(layers[node.input[1]])
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    n_groups = params['group'] if 'group' in params else 1
    dilation = params['dilations'][0] if 'dilations' in params else 1
    pads = params['pads'] if 'pads' in params else [0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1]

    if len(W.shape) == 5:  # 3D conv
        raise NotImplementedError('Not implemented')

    elif len(W.shape) == 4:  # 2D conv
        W = W.transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if n_groups > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with groups != 1')

        if dilation > 1:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=strides,
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            bias_initializer='zeros', kernel_initializer='zeros',
            name=keras_name
        )

        if 'output_shape' in params and 'pads' not in params:
            logger.debug('!!!!! Paddings will be calculated automatically !!!!!')
            pads = [strides[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][0],
                    strides[1] * (int(input_0.shape[3]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][1]]

        layers[node_name] = input_0 = conv(input_0)

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        # input_0.set_shape(input_0.shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if pads[0] > 0:
            logger.debug('Add cropping layer for output padding')
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=keras_name + '_crop'
            )
            layers[node_name] = crop(input_0)
    else:
        raise AttributeError('Layer is not supported for now')
