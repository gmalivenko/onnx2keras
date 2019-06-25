import keras.layers
from .utils import ensure_tf_type


def convert_padding(node, params, layers, node_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    # It's binary by-default
    params['mode'] = params['mode'].decode('ascii')
    input_0 = ensure_tf_type(layers[node.input[0]])

    if params['mode'] == 'constant':
        # raise AssertionError('Cannot convert non-constant padding')

        if params['value'] != 0.0:
            raise AssertionError('Cannot convert non-zero padding')

        # Magic ordering
        padding_layer = keras.layers.ZeroPadding2D(
            padding=((params['pads'][2], params['pads'][6]), (params['pads'][3], params['pads'][7])),
            name=node_name
        )

        layers[node_name] = padding_layer(input_0)
    elif params['mode'] == 'reflect':

        def target_layer(x, pads=params['pads']):
            import tensorflow as tf
            layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'REFLECT')
            return layer

        lambda_layer = keras.layers.Lambda(target_layer, name=node_name)
        layers[node_name] = lambda_layer(input_0)
    elif params['mode'] == 'edge':

        def target_layer(x, pads=params['pads']):
            import tensorflow as tf
            layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'SYMMETRIC')
            return layer

        lambda_layer = keras.layers.Lambda(target_layer, name=node_name)
        layers[node_name] = lambda_layer(input_0)

    else:
        raise AttributeError('Unknown padding')
