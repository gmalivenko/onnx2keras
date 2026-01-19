from .upsampling_layers import convert_upsample


def convert_alias_with_name(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    layers[node_name] = layers[node.input[0]]


def convert_resize_nearest(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    params['scales'] = (params['height_scale'], params['width_scale'])
    params['mode'] = 'nearest'.encode('utf-8')
    convert_upsample(node, params, layers, lambda_func, node_name, keras_name)

