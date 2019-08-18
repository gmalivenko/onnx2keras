import keras.layers
import numpy as np
import logging
from .utils import is_numpy, ensure_tf_type


def convert_transpose(node, params, layers, node_name, keras_name):
    """
    Convert transpose.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:transpose')
    input_name = node.input[0]

    if params['perm'][0] != 0:
        logger.warning('Can\'t permute batch dimension. Result may be wrong.')
        if is_numpy(layers[input_name]):
            logger.warning('Transposing numpy array.')
            layers[node_name] = np.transpose(layers[input_name], axes=params['perm'])
        else:
            raise NotImplementedError('Can\'t modify this type of data')
    else:
        permute = keras.layers.Permute(params['perm'][1:], name=keras_name)
        layers[node_name] = permute(layers[input_name])


def convert_shape(node, params, layers, node_name, keras_name):
    """
    Convert shape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:shape')
    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)
    
    logger.debug('Actual result:')
    logger.debug(np.array(input_0._keras_shape))

    layers[node_name] = np.array(input_0._keras_shape)


def convert_gather(node, params, layers, node_name, keras_name):
    """
    Convert gather.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:gather')

    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]):
        logger.debug('Gather from numpy array')

        if params['axis'] == 0:
            layers[node_name] = np.array(layers[node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 1:
            layers[node_name] = np.array(layers[:, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 2:
            layers[node_name] = np.array(layers[:, :, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 3:
            layers[node_name] = np.array(layers[:, :, :, node.input[0]][layers[node.input[1]]])
        else:
            raise AttributeError('Can\'t gather by axis more than 3.')
    else:
        raise AttributeError('Can\'t gather from tf tensor.')


def convert_concat(node, params, layers, node_name, keras_name):
    """
    Convert concat.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:concat')

    layer_input = [layers[node.input[i]] for i in range(len(node.input))]

    if all([is_numpy(layers[node.input[i]]) for i in range(len(node.input))]):
        logger.debug('Concat numpy arrays.')
        layers[node_name] = np.concatenate(layer_input, axis=params['axis'])
    else:
        logger.debug('Concat Keras layers.')
        layers[node_name] = keras.layers.concatenate(inputs=layer_input,
                                                     axis=params['axis'],
                                                     name=keras_name)

def convert_reshape(node, params, layers, node_name, keras_name):
    """
    Convert reshape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:reshape')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    if is_numpy(input_1):
        logger.debug('The second argument is numpy array.')
        if is_numpy(input_0):
            logger.debug('The first argument is numpy array. Apply np.reshape.')
            layers[node_name] = np.reshape(input_0, input_1)
        else:
            input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)
            logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
            reshape = keras.layers.Reshape(input_1[1:], name=keras_name)
            layers[node_name] = reshape(input_0)
    else:
        raise AttributeError('Can\'t reshape dynamic size.')


def convert_unsqueeze(node, params, layers, node_name, keras_name):
    """
    Convert unsqueeze.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:unsqueeze')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for unsqueeze layer')

    if len(params['axes']) != 1:
        raise AttributeError('Number of axes is not equal 1. Cannot unsqueeze')

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        layers[node_name] = np.expand_dims(layers[node.input[0]], params['axes'][0])
    else:
        if len(params['axes'][0]) != 0:
            raise AttributeError('Axes is not 0. Cannot unsqueeze')

        def target_layer(x):
            import keras
            return keras.backend.expand_dims(x)

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(layers[node.input[0]])


def convert_flatten(node, params, layers, node_name, keras_name):
    """
    Convert flatten.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:flatten')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for flatten layer')

    logger.debug('Convert inputs to Keras/TF layers if needed.')
    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

    reshape = keras.layers.Reshape([-1], name=keras_name)
    layers[node_name] = reshape(input_0)

   
def convert_slice(node, params, layers, node_name, keras_name):
    """
    Convert slice.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:slice')
    
    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for slice layer')
        
    logger.debug('Convert inputs to Keras/TF layers if needed.')
    
    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)
    layers[node_name] = input_0
    
    axes = params["axes"][0]
    ends = params["ends"][0]
    starts = params["starts"][0]
    
    if axes == 0:
        def target_layer(x):
            layer = x[starts:ends]
            return layer
        
        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(input_0)
    elif axes == 1:
        def target_layer(x):
            layer = x[:, starts:ends]
            return layer
        
        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(input_0)
    elif axes == 2:
        def target_layer(x):
            layer = x[:, :, starts:ends]
            return layer
        
        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(input_0)
    elif axes == 3:
        def target_layer(x):
            layer = x[:, :, :, starts:ends]
            return layer
        
        lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
        layers[node_name] = lambda_layer(input_0)
    else:
        raise AttributeError('Not implemented')
