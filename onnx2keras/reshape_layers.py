from tensorflow import keras
import numpy as np
import logging
from .utils import is_numpy, ensure_tf_type, ensure_numpy_type


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
    
    logger.debug('Actual shape:')
    logger.debug(np.array(input_0.shape))
    # logger.debug(np.array(input_0._keras_shape))
    # print(input_0.shape)
    # exit(0)
    # layers[node_name] = np.array(input_0._keras_shape)
    layers[node_name] = np.array([i.value for i in input_0.shape])


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
        if len(layer_input) > 1:
            layers[node_name] = keras.layers.concatenate(inputs=layer_input,
                                                         axis=params['axis'],
                                                         name=keras_name)
        else:
            layers[node_name] = layer_input[0]


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
            layers[node_name] = np.reshape(input_0, np.int32(input_1))
        else:
            if params['change_ordering']:
                input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)

                # Fix critical issue with NHWC
                if input_1[0] is None and input_1[1] == -1:
                    logger.warning('!!! IMPORTANT INFORMATION !!!')
                    logger.warning('The target shape if [None, -1] that means flatten.')
                    logger.warning('But the target ordering is NHWC, so we cant simply perform flatten')
                    logger.warning('The layer will be converted as lambda with tf.transpose')
                    logger.warning('---')

                    def target_layer(x):
                        import tensorflow as tf
                        x = tf.transpose(x, [0, 3, 1, 2])
                        return x

                    lambda_layer = keras.layers.Lambda(target_layer, name="%s_CHW" % keras_name)
                    layers[node_name] = lambda_layer(input_0)
                else:
                    layers[node_name] = input_0

                reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
                layers[node_name] = reshape(layers[node_name])

            else:
                input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]], name="%s_const" % keras_name)
                logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
                logger.debug('Target shape :')
                logger.debug(np.int32(input_1[1:]))

                if len(np.int32(input_1[1:])) == 1 and np.int32(input_1[1:])[0] == -1:
                    logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                    flatten = keras.layers.Flatten(name=keras_name)
                    layers[node_name] = flatten(input_0)
                else:
                    reshape = keras.layers.Reshape(np.int32(input_1[1:]), name=keras_name)
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

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        layers[node_name] = layers[node.input[0]]
        shift = 0
        for axis in params['axes']:
            layers[node_name] = np.expand_dims(layers[node_name], axis + shift)
            shift += axis
    else:

        if len(params['axes']) != 1:
            raise AttributeError('Number of axes is not equal 1. Cannot unsqueeze')

        # if params['axes'][0] != 0:
        #     raise AttributeError('Axes is not 0. Cannot unsqueeze')

        def target_layer(x, axis=params['axes'][0]):
            from tensorflow import keras
            return keras.backend.expand_dims(x, axis)

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

    if params['change_ordering']:
        # Fix critical issue with flatten
        def target_layer(x):
            import tensorflow as tf
            x = tf.transpose(x, [0, 3, 1, 2])
            return x

        lambda_layer = keras.layers.Lambda(target_layer,  name="%s_CHW" % keras_name)
        tensor_chw = lambda_layer(input_0)
        flatten = keras.layers.Flatten(name=keras_name)
        layers[node_name] = flatten(tensor_chw)
    else:
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


def convert_squeeze(node, params, layers, node_name, keras_name):
    """
    Convert Squeeze layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for squeeze layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    def target_layer(x, axis=params['axes'][0]):
        from tensorflow import keras
        return keras.backend.squeeze(x, axis)

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)


def convert_expand(node, params, layers, node_name, keras_name):
    """
    Convert Expand layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 2:
        assert AttributeError('More than 2 input for expand layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    input_1 = ensure_numpy_type(layers[node.input[1]])

    def target_layer(x, shape=input_1):
        from tensorflow import keras

        # if (len(x.shape) == len(shape)):
        #     for axis, new_shape in enumerate(shape):
        #         if axis == 0:
        #             continue
        #         x = keras.backend.repeat_elements(x, int(new_shape // x.shape[axis]), axis)
        #     pass

        x = keras.backend.repeat_elements(x, int(shape[1] // x.shape[1]), 1)
        x = keras.backend.repeat_elements(x, int(shape[2] // x.shape[2]), 2)
        return x

        # Proper version
        # return tf.broadcast_to(x, (1, *shape[1:]))

    lambda_layer = keras.layers.Lambda(target_layer, name=keras_name)
    layers[node_name] = lambda_layer(input_0)
