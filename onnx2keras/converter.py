"""
The ONNX to keras converter module
"""

from tensorflow import keras
import logging
import inspect
from onnx import numpy_helper

from .layers import AVAILABLE_CONVERTERS


def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_to_keras(onnx_model, input_names,
                  input_shapes=None, name_policy=None, verbose=True, change_ordering=False):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param input_shapes: override input shapes (experimental)
    :param name_policy: override layer names. None, "short" or "renumerate" (experimental)
    :param verbose: verbose output
    :param change_ordering: change ordering to HWC (experimental)
    :return: Keras model
    """
    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    keras.backend.set_image_data_format('channels_first')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('List input shapes:')
    logger.debug(input_shapes)

    logger.debug('List inputs:')
    for i, input in enumerate(onnx_inputs):
        logger.debug('Input {0} -> {1}.'.format(i, input.name))

    logger.debug('List outputs:')
    for i, output in enumerate(onnx_outputs):
        logger.debug('Output {0} -> {1}.'.format(i, output))

    logger.debug('Gathering weights to dictionary.')
    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     weights[onnx_extracted_weights_name].shape))

    layers = dict()
    lambda_funcs = dict()
    keras_outputs = []
    keras_inputs = []

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                layers[input_name] = keras.layers.InputLayer(
                    input_shape=input_shape, name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    node_names = []
    for node_index, node in enumerate(onnx_nodes):
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        node_params['change_ordering'] = change_ordering
        node_params['name_policy'] = name_policy

        node_name = str(node.output[0])
        keras_names = []
        for output_index, output in enumerate(node.output):
            if name_policy == 'short':
                keras_name = keras_name_i = str(output)[:8]
                suffix = 1
                while keras_name_i in node_names:
                    keras_name_i = keras_name + '_' + str(suffix)
                    suffix += 1
                keras_names.append(keras_name_i)
            elif name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
            node_params['_outputs'] = list(node.output)
            node_names.extend(keras_names)
        else:
            keras_names = keras_names[0]
            node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        logger.debug('Check if all inputs are available:')
        if len(node.input) == 0 and node_type != 'Constant':
            raise AttributeError('Operation doesn\'t have an input. Aborting.')

        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            if node_input not in layers:
                logger.debug('The input not found in layers / model inputs.')

                if node_input in weights:
                    logger.debug('Found in weights, add as a numpy constant.')
                    layers[node_input] = weights[node_input]
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all, continue')

        keras.backend.set_image_data_format('channels_first')
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            lambda_funcs,
            node_name,
            keras_names
        )
        if isinstance(keras_names, list):
            keras_names = keras_names[0]

        try:
            logger.debug('Output TF Layer -> ' + str(layers[keras_names]))
        except KeyError:
            pass

    # Check for terminal nodes
    for layer in onnx_outputs:
        if layer in layers:
            keras_outputs.append(layers[layer])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)

    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                                list(layer['config']['target_shape'][1:]) +
                                [layer['config']['target_shape'][0]]
                            ), -1),)

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                layer['config']['axis'] = 3

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                kerasf = list(layer['config']['function'])
                dargs = list(kerasf[1])
                func = lambda_funcs.get(layer['name'])

                if func:
                    if len(dargs) > 1:
                        params = inspect.signature(func).parameters
                        i = list(params.keys()).index('axes') if ('axes' in params) else -1

                        if i > 0:
                            i -= 1
                            axes = list(range(len(dargs[i].shape)))
                            axes = axes[0:1] + axes[2:] + axes[1:2]
                            dargs[i] = np.transpose(dargs[i], axes)

                        i = list(params.keys()).index('axis') if ('axis' in params) else -1

                        if i > 0:
                            i -= 1
                            axis = np.array(dargs[i])
                            axes_map = np.array([0, 3, 1, 2])
                            dargs[i] = axes_map[axis]
                    else:
                        if dargs[0] == -1:
                            dargs = [1]

                kerasf[1] = tuple(dargs)
                layer['config']['function'] = tuple(kerasf)

        keras.backend.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        for dst_layer, src_layer in zip(model_tf_ordering.layers, model.layers):
            dst_layer.set_weights(src_layer.get_weights())

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    return model
