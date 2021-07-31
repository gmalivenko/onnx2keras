"""
The ONNX to Keras converter module
"""

from tensorflow import keras
import logging
from onnx import numpy_helper

from onnx2keras.layers import AVAILABLE_CONVERTERS
from onnx2keras.utils import onnx_node_attributes_to_dict


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
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')
    logger.info('Converter is called.')

    # Use channels first format by default.
    keras_fmt = keras.backend.image_data_format()
    data_format = 'channels_first'

    if change_ordering:
        data_format = 'channels_last'

    keras.backend.set_image_data_format(data_format)

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

    converter_context = {
        'layers': dict(),
        'weights': dict(),
        'keras_outputs': [],
        'keras_inputs': [],
    }

    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            converter_context['weights'][onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            converter_context['weights'][onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     converter_context['weights'][onnx_extracted_weights_name].shape))

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                if change_ordering:
                    input_shape = input_shape[1:] + input_shape[:1]

                keras_input = keras.layers.InputLayer(
                    input_shape=input_shape, name=input_name
                )
                converter_context['layers'][input_name] = keras_input.output
                converter_context['keras_inputs'].append(keras_input.output)

                logger.debug('Input {0} has shape {1}'.format(input_name, input_shape))

    # Convert every layer separably
    logger.debug('Convert static layers / create converter classes')
    for node_index, node in enumerate(onnx_nodes):
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        # Add global converter info:
        # node_params['change_ordering'] = change_ordering
        # node_params['name_policy'] = name_policy

        node_name = str(node.output[0])
        keras_names = []
        for output_index, output in enumerate(node.output):
            if name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(output)

        # if len(node.output) != 1:
        #     logger.warning('Trying to convert multi-output node')
        #     node_params['_outputs'] = list(node.output)
        #     node_names.extend(keras_names)
        # else:
        #     keras_names = keras_names[0]
        #     node_names.append(keras_names)

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

        input_names = []
        for i, node_input in enumerate(node.input):
            logger.debug('Check input %i (name %s).', i, node_input)
            input_names.append(node_input)
            if node_input not in converter_context['layers']:
                logger.debug('The input is not found in layers / model inputs.')

                if node_input in converter_context['weights']:
                    logger.debug('Found input in weights, add to known layers dict a numpy constant.')
                    converter_context['layers'][node_input] = converter_context['weights'][node_input]
                else:
                    raise AttributeError('Current node is not in weights / model inputs / layers.')
        else:
            logger.debug('... found all inputs, continue')

        converter = AVAILABLE_CONVERTERS[node_type](
            node=node, params=node_params, input_names=input_names, output_names=keras_names, data_format=data_format
        )

        for output_index, output in enumerate(node.output):
            converter_context['layers'][str(output)] = converter

        converter.initialize(converter_context=converter_context)

    logger.debug('Convert dynamic layers / call converter classes')

    for node_index, node in enumerate(onnx_nodes):
        node_type = node.op_type
        node_params = onnx_node_attributes_to_dict(node.attribute)

        node_name = str(node.output[0])
        keras_names = []
        for output_index, output in enumerate(node.output):
            if name_policy == 'renumerate':
                postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                keras_names.append('LAYER_%s' % postfix)
            else:
                keras_names.append(output)

        if len(node.output) != 1:
            logger.warning('Trying to convert multi-output node')
        #     node_params['_outputs'] = list(node.output)
        #     node_names.extend(keras_names)
        # else:
        #     keras_names = keras_names[0]
        #     node_names.append(keras_names)

        logger.debug('######')
        logger.debug('...')
        logger.debug('Converting ONNX operation')
        logger.debug('type: %s', node_type)
        logger.debug('node_name: %s', node_name)
        logger.debug('node_params: %s', node_params)
        logger.debug('...')

        converted_layers = converter_context['layers'][node_name].convert(converter_context)

        if len(node.output) == 1:
            print('set result to', str(node.output[0]))
            converter_context['layers'][node_name] = converted_layers
        else:
            for output_index, output in enumerate(node.output):
                print('set result to', i, str(output))
                converter_context['layers'][str(output)] = converted_layers[output_index]
    # Check for terminal nodes
    for layer in onnx_outputs:
        if layer in converter_context['layers']:
            converter_context['keras_outputs'].append(converter_context['layers'][layer])

    # Create model
    model = keras.models.Model(inputs=converter_context['keras_inputs'], outputs=converter_context['keras_outputs'])

    keras.backend.set_image_data_format(keras_fmt)

    return model
