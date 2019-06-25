"""
The ONNX to keras converter module
"""

import keras
import logging
from onnx import numpy_helper

from .layers import AVAILABLE_CONVERTERS


# Use channels first format by default.
keras.backend.set_image_data_format('channels_first')


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


def onnx_to_keras(onnx_model, input_names, verbose=True):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param verbose: verbose output
    :return: Keras model
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger('onnx2keras')

    logger.info('Converter is called.')

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    logger.debug('Gathering weights to dictionary.')
    weights = {}
    for onnx_w in onnx_weights:
        onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
        weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

        logger.debug('Found weight {0} with shape {1}.'.format(
                     onnx_extracted_weights_name,
                     weights[onnx_extracted_weights_name].shape))

    layers = dict()
    keras_outputs = []
    keras_inputs = []

    for input_name in input_names:
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim]
                
                layers[input_name] = keras.layers.InputLayer(
                    input_shape=input_shape[1:], name=input_name
                ).output

                keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    for node in onnx_nodes:
        node_type = node.op_type
        node_name = str(node.output[0])
        node_params = onnx_node_attributes_to_dict(node.attribute)

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

        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            node_name
        )

        # Check for terminal nodes
        for noide_output in node.output:
            if noide_output in onnx_outputs:
                keras_outputs.append(layers[node_name])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)
    return model
