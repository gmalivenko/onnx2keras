"""
The ONNX to keras converter module
"""

import keras
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
        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_to_keras(
    onnx_model, input_names,
    verbose=True
):
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param verbose: verbose output
    :return: Keras model
    """

    onnx_weights = onnx_model.graph.initializer
    onnx_inputs = onnx_model.graph.input
    onnx_outputs = [i.name for i in onnx_model.graph.output]
    onnx_nodes = onnx_model.graph.node

    weights = {}
    for onnx_w in onnx_weights:
        onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
        weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        if verbose:
            print(
                'Found weight {0} with shape {1}'.format(onnx_extracted_weights_name,
                                                         weights[onnx_extracted_weights_name].shape)
            )

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

                if verbose:
                    print('Found input {0} with shape {1}'.format(input_name, input_shape))

    # Convert every operation separable
    for node in onnx_nodes:
        node_type = node.op_type
        node_name = str(node.output[0])
        node_params = onnx_node_attributes_to_dict(node.attribute)

        if verbose:
            print('Converting ONNX operation')
            print('type:', node_type)
            print('node_name:', node_name)
            print('node_params:', node_params)
            
        AVAILABLE_CONVERTERS[node_type](
            node,
            node_params,
            layers,
            weights,
            node_name
        )
        for noide_output in node.output:
            if noide_output in onnx_outputs:
                keras_outputs.append(layers[node_name])

    # Create model
    model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)
    return model
