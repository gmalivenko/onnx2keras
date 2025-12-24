"""
The ONNX to keras converter module
"""
import importlib.util
import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import keras
import keras.backend
import tensorflow as tf
from keras.models import Model

from .customonnxlayer import onnx_custom_objects_map
from .exceptions import UnsupportedLayer, OnnxUnsupported
from .layers import AVAILABLE_CONVERTERS
import re

onnx_imported = False
package_name = 'onnx'
spec = importlib.util.find_spec(package_name)
if spec is not None:
    from onnx import numpy_helper

    onnx_imported = True


@dataclass
class ConvertedResponse:
    converted_model: Model
    error_info: Optional[str] = None


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


def flatten_onnx_nodes(onnx_nodes):
    onnx_list = []
    for node in onnx_nodes:
        if node.op_type == "If":
            onnx_list += flatten_onnx_nodes(node.attribute[0].g.node)
            onnx_list += flatten_onnx_nodes(node.attribute[1].g.node)
        onnx_list.append(node)
    return onnx_list


def onnx_to_keras(onnx_model, input_names, name_policy=None, verbose=True, change_ordering=False, input_types=None,
                  allow_partial_compilation=True) \
        -> ConvertedResponse:
    """
    Convert ONNX graph to Keras model format
    :param onnx_model: loaded ONNX model
    :param input_names: list with input names
    :param name_policy: override layer names. None, "short", "renumerate" or "attach_weights_name" (last 2 are experimental)
    :param verbose: verbose output
    :param change_ordering: change ordering to HWC (experimental)
    :return: Keras model
    """
    if not onnx_imported:
        raise OnnxUnsupported()
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
    onnx_nodes = flatten_onnx_nodes(onnx_model.graph.node) # Pulls graph out of If nodes
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
                dtype = None if input_types is None else input_types[i]
                input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim]
                input_shape = [shape if shape != 0 else None for shape in input_shape]
                if len(input_shape) <= 1:
                    input_tensor = keras.layers.InputLayer(input_shape=input_shape, name=input_name, dtype=dtype).output
                    layers[input_name] = input_tensor[0]
                    keras_inputs.append(input_tensor)

                else:
                    batch_size = input_shape[0]
                    input_shape = input_shape[1:]
                    if batch_size is None:
                        layers[input_name] = keras.layers.InputLayer(
                            input_shape=input_shape, name=input_name, dtype=dtype).output
                    else:
                        layers[input_name] = keras.layers.InputLayer(
                            input_shape=input_shape, name=input_name, dtype=dtype, batch_size=batch_size).output

                    keras_inputs.append(layers[input_name])

                logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))

    keras_middle_outputs = {}
    error_info = None
    try:
        # Convert every operation separable
        node_names = []
        embedding_weights_mapping = {}
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
                elif name_policy == 'attach_weights_name':
                    attached_weights_names = []
                    for node_input in node.input:
                        if node_input in weights:
                            weight_name = ".".join(node_input.split(".")[:-1])
                            attached_weights_names.append(weight_name)
                    set_weights_names = set(attached_weights_names)
                    set_weights_names = "__".join(set_weights_names)
                    layer_name = output.replace(":", "_")
                    while not (str.isalpha(layer_name[0]) or str.isdigit(layer_name[0]) or layer_name[0] == "."):
                        layer_name = layer_name[1:]

                    if layer_name == "":
                        layer_name = str(uuid.uuid4())[:10]

                    if set_weights_names:
                        layer_name = f"{layer_name}__{set_weights_names}"

                    keras_names.append(layer_name)
                else:
                    output = output.replace(":", "_")
                    keras_names.append(output)
            keras_names = [k.lstrip("/") for k in keras_names]
            if len(node.output) != 1:
                logger.warning('Trying to convert multi-output node')
                node_params['_outputs'] = list(node.output)
                node_names.extend(keras_names)
            else:
                keras_names = keras_names[0]
                node_names.append(keras_names)
            pattern = r'[#:@]'  # Example pattern to match #, /, and :
            cleaned_node_name = re.sub(pattern, '_', node.name.rstrip("/").lstrip("/"))
            if len(cleaned_node_name) == 0:
                cleaned_node_name = re.sub(pattern, '_', node_name.rstrip("/").lstrip("/"))
            node_params['cleaned_name'] = f'{cleaned_node_name}_tl'
            logger.debug('######')
            logger.debug(f"{node_index/len(onnx_nodes):.1%} completed")
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

                # for case of weights sharing, map the shared weights to determine
                # if a Gather layer is an embedding layer
                if node_type == 'Identity' and node_input in weights:
                    embedding_weights_mapping[node_name] = node_input

                # check conditions for embedding layer
                is_in_weights = node_input in weights  # is this node input in weights
                is_mapped_to_weights = embedding_weights_mapping.get(node_input,
                                                                     '') in weights  # is this node inputs weights are shared with other input
                is_embedding = (
                                       is_in_weights or is_mapped_to_weights) and i == 0  # if either is true this layer is a possible embedding layer

                # if a layer is of type Gather and its input is in weights (or mapped to a weights input)
                # it's an embedding layer
                if node_type == "Gather" and is_embedding:
                    node_params['is_embedding'] = True

                if node_input not in layers:
                    logger.debug('The input not found in layers / model inputs.')
                    if node_input in weights:
                        logger.debug('Found in weights, add as a numpy constant.')
                        layers[node_input] = weights[node_input]
                    else:
                        if node_input == "" and node_type in ('Pad', 'Resize', 'Clip', 'LSTM', 'GRU'):
                            continue
                        else:
                            raise AttributeError('Current node is not in weights / model inputs / layers.')
            else:
                logger.debug('... found all, continue')

            keras.backend.set_image_data_format('channels_first')
            try:
                layer_converter_func = AVAILABLE_CONVERTERS[node_type]
            except KeyError:
                raise UnsupportedLayer(node_type)
            layer_converter_func(
                node,
                node_params,
                layers,
                lambda_funcs,
                node_name,
                keras_names
            )
            # remove node inputs
            for inp in node.input:
                keras_middle_outputs.pop(inp, None)
            # add node to middle map
            if tf.is_tensor(layers[node_name]):
                keras_middle_outputs[node_name] = layers[node_name]

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

        model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)
    except Exception as e:
        if not allow_partial_compilation:
            raise

        if len(keras_middle_outputs) == 0:
            raise e

        error_info = repr(e)
        if isinstance(e, UnsupportedLayer):
            error_info = e.layer_description

        keras_outputs = list(keras_middle_outputs.values())
        try:
            model = keras.models.Model(inputs=keras_inputs, outputs=keras_outputs)
        except:
            raise e

    if change_ordering:
        change_ord_axes_map = {
            3: 2,
            1: 3,
            -1: 1
        }

        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'shared_axes' in layer['config']:
                # TODO: check axes first (if it's not 4D tensor)
                layer['config']['shared_axes'] = [1, 2]

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
                        ), -1), )

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                axis = layer['config']['axis']
                # BatchNorm wrap axis with ListWrapper instead single INT value
                if isinstance(axis, (tuple, list)):
                    axis = axis[0]
                layer['config']['axis'] = change_ord_axes_map.get(axis, layer['config']['axis'])

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                kerasf = list(layer['config']['function'])
                dargs = list(kerasf[1])
                func = lambda_funcs.get(layer['name'])

                if func:
                    # ReduceSum operation has 'axis' param as array of ints. When onnx uses ReduceSum
                    # to reproduce SoftMax - dargs become something like [[1]] (list of lists)
                    # that why we handle collections.Iterable
                    if len(dargs) > 1 or isinstance(dargs[0], (tuple, list)):
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
                            # to list because some tf operations check only for core python types (e.g tf.norm)
                            dargs[i] = axes_map[axis].tolist()
                    else:
                        # if map exits will change else will remain the same
                        dargs[0] = change_ord_axes_map.get(dargs[0], dargs[0])

                kerasf[1] = tuple(dargs)
                layer['config']['function'] = tuple(kerasf)

        keras.backend.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf, custom_objects=onnx_custom_objects_map)

        for dst_layer, src_layer, conf in zip(model_tf_ordering.layers, model.layers, conf['layers']):
            W = src_layer.get_weights()
            # TODO: check axes first (if it's not 4D tensor)
            if conf['config'] and 'shared_axes' in conf['config']:
                W[0] = W[0].transpose(1, 2, 0)
            dst_layer.set_weights(W)

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    response = ConvertedResponse(model, error_info)
    return response


def extract_op_node(node_graph, layers, lambda_funcs, keras_names, change_ordering, name_policy):
    op_node = None
    for node_i, node in enumerate(node_graph):
        if node.op_type == 'Constant':
            node_params = onnx_node_attributes_to_dict(node.attribute)
            # Add global converter info:
            node_params['change_ordering'] = change_ordering
            node_params['name_policy'] = name_policy
            node_name = str(node.output[0])

            AVAILABLE_CONVERTERS[node.op_type](
                node,
                node_params,
                layers,
                lambda_funcs,
                node_name,
                keras_names
            )
        else:  # op type
            if op_node is not None:
                raise NotImplementedError('Not Implemented: inner graph in If node with multiple operator nodes')
            op_node = node
    if op_node is None:
        raise NotImplementedError('Something is off with If node')
    return op_node
