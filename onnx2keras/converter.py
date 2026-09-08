"""
The ONNX to keras converter module
"""

from tensorflow import keras
import logging
import inspect
import types
import numpy as np
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


# Axis of a 4D channels-first tensor -> the same axis once the tensor is channels-last.
NCHW_TO_NHWC_AXES = np.array([0, 3, 1, 2])

CHANGE_ORD_AXES_MAP = {
    0: 0,
    1: 3,
    2: 1,
    3: 2,
    -1: 2
}


def _to_channels_last(shape):
    """
    Move the channel dimension of a batch-free shape from the front to the back.
    :param shape: shape without the batch dimension
    :return: tuple with the channel dimension last
    """
    shape = list(shape)
    return tuple(shape[1:] + shape[:1])


def _relayout_lambda_function(func):
    """
    Copy a Lambda layer function, rewriting the arguments it captured at build time
    so that the function operates on channels-last instead of channels-first tensors.
    :param func: the function a Lambda layer was built from
    :return: an equivalent function for a channels-last graph
    """
    defaults = list(func.__defaults__ or ())
    if not defaults:
        return func

    # The first parameter is the tensor itself, the rest are the captured arguments.
    # Only the arguments that name an axis are remapped: everything else the lambdas
    # capture (a dtype, an exponent, a padding, ...) is layout independent.
    names = list(inspect.signature(func).parameters)[1:]

    for name in ('axis', 'axes'):
        if name not in names:
            continue
        i = names.index(name)
        # The argument holds either a single axis or a list of them (ReduceSum captures
        # something like [1] when onnx uses it to reproduce SoftMax), so map elementwise.
        # to list because some tf operations check only for core python types (e.g tf.norm)
        defaults[i] = NCHW_TO_NHWC_AXES[np.array(defaults[i])].tolist()

    # Lambdas that transpose to a fixed layout capture the data format they were built for.
    if 'data_format' in names:
        defaults[names.index('data_format')] = 'channels_last'

    relayout = types.FunctionType(func.__code__, func.__globals__, func.__name__,
                                  tuple(defaults), func.__closure__)
    relayout.__kwdefaults__ = func.__kwdefaults__
    return relayout


def _clone_layer_channels_last(layer, lambda_funcs):
    """
    Recreate a single layer of a channels-first model for a channels-last graph.
    :param layer: layer of the converted channels-first model
    :param lambda_funcs: functions the Lambda layers were built from, by layer name
    :return: an equivalent channels-last layer
    """
    if isinstance(layer, keras.layers.Lambda):
        func = lambda_funcs.get(layer.name)
        # Lambdas the converter did not register (wrapped constants) are layout independent,
        # and are reused as they are - Keras cannot deserialize them from a config.
        func = _relayout_lambda_function(func) if func is not None else layer.function
        return keras.layers.Lambda(func, name=layer.name)

    conf = layer.get_config()

    if isinstance(layer, keras.layers.Permute) and len(conf['dims']) == 3:
        # Permute dims are 1-based and batch free, which is exactly the ONNX permutation
        # without its batch entry. Re-express it between channels-last axes:
        # out_nhwc[i] = in_nhwc[perm_nhwc[i]] with perm_nhwc = A[perm[A^-1[i]]].
        perm = np.array([0] + list(conf['dims']))
        nhwc_to_nchw = np.argsort(NCHW_TO_NHWC_AXES)
        conf['dims'] = tuple(NCHW_TO_NHWC_AXES[perm[nhwc_to_nchw]].tolist()[1:])

    if 'shared_axes' in conf:
        # TODO: check axes first (if it's not 4D tensor)
        conf['shared_axes'] = [1, 2]

    if 'target_shape' in conf and len(list(conf['target_shape'][1:][:])) > 0:
        conf['target_shape'] = tuple(np.reshape(np.array(
            list(conf['target_shape'][1:]) + [conf['target_shape'][0]]
        ), -1))

    if 'data_format' in conf:
        conf['data_format'] = 'channels_last'

    if 'axis' in conf:
        axis = conf['axis']
        # BatchNorm wrap axis with ListWrapper instead single INT value
        if isinstance(axis, (tuple, list)):
            axis = axis[0]
        conf['axis'] = CHANGE_ORD_AXES_MAP.get(axis, conf['axis'])

    return layer.__class__.from_config(conf)


def sanitize_keras_name(name, used_names):
    """
    Make an ONNX tensor name usable as a Keras layer name.

    Keras 3 rejects any name containing '/', which ONNX exporters emit routinely
    (e.g. '/conv/Conv_output_0'), and requires layer names to be unique.
    :param name: ONNX tensor name
    :param used_names: set of already assigned Keras names, updated in place
    :return: sanitized, unique Keras layer name
    """
    name = str(name).replace('/', '_')

    if name in used_names:
        suffix = 1
        while '%s_%s' % (name, suffix) in used_names:
            suffix += 1
        name = '%s_%s' % (name, suffix)

    used_names.add(name)
    return name


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
    used_keras_names = set()

    for i, input_name in enumerate(input_names):
        for onnx_i in onnx_inputs:
            if onnx_i.name == input_name:
                if input_shapes:
                    input_shape = input_shapes[i]
                else:
                    input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                layers[input_name] = keras.layers.InputLayer(
                    shape=input_shape, name=sanitize_keras_name(input_name, used_keras_names)
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

        keras_names = [sanitize_keras_name(n, used_keras_names) for n in keras_names]

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
            if not node_input:
                # An omitted optional input is encoded as an empty name (e.g. `roi` in Resize).
                logger.debug('Optional input is not set, skip.')
                continue
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
        keras.backend.set_image_data_format('channels_last')

        # Rebuild the graph layer by layer with a channels-last configuration. The model
        # cannot be round-tripped through get_config()/from_config(): Keras does not
        # serialize the locally defined functions the Lambda layers are built from.
        inputs_channels_last = [
            keras.Input(shape=_to_channels_last(keras_input.shape[1:]),
                        dtype=keras_input.dtype,
                        name=keras_input.name)
            for keras_input in model.inputs
        ]

        model_tf_ordering = keras.models.clone_model(
            model,
            input_tensors=inputs_channels_last,
            clone_function=lambda layer: _clone_layer_channels_last(layer, lambda_funcs)
        )

        for dst_layer, src_layer in zip(model_tf_ordering.layers, model.layers):
            W = src_layer.get_weights()
            # TODO: check axes first (if it's not 4D tensor)
            if 'shared_axes' in src_layer.get_config():
                W[0] = W[0].transpose(1, 2, 0)
            dst_layer.set_weights(W)

        model = model_tf_ordering

    keras.backend.set_image_data_format(keras_fmt)

    return model
