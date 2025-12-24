import logging

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.keras_tensor import KerasTensor
from keras.layers import SlicingOpLambda, Lambda
from typing import Union
from .utils import is_numpy, ensure_tf_type, unsqueeze_tensors_of_rank_one
from .tfops_funcs import tf_reshape, tf_shape, tf_cast, tf_stack, tf_image_resize, tf_strided_slice,\
    tf_squeeze, tf_transpose, tf_where, tf_gather, tf_range, tf_reduce_sum, tf_abs, tf_expand_dims, tf_concat, \
    tf_shape, tf_tile, tf_fill, tf_gather_nd, tf_reduce_sum, tf_zeros_like, tf_multiply, tf_tensor_scatter_nd_update,\
    tf_ones


def convert_transpose(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert transpose.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.transpose')
    input_name = node.input[0]

    if params['perm'][0] != 0:
        logger.warning('Can\'t permute batch dimension. Result may be wrong.')
        if is_numpy(layers[input_name]):
            logger.warning('Transposing numpy array.')
            layers[node_name] = np.transpose(layers[input_name], axes=params['perm'])
        else:
            layers[node_name] = tf_transpose(layers[input_name], perm=params['perm'],
                                             tf_name=f"{params['cleaned_name']}_transpose")
    else:
        permute = keras.layers.Permute(params['perm'][1:], name=f"{params['cleaned_name']}_transpose")
        layers[node_name] = permute(layers[input_name])


def convert_shape(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert shape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.shape')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    logger.debug('Actual shape:')
    logger.debug(np.array(input_0.shape))
    is_unknown_tensor = input_0.shape == None
    if not is_unknown_tensor and (
            not K.is_keras_tensor(input_0) or not any([input_0.shape[i] == None for i in range(len(input_0.shape))])):
        shapes = []
        for i in input_0.shape:
            if i is not None:
                shapes.append(i)
            else:
                shapes.append(None)
        layers[node_name] = np.array(shapes)
    else:
        layers[node_name] = tf_shape(input_0, out_type=tf.int64, tf_name=f"{params['cleaned_name']}_shape")


def optimize_constant_array_for_serialization(input_0: tf.Tensor, params, indices: Union[np.ndarray, tf.Tensor], logger):
    """
    Some models contain repetition in the constant array tf.gather gathers from.
    In such cases, serialization takes a long time. serializing 3000 elements take ~3 sec (and scale linearly).
    To optimize for this case, we detect the repetition, take the modulu of the repetition from indices,
    and making the param array shorter.
    This is especially useful for models containing pre-top-k elements.
    Args:
        input_0: the array to gather on
        indices: the indices to gather

    Returns:
        input_0: the new array to gather on
        indices: the new indices to gather
    """
    # This Optimization is a must for models with Pre-Top-K needs
    logger.debug('onnx2keras.gather - Encountered long gather. '
                 'Trying to shorten it for easy serialization')
    max_inp = tf.reduce_max(input_0)
    min_inp = tf.reduce_min(input_0)
    if len(input_0) % (max_inp - min_inp + 1) == 0:
        range_inp = tf_range(min_inp, max_inp + 1, dtype=input_0.dtype,
                             tf_name=f"{params['cleaned_name']}_optimize_range")
        reshape_columns = tf_reshape(input_0, [-1, max_inp + 1], tf_name=f"{params['cleaned_name']}_optimize_reshape")
        if tf_reduce_sum(tf_abs((reshape_columns - range_inp[None, :]),
                                tf_name=f"{params['cleaned_name']}_optimize_abs"),
                         tf_name=f"{params['cleaned_name']}_optimize_sum") == 0:
            # This tests for a long constant array that has a repeating series like [0,1,2,3,0,1,2,3]
            logger.debug('onnx2keras.gather - Shortening sequence - columns')
            indices = indices % (max_inp + 1)
            input_0 = range_inp
        else:
            repetition_len = np.argmin(input_0 == input_0[0])
            if repetition_len > 0 and len(input_0) % repetition_len == 0:
                # This tests for a long constant array that has a repeating series like [0,0,0,1,1,1,2,2,2]
                reshaped_rows = tf_reshape(input_0, [-1, repetition_len],
                                           tf_name=f"{params['cleaned_name']}_optimize_reshape")
                first_row = reshaped_rows[:, 0]
                if tf_reduce_sum(tf_abs(reshaped_rows - first_row[:, None],
                                        tf_name=f"{params['cleaned_name']}_optimize_abs_2"),
                                 tf_name=f"{params['cleaned_name']}_optimize_sum_2") == 0:
                    logger.debug('onnx2keras.gather - Shortening sequence - rows')
                    indices = indices // repetition_len
                    input_0 = first_row
    return input_0, indices


def convert_gather(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert gather.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.gather')
    OPTIMIZE_ARRAY_LENGTH = 50000
    axis = params.get('axis', 0)
    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]) and not 'is_embedding' in params:
        logger.debug('Gather from numpy array')
        if axis == 0:
            gathered = np.array(layers[node.input[0]][layers[node.input[1]]])
        elif axis == 1:
            gathered = np.array(layers[:, node.input[0]][layers[node.input[1]]])
        elif axis == 2:
            gathered = np.array(layers[:, :, node.input[0]][layers[node.input[1]]])
        elif axis == 3:
            gathered = np.array(layers[:, :, :, node.input[0]][layers[node.input[1]]])
        else:
            raise AttributeError('Can\'t gather by axis more than 3.')

        if gathered.dtype == np.object0:
            try:
                gathered = gathered.astype(np.int32)
            except TypeError:
                pass
        layers[node_name] = gathered
    else:
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        if isinstance(layers[node.input[1]], (np.integer, int)) or \
            (not isinstance(layers[node.input[1]], np.ndarray) and \
                K.is_keras_tensor(layers[node.input[1]])):
            #indices are keras tensor or int
            indices = layers[node.input[1]]
        else: #indices are numpy/tf.eager
            indices = layers[node.input[1]]
            if not is_numpy(layers[node.input[1]]):
                indices = indices.numpy()
            indices = indices.tolist()
        if "is_embedding" in params:
            if len(input_0.shape) == 2:
                emb = tf.keras.layers.Embedding(input_0.shape[0], input_0.shape[1], weights=[layers[node.input[0]]],
                                                name=f"{params['cleaned_name']}_gather_emb")
                if isinstance(indices, list):
                    layers[node_name] = emb(np.array(indices))
                else:
                    layers[node_name] = emb(indices)
            else:
                raise AttributeError("Cannot transform gather into embedding with non 2D array")
        else:
            if tf.is_tensor(indices) and indices.dtype not in [tf.int16, tf.int32, tf.int64]:
                indices = tf_cast(indices, tf.int32, tf_name=f"{params['cleaned_name']}_gather_cast_indices")

            if type(indices) == int:
                out_type = tf.int32
            else:
                if isinstance(indices, list):
                    out_type = np.array(indices).dtype
                else:
                    out_type = indices.dtype

            dim_len = tf_shape(input_0, out_type=out_type,
                               tf_name=f"{params['cleaned_name']}_gather_input_shape")[axis]  # support None
            if isinstance(indices, list):
                for i in range(len(indices)):
                    try:
                        if indices[i] < 0:
                            indices[i] = int(indices[i]) + dim_len
                    except TypeError:
                        pass


            if isinstance(indices, (int, np.integer)) and indices < 0:
                indices += dim_len
            if tf.is_tensor(indices):
                indices = tf_where(indices < 0, indices + dim_len, indices,
                                   tf_name=f"{params['cleaned_name']}_gather_indices_where")
            if isinstance(input_0, np.ndarray) or not  K.is_keras_tensor(input_0):
                if len(input_0) > OPTIMIZE_ARRAY_LENGTH:
                    input_0, indices = optimize_constant_array_for_serialization(input_0, params, indices, logger)
            layers[node_name] = tf_gather(input_0, indices, axis=axis, tf_name=f"{params['cleaned_name']}_gather")


def convert_concat(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert concat.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.concat')

    layer_input = [layers[node.input[i]] for i in range(len(node.input))]

    if all([is_numpy(layers[node.input[i]]) for i in range(len(node.input))]):
        logger.debug('Concat numpy arrays.')
        layers[node_name] = np.concatenate(layer_input, axis=params['axis'])
    else:
        logger.debug('Concat Keras layers.')
        if len(layer_input) > 1:
            if not np.array([tf.is_tensor(layer_input[i]) and K.is_keras_tensor(layer_input[i]) for i in
                             range(len(layer_input))]).all() or any(
                [layer_input[i].shape == None for i in range(len(layer_input))]):
                try:
                    layers[node_name] = tf_concat(layer_input, axis=params['axis'],
                                                  tf_name=f"{params['cleaned_name']}_concat")
                except Exception as ex:
                    # might be due to type mismatch between different inputs of tf.concat
                    raise

            else:
                layer_input = unsqueeze_tensors_of_rank_one(layer_input, axis=params['axis'], name=params['cleaned_name'])
                layers[node_name] = keras.layers.concatenate(inputs=layer_input,
                                                             axis=params['axis'],
                                                             name=f"{params['cleaned_name']}_concat_2")
        else:
            layers[node_name] = layer_input[0]


def convert_reshape(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert reshape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.reshape')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    logger.debug(f'input_0: {str(input_0)}')
    logger.debug(f'input_1: {str(input_1)}')

    if is_numpy(input_1):
        dims_to_set_as_zero = None
        dims_to_keep_unchanged = None
        allow_zero = params.get('allowzero', False)
        contains_zero_dim = np.isin(input_1, 0).any()
        contains_infer_dim = np.isin(input_1, -1).any()
        if allow_zero:
            if contains_infer_dim and contains_zero_dim:
                raise ValueError(
                    "Reshape parameter 'allowzero' is set and reshaping argument contains both '0' dim and '-1'"
                    "which is not allowed"
                    f"node name: {node_name}")
            elif contains_zero_dim:
                dims_to_set_as_zero = np.argwhere(input_1 == 0)
        elif not allow_zero and contains_zero_dim:
            dims_to_keep_unchanged = np.squeeze(np.argwhere(input_1 == 0))

        logger.debug('The second argument is numpy array.')
        if is_numpy(input_0):
            logger.debug('The first argument is numpy array. Apply np.reshape.')
            layers[node_name] = np.reshape(input_0, np.int32(input_1))
        else:
            if params['change_ordering']:
                input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

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

                    lambda_layer = keras.layers.Lambda(target_layer,
                                                       name="%s_CHW" % f"{params['cleaned_name']}_reshape_lambda")
                    layers[node_name] = lambda_layer(input_0)
                    lambda_func[keras_name] = target_layer
                else:
                    layers[node_name] = input_0

                reshape = keras.layers.Reshape(np.int32(input_1[1:]),
                                               name=f"{params['cleaned_name']}_reshape_input_2_1")  # keras reshape ignores batch dimension but onnx axis do not
                layers[node_name] = reshape(layers[node_name])

            else:
                input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
                input_0_shape = input_0.shape
                if len(input_0_shape) > 0:
                    first_mismatch = np.argmin(np.array(input_0_shape[:len(input_1)]) == input_1)
                else:  # does not need to reshape dynamicalyy (skip next section)
                    first_mismatch = 0
                if (input_1 == None).any() and (np.array(input_0_shape) == None).any() and len(input_1) < len(
                        input_0_shape) \
                        and input_1[first_mismatch] == -1:  # reshape end
                    end_match_arr = np.array(input_0_shape[-len(input_1):]) == input_1
                    end_idx_match = np.argmax((np.array(input_0_shape[-len(input_1):]) == input_1))
                    end_idx_match = end_idx_match + len(input_0_shape) - len(input_1) if end_idx_match > first_mismatch \
                                                                                         and end_match_arr[
                                                                                             end_idx_match] else len(
                        input_0_shape) + 1
                    ten_shape = tf_shape(input_0, tf_name=f"{params['cleaned_name']}_shape")
                    layers[node_name] = tf_reshape(input_0,
                                                   [*ten_shape[:first_mismatch], -1,
                                                    *ten_shape[end_idx_match:]],
                                                   tf_name=f"{params['cleaned_name']}_reshape")
                else:
                    logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
                    logger.debug('Target shape :')
                    logger.debug(np.int32(input_1[1:]))
                    if len(np.int32(input_1[1:])) == 1 and np.int32(input_1[1:])[0] == -1:
                        if input_0.shape.rank == 1:
                            input_0 = tf_expand_dims(input_0, 0, tf_name=f"{params['cleaned_name']}_expand_dims")
                        logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                        flatten = keras.layers.Reshape(target_shape=input_1[1:],
                                                       name=f"{params['cleaned_name']}_reshape_input_2_2")
                        layers[node_name] = flatten(input_0)
                    elif len(input_1) == 1 and input_1[0] == -1:
                        layers[node_name] = tf_reshape(input_0, [-1], tf_name=f"{params['cleaned_name']}_reshape_1")
                    else:
                        if len(input_0.shape) == 0 or (
                                input_0.shape[0] != input_1[0] and input_1[0] != 0):  # keras reshape don't work
                            new_shape = input_1.copy()
                            if dims_to_set_as_zero is not None:
                                new_shape[dims_to_set_as_zero] = 0
                            elif dims_to_keep_unchanged is not None:
                                new_shape[dims_to_keep_unchanged] = np.array(input_0.shape)[dims_to_keep_unchanged]
                            layers[node_name] = tf_reshape(input_0, new_shape,
                                                           tf_name=f"{params['cleaned_name']}_reshape_2")
                        else:
                            reshape = keras.layers.Reshape(np.int32(input_1[1:]),
                                                           name=f"{params['cleaned_name']}_reshape_input_2_3")
                            layers[node_name] = reshape(input_0)
    else:  # dynamic reshape
        layers[node_name] = tf_reshape(input_0, input_1, tf_name=f"{params['cleaned_name']}_reshape_3")


def convert_unsqueeze(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert unsqueeze.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.unsqueeze')

    if len(node.input) != 1:
        if len(node.input) == 2:
            params['axes'] = layers[node.input[1]]
        else:
            raise AttributeError('Number of inputs is not equal 1 for unsqueeze layer')

    if len(np.unique(params['axes'])) < len(params['axes']):
        raise AttributeError(f"The specified axes contains duplicates values: {params['axes']}")

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        layers[node_name] = layers[node.input[0]]
        for axis in params['axes']:
            layers[node_name] = np.expand_dims(layers[node_name], axis)
    else:
        unsqueezed_input = layers[node.input[0]]
        for axis in params['axes']:
            unsqueezed_input = tf_expand_dims(unsqueezed_input, axis,
                                              tf_name=f"{params['cleaned_name']}_expand_dims_ax_{str(axis)}")

        layers[node_name] = unsqueezed_input


def convert_flatten(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert flatten.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.flatten')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for flatten layer')

    logger.debug('Convert inputs to Keras/TF layers if needed.')
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    input_dims = tf_shape(input_0, tf_name=f"{params['cleaned_name']}_shape")
    flatten_axis = params.get('axis', 1)
    reshaped_input = tf_reshape(input_0, [tf.reduce_prod(input_dims[:flatten_axis]),
                                          tf.reduce_prod(input_dims[flatten_axis:])],
                                tf_name=f"{params['cleaned_name']}_flatten")
    layers[node_name] = reshaped_input


def convert_slice(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert slice.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.slice')
    max_ends_val = np.iinfo(np.int32).max

    if params['change_ordering']:
        raise NotImplementedError("change_ordering for Slice is not implemented")
    if 'axes' in params:
        axes = list(params["axes"])
        ends = list(params["ends"])
        starts = list(params["starts"])
        steps = list(params.get("steps", [None] * len(axes)))
    else:
        starts = list(layers[node.input[1]])
        ends = list(layers[node.input[2]])
        try:
            axes = list(layers[node.input[3]])
        except:
            input_rank = len(layers[node.input[0]].shape)
            axes = list(range(input_rank))
        try:
            steps = list(layers[node.input[4]])
        except IndexError:
            steps = list(params.get("steps", [None] * len(axes)))

    # when the 'ends' value is the int64 maximum, probably happen because [idx:] sets large end num in conversion
    if not isinstance(ends[0], KerasTensor):
        for i in range(len(ends)):
            if hasattr(ends[i], 'dtype'):
                if ends[i].dtype == np.int64 and ends[i] > max_ends_val:
                    ends[i] = np.int32(max_ends_val)
            elif isinstance(ends[i], int) and ends[i] > max_ends_val:
                ends[i] = np.int32(max_ends_val)
    try:
        max_len = len(layers[node.input[0]].shape)
        axes_positives = [axis if axis >= 0 else max_len + axis for axis in axes]
    except ValueError as e:
        if layers[node.input[0]].shape == None:  # tensor with unknown shape (not the same as dynamic)
            max_len = max(axes) + 1
            if any([axis < 0 for axis in axes]):
                raise NotImplementedError("For a tensor with unknown shape, can't use negative axis")
            else:
                axes_positives = axes
        else:
            raise NotImplementedError(f"Couldn't transform the axis in a slice layer {node_name}")
    slice_spec_param = []
    is_dynamic = False
    for i in range(len(starts)):
        for index_li in [starts, steps, ends]:
            if index_li[i] is not None and not isinstance(index_li[i], int) and not is_numpy(
                    index_li[i]) and K.is_keras_tensor(index_li[i]):
                is_dynamic = True
    if not is_dynamic:
        for axis in range(max_len):
            if axis in axes_positives:
                axis_index = axes_positives.index(axis)
                start = starts[axis_index]
                end = ends[axis_index]
                step = steps[axis_index]
                slice_spec_param.append({'start': start, 'step': step, 'stop': end})
            else:
                slice_spec_param.append({'start': None, 'step': None, 'stop': None})
        if is_numpy(layers[node.input[0]]) and np.array([_shape is None for _shape in layers[node.input[0]]]).any() \
                and len(layers[node.input[0]].shape) == 1:  # slice numpy array which is a shape
            sliced = layers[node.input[0]][start:end:step]
        else:
            input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
            slicing_layer = SlicingOpLambda(tf.__operators__.getitem)
            sliced = slicing_layer(input_0, slice_spec=slice_spec_param)
            if is_numpy(layers[node.input[0]]) and not K.is_keras_tensor(sliced):
                sliced = sliced.numpy()
        layers[node_name] = sliced
    else:
        try:
            steps = list(layers[node.input[4]])
        except IndexError:
            steps = list(params.get("steps", [1] * len(axes)))
        input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
        keras_shape = tf_shape(layers[node.input[0]], tf_name=f"{params['cleaned_name']}_shape")
        start_vec = [0] * max_len
        end_vec = [keras_shape[i] for i in range(max_len)]
        step_vec = [1] * max_len
        for axis in range(max_len):
            if axis in axes_positives:
                axis_index = axes_positives.index(axis)
                for res_list, input_list in zip([start_vec, step_vec, end_vec], [starts, steps, ends]):
                    slice_index = input_list[axis_index]
                    if input_list[axis_index] is not None and not isinstance(slice_index, int) and not is_numpy(
                            input_list[axis_index]) and input_list[axis_index].dtype != tf.int32:
                        slice_index = tf_cast(slice_index, tf.int32, tf_name=f"{params['cleaned_name']}_cast")
                    res_list[axis] = slice_index
        layers[node_name] = tf_strided_slice(input_0,
                                             tf.concat([start_vec], axis=0),
                                             tf.concat([end_vec], axis=0),
                                             tf.concat([step_vec], axis=0),
                                             tf_name=f"{params['cleaned_name']}_strided_slice")


def convert_squeeze(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Squeeze layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    axis = None
    if 'axes' in params:
        axis = params['axes'][0]

    if len(node.input) == 2:
        axis = layers[node.input[1]].tolist()
    layers[node_name] = tf_squeeze(input_0, axis=axis, tf_name=f"{params['cleaned_name']}_squeeze")


def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
    logger = logging.getLogger('onnx2keras.reshape')
    input_tensor = layers[node.input[0]]
    roi = None if len(node.input[1]) == 0 else layers[node.input[1]]
    scales = [] if len(node.input[2]) == 0 else layers[node.input[2]]
    sizes = None
    nearest_mode = params.get('nearest_mode', b'round_prefer_floor')
    if len(node.input) == 4:
        sizes = layers[node.input[3]]
    if roi:
        raise Exception("Resize with roi not supported")
    if params['mode'] == b'nearest':
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif params['mode'] == b'cubic':
        resize_method = tf.image.ResizeMethod.BICUBIC
    elif params['mode'] == b'linear':
        resize_method = tf.image.ResizeMethod.BILINEAR
    else:
        raise Exception("unsupported resize method")

    rank = len(input_tensor.shape)
    axes = params.get('axes', list(range(rank)))  # Default to resizing all axes

    # Validate axes
    axes = [a if a >= 0 else a + rank for a in axes]  # Convert negative axes to positive
    if any(a < 0 or a >= rank for a in axes):  # check that all axes values are within input rank
        raise ValueError("Invalid axes value")
    to_channel_last = keras.layers.Permute((2, 3, 1),
                                           name=f"{params['cleaned_name']}_resize_channels_last")(input_tensor)  # (B, W, H, C)
    shape = tf_cast(tf_shape(input_tensor, tf_name=f"{params['cleaned_name']}_resize_shape"), tf.int32,
                    tf_name=f"{params['cleaned_name']}_resize_cast")
    if shape.shape != 4:
      raise Exception("resize layer for input tensor with rank != 4 is not supported")
    if isinstance(sizes, KerasTensor) or isinstance(scales, KerasTensor):
        tf_resize_shapes = tf_zeros_like(shape, tf_name=f"{params['cleaned_name']}_resize_zeros_like")

        if len(scales) > 0:
            for i, axis in enumerate(axes):
                indices = tf.constant([[axis]])
                update = tf_cast(tf_multiply(scales[i],
                                             tf_cast(shape[axis],
                                                     tf.float32,
                                                    tf_name=f"{params['cleaned_name']}_resize_cast_2_ax_{i}"
                                                     ),
                                             tf_name=f"{params['cleaned_name']}_resize_multiply_ax_{i}"),
                                 tf.int32,
                                 tf_name=f"{params['cleaned_name']}_resize_cast_3_ax_{i}")
                updates = tf_reshape(update, (1,), tf_name=f"{params['cleaned_name']}_resize_reshape_1_ax_{i}")
                tf_resize_shapes = tf_tensor_scatter_nd_update(tf_resize_shapes,
                                                               indices,
                                                               updates,
                                                               tf_name=f"{params['cleaned_name']}_resize_scatter_ax_{i}")

        else:
            for i, axis in enumerate(axes):
                indices = tf.constant([[axis]])
                # The value to update at the specified index
                update = tf_cast(sizes[i],
                                 tf.int32,
                                 tf_name=f"{params['cleaned_name']}_resize_cast_4_ax_{i}")
                updates = tf_reshape(update,
                                     (1,),
                                     tf_name=f"{params['cleaned_name']}_resize_reshape_1_ax_{i}")
                # Apply the update using tf.scatter_nd
                curr_name_str = f"{params['cleaned_name']}_resize_scatter_1_ax_{i}"
                tf_resize_shapes = tf_tensor_scatter_nd_update(tf_resize_shapes,
                                                               indices,
                                                               updates,
                                                               tf_name=curr_name_str)
        resize_size = tf_stack(tf_gather(tf_resize_shapes,
                                         [2, 3],
                                         tf_name=f"{params['cleaned_name']}_resize_gather"),
                               axis=0,
                               tf_name=f"{params['cleaned_name']}_resize_stack")
    else:
        tf_resize_shapes = [shape[i] for i in range(2, 4)] # (W, H) for input tensor of shape [B, C, W, H]
        if len(scales) > 0:
            for i, axis in enumerate(axes):
                if scales[i] != 1:
                    tf_resize_shapes[axis - 2] = tf_cast(scales[i] * tf_cast(tf_resize_shapes[axis - 2],
                                                                             tf.float32,
                                                                             tf_name=f"{params['cleaned_name']}_resize_cast_5_ax_{i}"),
                                                         tf.int32,
                                                         tf_name=f"{params['cleaned_name']}_resize_cast_6_ax_{i}")
        else:
            for i, axis in enumerate(axes):
                if sizes[i] != input_tensor.shape[axis]:
                    tf_resize_shapes[axis - 2] = int(sizes[i])
        resize_size = tf_stack(tf_resize_shapes,
                               axis=0,
                               tf_name=f"{params['cleaned_name']}_resize_stack_1")

    if (
        resize_method == tf.image.ResizeMethod.NEAREST_NEIGHBOR
        and isinstance(resize_size, keras.engine.keras_tensor.KerasTensor)
        and nearest_mode.decode() == "floor"
    ):
        logger.warning("floor nearest mode will result in faulty conversion")
    if resize_method == tf.image.ResizeMethod.NEAREST_NEIGHBOR and nearest_mode.decode() == 'floor'\
        and not isinstance(resize_size, keras.engine.keras_tensor.KerasTensor):
        if not isinstance(resize_size, np.ndarray) :
            resize_size = np.array(resize_size)

        def target_layer(x, resize_size=resize_size):
            from tensorflow.python.ops.image_ops import resize_nearest_neighbor

            return resize_nearest_neighbor(x, resize_size, half_pixel_centers=False)
        lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_resize_lambda")
        resized = lambda_layer(to_channel_last)
    else:
        resized = tf_image_resize(to_channel_last,
                                  size=resize_size,
                                  method=resize_method,
                                  tf_name=f"{params['cleaned_name']}_image_resize")
    to_channel_first = keras.layers.Permute((3, 1, 2),
                                            name=f"{params['cleaned_name']}_resize_channels_first")(resized)
    layers[node_name] = to_channel_first


def convert_expand(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Expand layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 2:
        assert AttributeError('More than 2 input for expand layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    input_1 = layers[node.input[1]]
    was_converted_from_bool = False
    if input_0.dtype.is_bool:
        input_0 = tf_cast(input_0, dtype='int32', tf_name=f"{params['cleaned_name']}_bool_to_int")
        was_converted_from_bool = True
    multiply_res = input_0 * tf_ones(shape=input_1, dtype=input_0.dtype,
                                     tf_name=f"{params['cleaned_name']}_expand_use_ones")
    # input_0.dtype == np.int32 since we can't serialize constants as int64, need to cast to true type
    if layers[node.input[0]].dtype == np.int64:
        multiply_res = tf_cast(multiply_res, tf.int64, tf_name=f"{params['cleaned_name']}_to_int64")
    if was_converted_from_bool:
        multiply_res = tf_cast(multiply_res, tf.bool, tf_name=f"{params['cleaned_name']}_int_to_bool")
    layers[node_name] = multiply_res


def convert_tile(node, params, layers, lambda_func, node_name, keras_name):
    layers[node_name] = tf_tile(layers[node.input[0]], layers[node.input[1]], tf_name=f"{params['cleaned_name']}_tile")


def convert_gather_elements(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert gather.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.gather_elements')
    axis = params.get('axis', 0)
    data_input = layers[node.input[0]]
    indices_input = layers[node.input[1]]

    def torch_gather(x, indices, gather_axis):
        all_indices = tf_where(tf_fill(indices.shape, True, tf_name=f"{params['cleaned_name']}_gather_fill"),
                               tf_name=f"{params['cleaned_name']}_gather_where")
        gather_locations = tf_reshape(indices, [indices.shape.num_elements()],
                                      tf_name=f"{params['cleaned_name']}_gather_reshape")

        gather_indices = []
        for axis in range(len(indices.shape)):
            if axis == gather_axis:
                gather_indices.append(tf_cast(gather_locations, dtype=tf.int64,
                                              tf_name=f"{params['cleaned_name']}_gather_cast_loc"))
            else:
                gather_indices.append(tf_cast(all_indices[:, axis], dtype=tf.int64,
                                              tf_name=f"{params['cleaned_name']}_gather_cast_all"))

        gather_indices = tf_stack(gather_indices, axis=-1, tf_name=f"{params['cleaned_name']}_gather_indices")
        gathered = tf_gather_nd(x, gather_indices, tf_name=f"{params['cleaned_name']}_gather_nd")
        reshaped = tf_reshape(gathered, indices.shape, tf_name=f"{params['cleaned_name']}_reshape")
        return reshaped

    layers[node_name] = torch_gather(data_input, indices_input, axis)


def col2im_onnx(node, params, layers, lambda_func, node_name, keras_name):

    input_cols  = layers[node.input[0]]   # (N, C*kH*kW, L) KerasTensor
    image_shape = layers[node.input[1]]   # constant [H_img, W_img]
    block_shape = layers[node.input[2]]   # constant [kH, kW]

    dilations = params.get('dilations', [1, 1])
    pads      = params.get('pads',      [0, 0, 0, 0])  # [pt, pl, pb, pr]
    strides   = params.get('strides',   [1, 1])

    def _col2im_lambda(x):
        """x is input_cols inside Lambda → symbolic tf.Tensor"""

        input_cols_sym = x  # (N, C*kH*kW, L)

        # image_shape / block_shape come from ONNX initializers
        H_img = int(image_shape[0])
        W_img = int(image_shape[1])

        kH = int(block_shape[0])
        kW = int(block_shape[1])

        dH, dW = dilations
        pt, pl, pb, pr = pads
        sH, sW = strides

        # -----------------------------
        # Shapes
        # -----------------------------
        N  = tf.shape(input_cols_sym)[0]
        Ck = tf.shape(input_cols_sym)[1]  # C * kH * kW
        L  = tf.shape(input_cols_sym)[2]

        C = Ck // (kH * kW)  # recover C

        # Effective kernel with dilation
        kH_eff = kH + (kH - 1) * (dH - 1)
        kW_eff = kW + (kW - 1) * (dW - 1)

        # Padded output spatial size
        H_pad = H_img + pt + pb
        W_pad = W_img + pl + pr

        # Sliding-window grid (ONNX spec)
        H_out = (H_pad - kH_eff) // sH + 1
        W_out = (W_pad - kW_eff) // sW + 1
        # Optional sanity check:
        # tf.debugging.assert_equal(L, H_out * W_out)

        # -----------------------------
        # Reshape ONNX (N,C*kH*kW,L) → (N,C,kH,kW,H_out,W_out)
        # -----------------------------
        cols = tf.reshape(input_cols_sym, (N, C, kH, kW, L))
        cols = tf.reshape(cols,          (N, C, kH, kW, H_out, W_out))

        # -----------------------------
        # Prepare accumulation buffer (no normalization)
        # -----------------------------
        out_pad = tf.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

        # Precompute sliding indices (int32)
        h_idx_base = tf.range(H_out, dtype=tf.int32)[None, :, None]  # (1,H_out,1)
        w_idx_base = tf.range(W_out, dtype=tf.int32)[None, None, :]  # (1,1,W_out)

        # -----------------------------
        # Main accumulation loop (sum overlaps like nn.Fold)
        # -----------------------------
        for kh in range(kH):
            for kw in range(kW):
                # Offset inside effective kernel
                h_off = kh * dH
                w_off = kw * dW

                # Target pixel coordinates for this kernel position
                H_idx = h_off + h_idx_base * sH  # (1,H_out,1), int32
                W_idx = w_off + w_idx_base * sW  # (1,1,W_out), int32

                # Values: (N,C,H_out,W_out)
                patch_vals = cols[:, :, kh, kw]
                pv_flat = tf.reshape(patch_vals, [-1])

                # -------------------------
                # Build index tensor (N*C*H_out*W_out, 4)
                # All int32 to avoid concat/type issues
                # -------------------------
                # N indices
                NN = tf.repeat(
                    tf.range(N, dtype=tf.int32),
                    C * H_out * W_out
                )[:, None]  # (?,1)

                # C indices
                CC = tf.tile(
                    tf.repeat(
                        tf.range(C, dtype=tf.int32),
                        H_out * W_out
                    )[None],
                    [N, 1]
                )
                CC = tf.reshape(CC, [-1, 1])  # (?,1)

                # H indices
                HH_base = tf.repeat(tf.reshape(H_idx, [-1]), W_out)  # (H_out*W_out,)
                HH = tf.tile(HH_base[None], [N * C, 1])
                HH = tf.reshape(HH, [-1, 1])  # (?,1)
                HH = tf.cast(HH, tf.int32)

                # W indices
                WW_base = tf.reshape(W_idx, [-1])  # (W_out,)
                WW = tf.tile(WW_base, [N * C * H_out])  # (N*C*H_out*W_out,)
                WW = tf.reshape(WW, [-1, 1])  # (?,1)
                WW = tf.cast(WW, tf.int32)

                idx = tf.concat([NN, CC, HH, WW], axis=1)  # (?,4), int32

                # -------------------------
                # Scatter-add into padded output (SUM, no normalization)
                # -------------------------
                out_pad = tf.tensor_scatter_nd_add(out_pad, idx, pv_flat)

        # Crop padding, keep NCHW (matches torch.nn.Fold)
        out_nchw = out_pad[:, :, pt:pt + H_img, pl:pl + W_img]
        # If you later want NHWC for Keras, uncomment:
        # out_nhwc = tf.transpose(out_nchw, (0, 2, 3, 1))
        # return out_nhwc

        return out_nchw

    # Wrap in a Lambda and return a KERASTENSOR
    out = tf.keras.layers.Lambda(_col2im_lambda, name=keras_name)(input_cols)
    layers[node_name] = out

