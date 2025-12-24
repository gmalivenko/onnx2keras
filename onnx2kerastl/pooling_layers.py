import keras
import logging

from .utils import ensure_tf_type, is_numpy
from .tfops_funcs import tf_reshape, tf_rank, tf_concat, tf_shape, tf_cast, tf_image_crop_and_resize, tf_ones, \
    tf_nn_avg_pool, tf_nn_max_pool
import numpy as np
import string
import random
import tensorflow as tf
import keras.backend as K


def convert_maxpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert MaxPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.maxpool')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]
    pad = 'valid'

    if all([shape % 2 == 1 for shape in kernel_shape]) and \
            all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
            all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = f"{params['cleaned_name']}_maxpool" + '_pad'
        if len(kernel_shape) == 2:
            padding = None

            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))

            if padding is not None:
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=padding,
                    name=padding_name
                )
                layers[padding_name] = input_0 = padding_layer(input_0)
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
    if len(kernel_shape) == 2:
        pooling = keras.layers.MaxPooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=f"{params['cleaned_name']}_maxpool",
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.MaxPooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=f"{params['cleaned_name']}_maxpool",
            data_format='channels_first'
        )
    ceil_mode = params.get('ceil_mode', False)
    if ceil_mode:
        if pad == 'valid':
            output_shape = ((np.array(input_0.shape[-len(kernel_shape):]) - np.array(kernel_shape)) / np.array(
                stride_shape)) + 1
        else:
            output_shape = np.floor((np.array(input_0.shape[-len(kernel_shape):]) - 1) / np.array(stride_shape)) + 1
        if not np.array([output_shape[i].is_integer() for i in range(len(output_shape))]).all():
            padding = [0 if output_shape[i].is_integer() else stride_shape[i] for i in range(len(kernel_shape))]
            rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
            if len(kernel_shape) == 2:
                layers[node_name + "_pre_" + rand_string] = keras.layers.ZeroPadding2D(
                    ((0, padding[0]), (0, padding[1])), name=f"{params['cleaned_name']}_pre")(input_0)
            else:
                layers[node_name + "_pre_" + rand_string] = keras.layers.ZeroPadding3D(
                    ((0, padding[0]), (0, padding[1]), (0, padding[2])), name=f"{params['cleaned_name']}_pre")(input_0)
            input_0 = layers[node_name + "_pre_" + rand_string]
    layers[node_name] = pooling(input_0)

def convert_global_max_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalMaxPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    tensor_dim = len(input_0.shape)
    if tensor_dim == 3:
        global_pool = keras.layers.GlobalMaxPooling1D(data_format='channels_first',
                                                      name=f"{params['cleaned_name']}_global_max_pool_3")
    elif tensor_dim == 4:
        global_pool = keras.layers.GlobalMaxPooling2D(data_format='channels_first',
                                                      name=f"{params['cleaned_name']}_global_max_pool_4")
    elif tensor_dim == 5:
        global_pool = keras.layers.GlobalMaxPooling3D(data_format='channels_first',
                                                      name=f"{params['cleaned_name']}_global_max_pool_5")
    else:
        raise NotImplementedError("Global max pooling of dims < 3 or dims > 5 is not supported")
    input_0 = global_pool(input_0)
    new_shape = input_0.shape.as_list()
    new_shape = new_shape[1:]
    new_shape.extend([1] * (tensor_dim - 2))
    reshape_layer = keras.layers.Reshape(new_shape, name=f"{params['cleaned_name']}_global_max_pool_reshape")
    input_0 = reshape_layer(input_0)

    layers[node_name] = input_0


def convert_avgpool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert AvgPooling layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.avgpool')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    kernel_shape = params['kernel_shape']
    stride_shape = params['strides']

    pads = params['pads'] if 'pads' in params else [0, 0, 0, 0, 0, 0]

    if not any(pads):
        pad = 'valid'

    elif all([shape % 2 == 1 for shape in kernel_shape]) and \
            all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
            all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        logger.debug('Use `same` padding parameters.')
    else:
        pad = 'valid'
        logger.warning('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = f"{params['cleaned_name']}_avgpool" + '_pad'
        if len(kernel_shape) == 2:
            padding_layer = keras.layers.ZeroPadding2D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
        layers[padding_name] = input_0 = padding_layer(input_0)

    if len(kernel_shape) == 2:
        pooling = keras.layers.AveragePooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=f"{params['cleaned_name']}_avgpool",
            data_format='channels_first'
        )
    elif len(kernel_shape) == 1:
        pooling = keras.layers.AveragePooling1D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=f"{params['cleaned_name']}_avgpool",
            data_format='channels_first'
        )
    else:
        pooling = keras.layers.AveragePooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
            name=f"{params['cleaned_name']}_avgpool",
            data_format='channels_first'
        )
    layers[node_name] = pooling(input_0)


def convert_global_avg_pool(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert GlobalAvgPool layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    tensor_dim = len(input_0.shape)
    if tensor_dim == 3:
        global_pool = keras.layers.GlobalAveragePooling1D(data_format='channels_first',
                                                          name=f"{params['cleaned_name']}_global_avg_pool_3",
                                                          keepdims=True)
    elif tensor_dim == 4:
        global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first',
                                                          name=f"{params['cleaned_name']}_global_avg_pool_4",
                                                          keepdims=True)
    elif tensor_dim == 5:
        global_pool = keras.layers.GlobalAveragePooling3D(data_format='channels_first',
                                                          name=f"{params['cleaned_name']}_global_avg_pool_5",
                                                          keepdims=True)
    else:
        raise NotImplementedError("Global average pooling of dims < 3 or dims > 5 is not supported")
    input_0 = global_pool(input_0)
    layers[node_name] = input_0


def convert_topk(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert topk layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    axis = params.get('axis', -1)
    largest = bool(params.get('largest', 1))
    to_sort = bool(params.get('sorted', 1))
    x = layers[node.input[0]]
    k = layers[node.input[1]][0]
    if not is_numpy(k) and not K.is_keras_tensor(k):  # Eager tensor does not serialize well
        k = k.numpy().astype(np.int32)
    if not largest:
        in_tensor = -x
    else:
        in_tensor = x

    def target_layer(composed_input, to_sort=to_sort, axis=axis):
        in_tensor = composed_input[..., :-1]
        k = composed_input[..., -1]
        for i in range(len(k.shape)):
            k = k[0]
        k = tf.cast(k, tf.int32)
        rank = len(in_tensor.shape)

        if axis >= rank - 1 or axis == -1:
            permuted = in_tensor
        else:
            ord_permute = np.arange(rank)
            ord_permute[axis] = rank - 1
            ord_permute[-1] = axis
            permuted = tf.transpose(in_tensor, ord_permute)

        topk_res = tf.math.top_k(permuted, k=k, sorted=to_sort)
        values_pre_permute = topk_res[0]
        indices_pre_permute = topk_res[1]
        topk_concat = tf.stack([values_pre_permute, tf.cast(indices_pre_permute, tf.float32)])
        if axis >= rank - 1 or axis == -1:
            out = topk_concat
        else:
            ord_permute = [0] + (ord_permute + 1).tolist()
            out = tf.transpose(topk_concat, ord_permute)
        return out
    in_shape = tf_shape(in_tensor, tf_name=f"{params['cleaned_name']}_topk_in_shape")
    k_needed_shape_possible_keras_tensor = tf_concat(
            [(in_shape)[:-1],[1]], axis=-1,tf_name=f"{params['cleaned_name']}_topk_k_needed_shape")

    if hasattr(k_needed_shape_possible_keras_tensor, "_inferred_value"): #is keras tensor
        k_needed_shape = k_needed_shape_possible_keras_tensor._inferred_value
    else:
        k_needed_shape = k_needed_shape_possible_keras_tensor
    k_unsqueezed = tf_ones(k_needed_shape, tf_name=f"{params['cleaned_name']}_topk_k_shape")*\
                   tf_cast(k, tf.float32, tf_name=f"{params['cleaned_name']}_topk_k_cast")
    k_reshaped = tf_cast(k_unsqueezed, in_tensor.dtype, tf_name=f"{params['cleaned_name']}_topk_k_reshaped")
    composed_input = tf_concat([in_tensor, k_reshaped], axis=-1,
                               tf_name=f"{params['cleaned_name']}_topk_k_concat")
    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_topk")
    result = lambda_layer(composed_input)
    pos_axis = axis if axis > 0 else in_shape.shape[0]-1
    new_shape = tf_concat([in_shape[:pos_axis], [k], in_shape[pos_axis+1:]], axis=-1,
                          tf_name=f"{params['cleaned_name']}_topk_output_shape")
    values = tf_reshape(result[0], new_shape,
                        tf_name=f"{params['cleaned_name']}_topk_values_reshape")
    indices = tf_reshape(tf_cast(result[1],
                                 tf.int32,
                                 tf_name=f"{params['cleaned_name']}_topk_indices_cast"),
                         new_shape,
                         tf_name=f"{params['cleaned_name']}_topk_indices_reshape")
    if not largest:
        out_tensor = -values
    else:
        out_tensor = values
    layers[params['_outputs'][0]] = out_tensor
    layers[params['_outputs'][1]] = indices


def convert_roi_align(node, params, layers, lambda_func, node_name, keras_name):
    # extract params
    output_height = params.get('output_height', 1)
    output_width = params.get('output_width', 1)
    sampling_ratio = params.get('sampling_ratio', 0)
    spatial_scale = params.get('spatial_scale', 1.0)
    mode = params.get('mode', 'avg')
    if isinstance(mode, bytes):
        mode = mode.decode('utf-8')

    feature_map = layers[node.input[0]]
    rois = layers[node.input[1]]
    batch_indices = layers[node.input[2]]

    adaptive_ratio = False
    if sampling_ratio <= 0:
        sampling_ratio = int((output_height + output_width) / 2)
        adaptive_ratio = True

    rois = rois * spatial_scale
    box_ind = tf_cast(batch_indices, tf.int32, tf_name=f"{params['cleaned_name']}_roi_cast_batch")
    if keras.backend.image_data_format() == 'channels_first':
        fm_shape = tf_shape(feature_map, tf_name=f"{params['cleaned_name']}_roi_hw")[2:]  # H, W
    else:
        raise NotImplementedError("To support channels_last in RoiAlign - need to remove permutes")
    # extract inputs
    x0 = rois[:, 0:1]
    y0 = rois[:, 1:2]
    x1 = rois[:, 2:3]
    y1 = rois[:, 3:4]
    if not adaptive_ratio:
        crop_shape = (
            output_height * sampling_ratio,
            output_width * sampling_ratio,
        )
        spacing_w = (x1 - x0) / tf_cast(crop_shape[1], dtype=tf.float32,
                                        tf_name=f"{params['cleaned_name']}_roi_cast_crop1")
        spacing_h = (y1 - y0) / tf_cast(crop_shape[0], dtype=tf.float32,
                                        tf_name=f"{params['cleaned_name']}_roi_cast_crop0")
        nx0 = (x0 + spacing_w / 2) / tf_cast(fm_shape[1] - 1, dtype=tf.float32,
                                             tf_name=f"{params['cleaned_name']}_roi_cast_fm_1")
        ny0 = (y0 + spacing_h / 2) / tf_cast(fm_shape[0] - 1, dtype=tf.float32,
                                             tf_name=f"{params['cleaned_name']}_roi_cast_fm_0")

        nw = spacing_w * tf_cast(
            crop_shape[1] - 1,
            dtype=tf.float32,
            tf_name=f"{params['cleaned_name']}_roi_cast_crop_2"
        ) / tf_cast(
            fm_shape[1] - 1,
            dtype=tf.float32,
            tf_name=f"{params['cleaned_name']}_roi_cast_crop_3"
        )
        nh = spacing_h * tf_cast(
            crop_shape[0] - 1,
            dtype=tf.float32,
            tf_name=f"{params['cleaned_name']}_roi_cast_crop_3"
        ) / tf_cast(
            fm_shape[0] - 1,
            dtype=tf.float32,
            tf_name=f"{params['cleaned_name']}_roi_cast_crop_4"
        )
    else:
        roi_width = x1 - x0
        roi_height = y1 - y0
        fm_shape_1 = tf_cast(fm_shape[1] - 1, dtype=tf.float32, tf_name=f"{params['cleaned_name']}_roi_cast_fm_1")
        fm_shape_0 = tf_cast(fm_shape[0] - 1, dtype=tf.float32, tf_name=f"{params['cleaned_name']}_roi_cast_fm_0")
        nx0 = x0 / fm_shape_1
        ny0 = y0 / fm_shape_0
        nw = roi_width / fm_shape_1
        nh = roi_height / fm_shape_0

    boxes = tf_concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1,
                      tf_name=f"{params['cleaned_name']}_roi_concat"
                      )

    permuted_features = keras.layers.Permute([2, 3, 1],
                                             name=f"{params['cleaned_name']}_roi_channels_last"
                                             )(feature_map)  # move to channels last
    cropped_tensor = tf_image_crop_and_resize(
        permuted_features,
        boxes,
        tf.cast(box_ind, dtype=tf.int32),
        crop_size=(
            output_height * sampling_ratio,
            output_width * sampling_ratio,
        ),
        method='bilinear',
        tf_name=f"{params['cleaned_name']}_roi_crop_resize"
    )

    pooled_tensor = None
    if mode.lower() == 'avg':
        pooled_tensor = tf_nn_avg_pool(
            input=cropped_tensor,
            ksize=[1, sampling_ratio, sampling_ratio, 1],
            strides=[1, sampling_ratio, sampling_ratio, 1],
            padding='SAME',
            tf_name=f"{params['cleaned_name']}_roi_avg_pool",
            data_format='NHWC'
        )
    elif mode.lower() == 'max':
        pooled_tensor = tf_nn_max_pool(
            input=cropped_tensor,
            ksize=[1, sampling_ratio, sampling_ratio, 1],
            strides=[1, sampling_ratio, sampling_ratio, 1],
            padding='SAME',
            tf_name=f"{params['cleaned_name']}_roi_max_pool",
            data_format='NHWC'
        )
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
    layers[node_name] = keras.layers.Permute([3, 1, 2],
                                             name=f"{params['cleaned_name']}_roi_channels_first")(pooled_tensor)
