import keras
import tensorflow as tf


def convert_range(node, params, layers, lambda_func, node_name, keras_name):
    start_range = layers[node.input[0]]
    limit_range = layers[node.input[1]]
    delta_range = layers[node.input[2]]
    layers[node_name] = tf.range(start_range, limit_range, delta_range)


def convert_gridsample(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert gridsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    assert params['mode'].decode('ascii') == 'bilinear'
    assert params['padding_mode'].decode('ascii') == 'zeros'
    assert params['align_corners'] == 1
    params['mode'] = params['mode'].decode('ascii')
    params['padding_mode'] = params['padding_mode'].decode('ascii')
    img = layers[node.input[0]]
    sample_grid = layers[node.input[1]]
    torch_shape = tf.shape(img)
    max_xy = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([torch_shape[3] - 1, torch_shape[2] - 1]), 0), 0), 0)
    max_xy = tf.cast(max_xy, tf.float32)
    grid_index_coords = 0.5 * (sample_grid + 1.) * max_xy  # transform from [-1,1] to [0,H-1]/[0,W-1]
    grid_index_coords = grid_index_coords + 1  # fix locs considering we add padding
    orig_query_shape = tf.shape(grid_index_coords)
    query_points = tf.reshape(grid_index_coords, [orig_query_shape[0], -1, 2])
    padded_img = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format="channels_first")(img)
    grid = tf.keras.layers.Permute((2, 3, 1))(padded_img)
    indexing = 'ji'
    grid_shape = tf.shape(grid)
    query_shape = tf.shape(query_points)
    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )
    num_queries = query_shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    # unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

    for i, dim in enumerate(index_order):
        queries = query_points[:, :, dim]
        # queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.math.minimum(
            tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
        )
        int_floor = tf.cast(floor, tf.dtypes.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)

        flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = tf.reshape(
            tf.range(batch_size) * height * width, [batch_size, 1]
        )

    # This wraps tf.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using tf.gather_nd.
    def gather(y_coords, x_coords, name=None):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top
    tf_reshaped_results = tf.reshape(interp, tf.concat([orig_query_shape[:-1], torch_shape[1:2]], axis=0))
    ret = tf.keras.layers.Permute((3, 1, 2))(tf_reshaped_results)
    layers[node_name] = ret


def convert_unique(node, params, layers, lambda_func, node_name, keras_name):
    to_sort = params.get('sorted', 1) == 1
    axis = params.get('axis')
    if axis is not None:
        raise AttributeError("Onnx2kerras: Unique does not currently support an operation on a non-flattened array")
    lambda_input = layers[node.input[0]]
    rev_idx_length = tf.size(lambda_input)

    def target_layer(x):
        input_keras = x
        if axis is None:
            input_final = tf.reshape(input_keras, [-1])
        res, rev_idx, count = tf.unique_with_counts(input_final)
        idx = tf.math.unsorted_segment_min(tf.range(tf.shape(rev_idx)[0]), rev_idx, tf.shape(res)[0])
        if to_sort:
            linspace = tf.range(tf.shape(count)[0])
            argsorted = tf.argsort(res)
            lookup_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(linspace, argsorted),
                                                     default_value=-1)

            rev_idx_sorted = lookup_table.lookup(rev_idx)

            res_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), res, tf.shape(res))
            count_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), count, tf.shape(res))
            idx_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), idx, tf.shape(res))
            return tf.concat([tf.cast(rev_idx_sorted, tf.float32), res_sorted, tf.cast(idx_sorted, tf.float32), tf.cast(count_sorted, tf.float32)],
                      axis=0)
        else:
            return tf.concat([tf.cast(rev_idx, tf.float32), res, tf.cast(idx, tf.float32), tf.cast(count, tf.float32)],
                             axis=0)
    lambda_layer = keras.layers.Lambda(target_layer, name='lambda' + "_".join(keras_name))
    lambda_res = lambda_layer(lambda_input)
    rev_idx = lambda_res[:rev_idx_length]
    lambda_length = tf.size(lambda_res)
    remainder = tf.cast((lambda_length-rev_idx_length)/3, tf.int32)          #not working need to fix
    count = tf.cast(lambda_res[-remainder:], tf.int32)
    idx = tf.cast(lambda_res[-2*remainder:-remainder], tf.int32)
    res = tf.cast(lambda_res[-3*remainder:-2*remainder], tf.int32)
    layers[keras_name[0]] = res
    layers[keras_name[1]] = idx
    layers[keras_name[2]] = rev_idx
    layers[keras_name[3]] = count
