import numpy as np
import tensorflow as tf
from typing import Callable
from keras import backend as K
import logging
logger = logging.getLogger('onnx2keras.tfops_funcs')

layer_names_counter = {}


def named_tfop(func: Callable):
    def wrapped_function(*args, tf_name=None, **kwargs):
        result = func(*args, **kwargs)
        if tf_name is None or tf_name=="" or not isinstance(tf_name, str):
            raise ValueError(f"The layer {result.node.layer} with name"
                             f" {result.node.layer.name} was provided with an empty or None Name")
        if not isinstance(result, (tf.Tensor, np.ndarray)):
            if tf_name not in layer_names_counter:
                layer_names_counter[tf_name] = 0
            else:
                layer_names_counter[tf_name] = layer_names_counter[tf_name] + 1
                tf_name = tf_name + f"_{layer_names_counter[tf_name]}"
                if not isinstance(result, (tf.Tensor, np.ndarray)):
                    logger.debug(f"The op {result.node.layer.symbol} with name"
                                     f"{result.node.layer.name} has a duplicate name {tf_name}")
            result.node.layer._name = tf_name
        return result
    return wrapped_function


tf_cast = named_tfop(tf.cast)
tf_shape = named_tfop(tf.shape)
tf_math_abs = named_tfop(tf.math.abs)
tf_reshape = named_tfop(tf.reshape)
tf_stack = named_tfop(tf.stack)
tf_add = named_tfop(tf.add)
tf_image_resize = named_tfop(tf.image.resize)
tf_multiply = named_tfop(tf.multiply)
tf_clip_by_value = named_tfop(tf.clip_by_value)
tf_math_negative = named_tfop(tf.math.negative)
tf_tensor_scatter_nd_update = named_tfop(tf.tensor_scatter_nd_update)
K_mean = named_tfop(K.mean)
tf_math_reduce_prod = named_tfop(tf.math.reduce_prod)
tf_math_reduce_min = named_tfop(tf.math.reduce_min)
tf_math_pow = named_tfop(tf.math.pow)
tf_math_sqrt = named_tfop(tf.math.sqrt)
tf_strided_slice = named_tfop(tf.strided_slice)
tf_squeeze = named_tfop(tf.squeeze)
tf_argmax = named_tfop(tf.argmax)
tf_expand_dims = named_tfop(tf.expand_dims)
tf_maximum = named_tfop(tf.maximum)
tf_minimum = named_tfop(tf.minimum)
tf_repeat = named_tfop(tf.repeat)
tf_matmul = named_tfop(tf.matmul)
tf_concat = named_tfop(tf.concat)
tf_transpose = named_tfop(tf.transpose)
tf_math_reduce_variance = named_tfop(tf.math.reduce_variance)
tf_math_reduce_mean = named_tfop(tf.math.reduce_mean)
tf_sqrt = named_tfop(tf.sqrt)
tf_where = named_tfop(tf.where)
tf_gather = named_tfop(tf.gather)
tf_range = named_tfop(tf.range)
tf_abs = named_tfop(tf.abs)
tf_reduce_sum = named_tfop(tf.reduce_sum)
tf_pad = named_tfop(tf.pad)
tf_math_erf = named_tfop(tf.math.erf)
tf_math_reciprocal = named_tfop(tf.math.reciprocal)
tf_logical_not = named_tfop(tf.logical_not)
tf_equal = named_tfop(tf.equal)
tf_tile = named_tfop(tf.tile)
tf_math_minimum = named_tfop(tf.math.minimum)
tf_math_maximum = named_tfop(tf.math.maximum)
tf_math_sign = named_tfop(tf.math.sign)
tf_math_sin = named_tfop(tf.math.sin)
tf_math_cosh = named_tfop(tf.math.cosh)
tf_math_ceil = named_tfop(tf.math.ceil)
tf_math_acosh = named_tfop(tf.math.acosh)
tf_math_acos = named_tfop(tf.math.acos)
tf_math_asinh = named_tfop(tf.math.asinh)
tf_math_asin = named_tfop(tf.math.asin)
tf_math_atanh = named_tfop(tf.math.atanh)
tf_math_tan = named_tfop(tf.math.tan)
tf_math_atan = named_tfop(tf.math.atan)
tf_math_sinh = named_tfop(tf.math.sinh)
tf_math_less_equal = named_tfop(tf.math.less_equal)
tf_bitwise_invert = named_tfop(tf.bitwise.invert)
tf_bitwise_bitwise_and = named_tfop(tf.bitwise.bitwise_and)
tf_bitwise_bitwise_or = named_tfop(tf.bitwise.bitwise_or)
tf_bitwise_bitwise_xor = named_tfop(tf.bitwise.bitwise_xor)
tf_cos = named_tfop(tf.cos)
tf_math_greater = named_tfop(tf.math.greater)
tf_math_greater_equal = named_tfop(tf.math.greater_equal)
tf_logical_and = named_tfop(tf.logical_and)
tf_math_logical_xor = named_tfop(tf.math.logical_xor)
tf_math_logical_or = named_tfop(tf.math.logical_or)
tf_argmin = named_tfop(tf.argmin)
tf_one_hot = named_tfop(tf.one_hot)
tf_round = named_tfop(tf.round)
tf_math_cumsum = named_tfop(tf.math.cumsum)
tf_math_is_inf = named_tfop(tf.math.is_inf)
tf_math_is_nan = named_tfop(tf.math.is_nan)
tf_size = named_tfop(tf.size)
tf_linalg_det = named_tfop(tf.linalg.det)
tf_not_equal = named_tfop(tf.not_equal)
tf_gather_nd = named_tfop(tf.gather_nd)
tf_math_softplus = named_tfop(tf.math.softplus)
tf_math_tanh = named_tfop(tf.math.tanh)
tf_signal_irfft = named_tfop(tf.signal.irfft)
tf_signal_ifft = named_tfop(tf.signal.ifft)
tf_signal_rfft = named_tfop(tf.signal.rfft)
tf_signal_fft = named_tfop(tf.signal.fft)
tf_sign = named_tfop(tf.sign)
tf_abs = named_tfop(tf.abs)
tf_math_mod = named_tfop(tf.math.mod)
tf_bitwise_left_shift = named_tfop(tf.bitwise.left_shift)
tf_bitwise_right_shift = named_tfop(tf.bitwise.right_shift)
tf_rank = named_tfop(tf.rank)
tf_fill = named_tfop(tf.fill)
tf_image_non_max_suppression = named_tfop(tf.image.non_max_suppression)
tf_ones_like = named_tfop(tf.ones_like)
tf_image_crop_and_resize = named_tfop(tf.image.crop_and_resize)
tf_ones = named_tfop(tf.ones)
tf_math_floor = named_tfop(tf.math.floor)
tf_zeros_like = named_tfop(tf.zeros_like)
tf_tensor_scatter_nd_update = named_tfop(tf.tensor_scatter_nd_update)
tf_nn_avg_pool = named_tfop(tf.nn.avg_pool)
tf_nn_max_pool = named_tfop(tf.nn.max_pool)
tf_linalg_matmul = named_tfop(tf.linalg.matmul)