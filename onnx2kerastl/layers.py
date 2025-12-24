from .convolution_layers import convert_conv, convert_convtranspose
from .activation_layers import convert_relu, convert_elu, convert_lrelu, convert_selu, \
    convert_sigmoid, convert_tanh, convert_softmax, convert_prelu, convert_hard_sigmoid, convert_erf, convert_soft_plus, \
    convert_soft_sign, convert_mish, convert_gelu, convert_hard_swish
from .ltsm_layers import convert_lstm, convert_gru
from .operation_layers import convert_clip, convert_exp, convert_neg, convert_reduce_sum, convert_reduce_mean, \
    convert_log, convert_pow, convert_sqrt, convert_split, convert_cast, convert_floor, convert_identity, \
    convert_argmax, convert_reduce_l2, convert_reduce_max, convert_reciprocal, convert_abs, convert_not, convert_cosine, \
    convert_less, convert_less_equal, convert_and, convert_greater, convert_greater_equal, convert_xor, convert_or, \
    convert_trilu, convert_sign, convert_cosh, convert_sin, convert_sinh, convert_ceil, convert_acosh, convert_acos, \
    convert_asinh, convert_asin, convert_atanh, convert_atan, convert_bitwise_and, convert_argmin, convert_bitwise_xor, \
    convert_bitwise_or, convert_tan, convert_cumsum, convert_bitwise_not, convert_reduce_prod, convert_reduce_min, \
    convert_is_inf, convert_is_nan, convert_size, convert_non_zero, convert_gather_nd, convert_nms, convert_if, \
    convert_einsum
from .elementwise_layers import convert_elementwise_div, convert_elementwise_add, convert_elementwise_mul, \
    convert_elementwise_sub, convert_max, convert_min, convert_mean, convert_equal, convert_where, convert_scatter_nd, \
    convert_round, convert_mod, convert_bitshift
from .linear_layers import convert_gemm, convert_det
from .reshape_layers import convert_transpose, convert_shape, convert_gather, convert_unsqueeze, \
    convert_concat, convert_reshape, convert_flatten, convert_slice, convert_squeeze, convert_expand, convert_resize, \
    convert_tile, convert_gather_elements, col2im_onnx
from .constant_layers import convert_constant, convert_constant_of_shape, convert_one_hot
from .normalization_layers import convert_batchnorm, convert_instancenorm, convert_dropout, convert_lrn, convert_layernorm
from .pooling_layers import convert_avgpool, convert_global_max_pool, convert_maxpool, convert_global_avg_pool, convert_topk, convert_roi_align
from .padding_layers import convert_padding
from .upsampling_layers import convert_upsample
from .caffe2_layers import convert_alias_with_name, convert_resize_nearest
from .sampling_layers import convert_gridsample, convert_range, convert_unique
from .fft_layers import convert_dft

AVAILABLE_CONVERTERS = {
    'Abs': convert_abs,
    'AliasWithName': convert_alias_with_name,
    'Conv': convert_conv,
    'ConvTranspose': convert_convtranspose,
    'Relu': convert_relu,
    'Resize': convert_resize,
    'Elu': convert_elu,
    'LeakyRelu': convert_lrelu,
    'Sigmoid': convert_sigmoid,
    'HardSigmoid': convert_hard_sigmoid,
    'Tanh': convert_tanh,
    'Selu': convert_selu,
    'Clip': convert_clip,
    'Exp': convert_exp,
    'Neg': convert_neg,
    'Log': convert_log,
    'Softmax': convert_softmax,
    "ScatterND": convert_scatter_nd,
    'PRelu': convert_prelu,
    'ReduceMax': convert_reduce_max,
    'ReduceSum': convert_reduce_sum,
    'ReduceMean': convert_reduce_mean,
    'ReduceProd': convert_reduce_prod,
    'ReduceMin': convert_reduce_min,
    'Pow': convert_pow,
    'Slice': convert_slice,
    'Squeeze': convert_squeeze,
    'Expand': convert_expand,
    'Sqrt': convert_sqrt,
    'Split': convert_split,
    'Cast': convert_cast,
    'Floor': convert_floor,
    'Identity': convert_identity,
    'ArgMax': convert_argmax,
    'ReduceL2': convert_reduce_l2,
    'Max': convert_max,
    'Min': convert_min,
    'Mean': convert_mean,
    'Div': convert_elementwise_div,
    'Add': convert_elementwise_add,
    'Sum': convert_elementwise_add,
    'Mul': convert_elementwise_mul,
    'Sub': convert_elementwise_sub,
    'Gemm': convert_gemm,
    'MatMul': convert_gemm,
    'Transpose': convert_transpose,
    'Constant': convert_constant,
    'BatchNormalization': convert_batchnorm,
    'InstanceNormalization': convert_instancenorm,
    'Dropout': convert_dropout,
    'LRN': convert_lrn,
    'MaxPool': convert_maxpool,
    'GlobalMaxPool': convert_global_max_pool,
    'AveragePool': convert_avgpool,
    'GlobalAveragePool': convert_global_avg_pool,
    'Shape': convert_shape,
    'Gather': convert_gather,
    'Unsqueeze': convert_unsqueeze,
    'Concat': convert_concat,
    'Reshape': convert_reshape,
    'ResizeNearest': convert_resize_nearest,
    'Pad': convert_padding,
    'Flatten': convert_flatten,
    'Upsample': convert_upsample,
    'Erf': convert_erf,
    'Reciprocal': convert_reciprocal,
    'ConstantOfShape': convert_constant_of_shape,
    'Equal': convert_equal,
    'Where': convert_where,
    'LSTM': convert_lstm,
    'GRU': convert_gru,
    'Tile': convert_tile,
    'GridSample': convert_gridsample,
    'Range': convert_range,
    'Not': convert_not,
    'Less': convert_less,
    'Sign': convert_sign,
    'Cosh': convert_cosh,
    'Sin': convert_sin,
    'Sinh': convert_sinh,
    'LessOrEqual': convert_less_equal,
    "And": convert_and,
    "Greater": convert_greater,
    "GreaterOrEqual": convert_greater_equal,
    "Xor": convert_xor,
    "Or": convert_or,
    'Cos': convert_cosine,
    "Trilu": convert_trilu,
    "Ceil": convert_ceil,
    "Acosh": convert_acosh,
    "Acos": convert_acos,
    "Asinh": convert_asinh,
    "Asin": convert_asin,
    "Atanh": convert_atanh,
    "Atan": convert_atan,
    "BitwiseAnd": convert_bitwise_and,
    "BitwiseOr": convert_bitwise_or,
    "BitwiseXor": convert_bitwise_xor,
    "BitwiseNot": convert_bitwise_not,
    "ArgMin": convert_argmin,
    "OneHot": convert_one_hot,
    "Round": convert_round,
    "Tan": convert_tan,
    "CumSum": convert_cumsum,
    "IsInf": convert_is_inf,
    "IsNaN": convert_is_nan,
    "Size": convert_size,
    "Det": convert_det,
    "NonZero": convert_non_zero,
    "GatherND": convert_gather_nd,
    "Softplus": convert_soft_plus,
    "Softsign": convert_soft_sign,
    "Mish": convert_mish,
    "Gelu": convert_gelu,
    "HardSwish": convert_hard_swish,
    "DFT": convert_dft,
    "Mod": convert_mod,
    "BitShift": convert_bitshift,
    "TopK": convert_topk,
    'GatherElements': convert_gather_elements,
    'NonMaxSuppression': convert_nms,
    'Unique': convert_unique,
    'If': convert_if,
    'RoiAlign': convert_roi_align,
    'Einsum': convert_einsum,
    'LayerNormalization': convert_layernorm,
    'Col2Im': col2im_onnx,
}
