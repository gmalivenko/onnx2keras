from .convolution_layers import convert_conv, convert_convtranspose
from .activation_layers import convert_relu, convert_lrelu, convert_selu, \
    convert_sigmoid, convert_tanh
from .operation_layers import convert_clip, convert_exp, convert_reduce_sum, convert_reduce_mean, convert_log
from .elementwise_layers import convert_elementwise_div, convert_elementwise_add, convert_elementwise_mul, convert_elementwise_sub
from .linear_layers import convert_gemm
from .reshape_layers import convert_transpose, convert_shape, convert_gather, convert_unsqueeze, convert_concat, convert_reshape, convert_flatten
from .constant_layers import convert_constant
from .normalization_layers import convert_batchnorm, convert_instancenorm
from .pooling_layers import convert_avgpool, convert_maxpool, convert_global_avg_pool
from .padding_layers import convert_padding


AVAILABLE_CONVERTERS = {
    'Conv': convert_conv,
    'ConvTranspose': convert_convtranspose,
    'Relu': convert_relu,
    'LeakyRelu': convert_lrelu,
    'Sigmoid': convert_sigmoid,
    'Tanh': convert_tanh,
    'Selu': convert_selu,
    'Clip': convert_clip,
    'Exp': convert_exp,
    'Log': convert_log,
    'ReduceSum': convert_reduce_sum,
    'ReduceMean': convert_reduce_mean,
    'Div': convert_elementwise_div,
    'Add': convert_elementwise_add,
    'Mul': convert_elementwise_mul,
    'Sub': convert_elementwise_sub,
    'Gemm': convert_gemm,
    'MatMul': convert_gemm,
    'Transpose': convert_transpose,
    'Constant': convert_constant,
    'BatchNormalization': convert_batchnorm,
    'InstanceNormalization': convert_instancenorm,
    'MaxPool': convert_maxpool,
    'AveragePool': convert_avgpool,
    'GlobalAveragePool': convert_global_avg_pool,
    'Shape': convert_shape,
    'Gather': convert_gather,
    'Unsqueeze': convert_unsqueeze,
    'Concat': convert_concat,
    'Reshape': convert_reshape,
    'Pad': convert_padding,
    'Flatten': convert_flatten
}
