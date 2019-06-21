from .convolution_layers import convert_conv, convert_convtranspose

AVAILABLE_CONVERTERS = {
    'Conv': convert_conv,
    'ConvTranspose': convert_convtranspose,
}
