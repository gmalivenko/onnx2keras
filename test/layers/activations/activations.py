import torch
import numpy as np
import onnx
import os

from onnx2keras import onnx_to_keras, check_torch_keras_error

from relu import LayerReLUTest, FReLUTest
from hard_tanh import LayerHardtanhTest, FHardtanhTest
from leaky_relu import LayerLeakyReLUTest, FLeakyReLUTest
from selu import LayerSELUTest, FSELUTest
from sigmoid import LayerSigmoidTest, FSigmoidTest
from tanh import LayerTanhTest, FTanhTest
from log_sigmoid import LayerLogSigmoidTest, FLogSigmoidTest
# from threshold import LayerThresholdTest, FThresholdTest  # Not Supported by ONNX
from relu6 import LayerReLU6Test, FReLU6Test
from softmax import LayerSoftmaxTest, FSoftmaxTest
from prelu import LayerPReLUTest, FPReLUTest
from elu import LayerELUTest, FPELUTest
# from log_softmax import LayerLogSoftmaxTest, FLogSoftmaxTest  # Not Supported by ONNX


# TODO:
# Threshold, Softmax2d, LogSoftmax, CELU, Hardshrink,  \
# Softplus, Softshrink, MultiheadAttention, Softsign, Softmin, Tanhshrink, RReLU, GLU


if __name__ == '__main__':
    max_error = 0
    for act_type in [
                     # LayerLogSoftmaxTest, FLogSoftmaxTest,  # Not Supported by ONNX
                     LayerPReLUTest, FPReLUTest,
                     LayerSoftmaxTest, FSoftmaxTest,
                     LayerReLU6Test, FReLU6Test,
                     # LayerThresholdTest, FThresholdTest, # Not Supported by ONNX
                     LayerLogSigmoidTest, FLogSigmoidTest,
                     LayerTanhTest, FTanhTest,
                     LayerSigmoidTest, FSigmoidTest,
                     LayerSELUTest, FSELUTest,
                     LayerLeakyReLUTest, FLeakyReLUTest,
                     LayerHardtanhTest, FHardtanhTest,
                     LayerReLUTest, FReLUTest,
                     LayerELUTest, FPELUTest,
    ]:
        for i in range(10):
            model = act_type()
            model.eval()

            input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
            input_var = torch.FloatTensor(input_np)

            torch.onnx.export(model, input_var, "_tmpnet.onnx", verbose=True, input_names=['test_in'],
                              output_names=['test_out'])

            onnx_model = onnx.load('_tmpnet.onnx')
            k_model = onnx_to_keras(onnx_model, ['test_in'])
            os.unlink('_tmpnet.onnx')

            error = check_torch_keras_error(model, k_model, input_np)
            print('Error:', error)
            if max_error < error:
                max_error = error

    print('Max error: {0}'.format(max_error))
