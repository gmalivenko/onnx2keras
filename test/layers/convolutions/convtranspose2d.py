import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from onnx2keras import onnx_to_keras, check_torch_keras_error
import onnx


class LayerTest(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False):
        super(LayerTest, self).__init__()
        self.conv = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, padding=padding,
                                       stride=stride, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for change_ordering in [False, True]:
        for kernel_size in [1, 3, 5]:
            for padding in [0, 1, 3]:
                for stride in [1, 2]:
                    for bias in [True, False]:
                        outs = np.random.choice([1, 3, 7])

                        model = LayerTest(3, outs, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
                        model.eval()

                        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
                        input_var = Variable(torch.FloatTensor(input_np))

                        torch.onnx.export(model, input_var, "_tmpnet.onnx",
                                          verbose=True, input_names=['test_in'], output_names=['test_out'])

                        onnx_model = onnx.load('_tmpnet.onnx')
                        k_model = onnx_to_keras(onnx_model, ['test_in'], change_ordering=change_ordering)

                        error = check_torch_keras_error(model, k_model, input_np, change_ordering=change_ordering)
                        print('Error:', error)

                        if max_error < error:
                            max_error = error

    print('Max error: {0}'.format(max_error))
