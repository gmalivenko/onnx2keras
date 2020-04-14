import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from onnx2keras import onnx_to_keras, check_torch_keras_error
import onnx


class LayerTest(nn.Module):
    def __init__(self,  kernel_size=3, padding=1, stride=1):
        super(LayerTest, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for change_ordering in [True, False]:
        for kernel_size in [1, 3, 5, 7]:
            for padding in [0, 1, 3]:
                for stride in [1, 2, 3, 4]:
                    # RuntimeError: invalid argument 2: pad should be smaller than half of kernel size, but got padW = 1, padH = 1, kW = 1,
                    if padding > kernel_size / 2:
                        continue

                    model = LayerTest(kernel_size=kernel_size, padding=padding, stride=stride)
                    model.eval()

                    input_np = np.random.uniform(0, 1, (1, 3, 20, 224, 224))
                    input_var = Variable(torch.FloatTensor(input_np))

                    torch.onnx.export(model, input_var, "_tmpnet.onnx", verbose=True, input_names=['test_in'],
                                      output_names=['test_out'])

                    onnx_model = onnx.load('_tmpnet.onnx')
                    k_model = onnx_to_keras(onnx_model, ['test_in'], change_ordering=change_ordering)

                    error = check_torch_keras_error(model, k_model, input_np, change_ordering=change_ordering)
                    print('Error:', error)

                    if max_error < error:
                        max_error = error

    print('Max error: {0}'.format(max_error))
