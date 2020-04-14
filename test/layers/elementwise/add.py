import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from onnx2keras import onnx_to_keras, check_torch_keras_error
import onnx


class FTest(nn.Module):
    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x + y + np.float32(0.1)
        # x = x
        return x


if __name__ == '__main__':
    max_error = 0

    for change_ordering in [False, True]:
        for i in range(10):
            model = FTest()
            model.eval()

            input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
            input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
            input_var1 = Variable(torch.FloatTensor(input_np1))
            input_var2 = Variable(torch.FloatTensor(input_np2))

            torch.onnx.export(model, (input_var1, input_var2), "_tmpnet.onnx", verbose=True, input_names=['test_in1', 'test_in2'],
                              output_names=['test_out'])

            onnx_model = onnx.load('_tmpnet.onnx')
            k_model = onnx_to_keras(onnx_model, ['test_in1', 'test_in2'], change_ordering=change_ordering)

            error = check_torch_keras_error(model, k_model, [input_np1, input_np2], change_ordering=change_ordering)
            if max_error < error:
                max_error = error

    print('Max error: {0}'.format(max_error))
