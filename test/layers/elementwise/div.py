import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from onnx2keras import onnx_to_keras
import onnx


class FTest(nn.Module):
    def __init__(self):
        super(FTest, self).__init__()

    def forward(self, x, y):
        x = x / 2
        y = y / 2

        x = x / y
        return x


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0

    for i in range(10):
        model = FTest()
        model.eval()

        input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var1 = Variable(torch.FloatTensor(input_np1))
        input_var2 = Variable(torch.FloatTensor(input_np2))
        output = model(input_var1, input_var2)

        torch.onnx.export(model, (input_var1, input_var2), "_tmpnet.onnx", verbose=True,
                          input_names=['test_in1', 'test_in2'],
                          output_names=['test_out'])

        onnx_model = onnx.load('_tmpnet.onnx')
        k_model = onnx_to_keras(onnx_model, ['test_in1', 'test_in2'])

        error = check_error(output, k_model, [input_np1, input_np2])
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
