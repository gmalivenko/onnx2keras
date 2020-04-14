import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from onnx2keras import onnx_to_keras, check_torch_keras_error
import onnx


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()

    def forward(self, x):
        x = x[:1, :2, :3, :4]
        return x


if __name__ == '__main__':
    max_error = 0

    for change_ordering in [True, False]:
        model = LayerTest()
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
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
