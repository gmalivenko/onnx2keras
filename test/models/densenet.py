import numpy as np
import torch
from torch.autograd import Variable
from onnx2keras import onnx_to_keras, check_torch_keras_error
import onnx
from torchvision.models.densenet import densenet121


if __name__ == '__main__':
    model = densenet121()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = Variable(torch.FloatTensor(input_np), requires_grad=False)
    output = model(input_var)

    torch.onnx.export(model, (input_var), "_tmpnet.onnx",
                      verbose=True,
                      input_names=['test_in1'],
                      output_names=['test_out']
    )

    onnx_model = onnx.load('_tmpnet.onnx')
    k_model = onnx_to_keras(onnx_model, ['test_in1', 'test_in2'], change_ordering=True)

    error = check_torch_keras_error(model, k_model, input_np, change_ordering=True)

    print('Max error: {0}'.format(error))
