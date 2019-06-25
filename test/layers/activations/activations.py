from onnx2keras import onnx_to_keras, check_torch_keras_error
from relu import LayerReLUTest, FReLUTest
from hard_tanh import LayerHardtanhTest, FHardtanhTest

import torch
import numpy as np
import onnx

if __name__ == '__main__':
    max_error = 0
    for act_type in [LayerReLUTest, FReLUTest,
                     LayerHardtanhTest, FHardtanhTest]:
        for i in range(100):
            model = act_type()
            model.eval()

            input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
            input_var = torch.FloatTensor(input_np)

            torch.onnx.export(model, input_var, "_tmpnet.onnx", verbose=True, input_names=['test_in'],
                              output_names=['test_out'])

            onnx_model = onnx.load('_tmpnet.onnx')
            k_model = onnx_to_keras(onnx_model, ['test_in'])

            error = check_torch_keras_error(model, k_model, input_np)
            print('Error:', error)
            if max_error < error:
                max_error = error

    print('Max error: {0}'.format(max_error))
