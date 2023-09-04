import torch
from test.utils import convert_and_test
import numpy as np
import torch
from test.utils import convert_and_test
import numpy as np
import pytest
import io
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import itertools


class TorchGather(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(TorchGather, self).__init__()

    def forward(self, x, y):
        return torch.gather(x, self.dim, y)


def test_gather_elements():
    t = torch.tensor([[1, 2], [3, 4]])
    pt_model = TorchGather(1)
    temp_f = io.BytesIO()
    idx_arr = [[0, 0], [1, 0]]
    torch_idx_arr = torch.tensor(idx_arr)
    torch.onnx.export(pt_model, (t, torch_idx_arr), temp_f, verbose=True,
                      input_names=['test_in_1', 'test_in_2'],
                      output_names=['test_out'])
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    keras_model = onnx_to_keras(onnx_model, ['test_in_1', 'test_in_2'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    assert (final_model([t.numpy(), np.array(idx_arr)])-pt_model(t, torch_idx_arr) < 1).numpy().all()

