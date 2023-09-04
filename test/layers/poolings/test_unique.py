import torch
from test.utils import convert_and_test
import numpy as np
import pytest
import io
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import itertools


class Unique(torch.nn.Module):
    def __init__(self, to_sort=False, return_inverse=False, return_counts=False, dim=None):
        super(Unique, self).__init__()
        self.to_sort = to_sort
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.dim = dim

    def forward(self, x):
        return torch.unique(x, sorted=self.to_sort, return_inverse=self.return_inverse,
                            return_counts=self.return_counts, dim=self.dim)


@pytest.mark.parametrize('return_inverse', [True])
@pytest.mark.parametrize('return_counts', [True])
@pytest.mark.parametrize('to_sort', [True, False])
def test_unique(return_inverse, return_counts, to_sort):
    to_sort = True
    return_inverse = True
    return_counts = True
    dim = None
    pt_model = Unique(to_sort=to_sort, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
    output_names = ['unique']
    if return_inverse:
        output_names += ['inverse']
    if return_counts:
        output_names += ['counts']
    torch_input = torch.randint(5, 9, (1, 8, 10))
    temp_f = io.BytesIO()
    # torch.onnx.export(pt_model, torch_input, temp_f, verbose=True,
    #                   input_names=['x'],
    #                   output_names=output_names)
    torch.onnx.export(pt_model, torch_input, temp_f, verbose=True,
                      input_names=['x'],
                      output_names=output_names)
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    keras_model = onnx_to_keras(onnx_model, ['x'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    rotated_input = np.swapaxes(torch_input.numpy(), 1, 2)
    res_keras = final_model(rotated_input)
    assert ((res_keras[0].numpy()[
                 np.swapaxes(res_keras[1].numpy()[..., 0].reshape((1, 8, 10)), 1, 2).astype(int)][
                 ..., 0] - rotated_input) < 0.5).all()

    # keras_res = final_model([input_t.numpy().swapaxes(1, 2), h0_t.numpy().swapaxes(1, 2)])
    # pt_res = pt_model(input_t, h0_t)
    # diff_tens_state = (pt_res[0].swapaxes(1, 2).detach().numpy() - keras_res[0]).numpy().__abs__()
    # diff_tens_out = (pt_res[1].swapaxes(1, 2).detach().numpy() - keras_res[1]).numpy().__abs__()
    # eps = 10**(-5)
    # if (diff_tens_state.max() < eps) & (diff_tens_out.max() < eps) is False:
    #     print(1)
    # assert (diff_tens_state.max() < eps) & (diff_tens_out.max() < eps)
