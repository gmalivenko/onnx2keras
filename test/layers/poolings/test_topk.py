import torch
from test.utils import convert_and_test
import numpy as np
import pytest
import io
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import itertools


class TopK(torch.nn.Module):
    def __init__(self, k, dim, largest, return_sorted):
        self.k = k
        self.dim = dim
        self.largest = largest
        self.return_sorted = return_sorted
        super(TopK, self).__init__()

    def forward(self, x):
        return torch.topk(x, self.k, self.dim, self.largest, self.return_sorted)


@pytest.mark.parametrize('return_sorted', [True, False])
@pytest.mark.parametrize('largest', [True, False])
@pytest.mark.parametrize('dim', [-1, 1])
@pytest.mark.parametrize('k', [2])
def test_topk(return_sorted, largest, dim, k):
    k=2
    dim=1
    return_sorted=False
    largest=True
    np_input = np.random.random((1, 3, 8))
    pt_model = TopK(k, dim, largest, return_sorted)
    temp_f = io.BytesIO()
    torch.onnx.export(pt_model, torch.from_numpy(np_input), temp_f, verbose=True,
                      input_names=['test_in'],
                      output_names=['test_out_1', 'test_out_2'])
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    keras_model = onnx_to_keras(onnx_model, ['test_in'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_res = final_model(np.swapaxes(np_input, 1, 2))
    pt_res = pt_model(torch.from_numpy(np_input))
    if not return_sorted:
        reshaped_pt_pred = np.array(pt_res[0].transpose(1,2)).reshape(-1,k)
        reshaped_keras_pred = np.array(keras_res[0]).reshape(-1, k)
        for i in range(reshaped_pt_pred.shape[0]):
            has_valid_permute = False
            pt_i_res = reshaped_pt_pred[i, :]
            for permutation in itertools.permutations(reshaped_keras_pred[i,:]):
                if ((permutation - pt_i_res).__abs__() < 10**(-6)).all():
                    has_valid_permute = True
            if has_valid_permute is False:
                assert False
    else:
        value_same = ((np.swapaxes(keras_res[0], 1, 2) - pt_res[0].numpy()).__abs__() < 10**(-6)).all()
        index_same = ((np.swapaxes(keras_res[1], 1, 2) - pt_res[1].numpy()).__abs__() < 1).all()
        assert value_same and index_same

