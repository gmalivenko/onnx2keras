# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import torch
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import pytest


class SplitSection(torch.nn.Module):

    def __init__(self):
        super(SplitSection, self).__init__()

    def forward(self, x):
        return torch.split(x, [5, 5], dim=-1)


@pytest.mark.parametrize('opset_version', [12, 14])
def test_split_v8(opset_version):
    model = SplitSection()
    inpt = torch.ones([1, 10, 10])
    torch.onnx.export(model, inpt, 'split_model.onnx', opset_version=opset_version)
    onnx_model = onnx.load('split_model.onnx')
    keras_model = onnx_to_keras(onnx_model, ['tensor'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, ['tensor'])
    assert final_model(inpt)[0].shape == final_model(inpt)[1].shape == [1, 5, 10]
    # assert np.abs(keras_preds-this_pred.detach().numpy()).max() < 1e-04