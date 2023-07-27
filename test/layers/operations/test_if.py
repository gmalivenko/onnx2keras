import torch
from test.utils import convert_and_test, torch2keras
from keras_data_format_converter import convert_channels_first_to_last
import numpy as np


class TorchIfInf(torch.nn.Module):
    def __init__(self):
        super(TorchIfInf, self).__init__()

    def forward(self, x):
        return torch.isinf(x)


class TorchIfNan(torch.nn.Module):
    def __init__(self):
        super(TorchIfNan, self).__init__()

    def forward(self, x):
        return torch.isnan(x)


def test_ifinf():
    pt_model = TorchIfInf()
    np_input = 1/np.random.randint(low=0,high=2,size=(1, 8, 3))
    k_model = torch2keras(pt_model, np_input, verbose=False, change_ordering=False)
    final_k = convert_channels_first_to_last(k_model, ['test_in'])
    assert (pt_model(torch.from_numpy(np_input)).numpy() == np.swapaxes(final_k(np.swapaxes(np_input, 1, 2)), 2, 1)).all()


def test_ifnan():
    pt_model = TorchIfNan()
    np_input = np.random.random(size=(1, 8, 3))
    selection = np.random.randint(low=0, high=2, size=(1, 8, 3))
    np_input[selection == 1] = np.nan
    k_model = torch2keras(pt_model, np_input, verbose=False, change_ordering=False)
    final_k = convert_channels_first_to_last(k_model, ['test_in'])
    assert (pt_model(torch.from_numpy(np_input)).numpy() == np.swapaxes(final_k(np.swapaxes(np_input, 1, 2)), 2, 1)).all()
