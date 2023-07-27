import torch
import numpy as np
from test.utils import convert_and_test, torch2keras
from keras_data_format_converter import convert_channels_first_to_last
import pytest


class TorchMod(torch.nn.Module):
    def __init__(self):
        super(TorchMod, self).__init__()

    def forward(self, x, y):
        return torch.fmod(x, y)


def test_mod():
    pt_model = TorchMod()
    x = np.random.randint(low=-100, high=100, size=(1, 50, 50))
    y = np.random.randint(low=-100, high=100, size=(1, 50, 50))
    y[y==0] = 1
    k_model = torch2keras(pt_model, (x, y), verbose=False, change_ordering=False)
    final_k = convert_channels_first_to_last(k_model, ['test_in_1', 'test_in_2'])
    assert (final_k((x, y))-pt_model(torch.from_numpy(x), torch.from_numpy(y))).numpy().max() < 10**(-5)


