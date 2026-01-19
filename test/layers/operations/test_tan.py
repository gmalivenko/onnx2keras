import torch
from test.utils import convert_and_test
import numpy as np


class TorchTan(torch.nn.Module):
    def __init__(self):
        super(TorchTan, self).__init__()

    def forward(self, x):
        return torch.tan(x)


def test_tan():
    pt_model = TorchTan()
    error = convert_and_test(pt_model, (np.random.random((1, 8, 3))), verbose=False)

