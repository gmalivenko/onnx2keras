import numpy as np
import torch
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class FConstant(nn.Module):
    def __init__(self, constant):
        super(FConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x + torch.FloatTensor([self.constant])


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('constant', [-1.0, 0.0, 1.0])
def test_constant(change_ordering, constant):
    model = FConstant(constant)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
