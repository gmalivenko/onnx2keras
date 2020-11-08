import numpy as np
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class FMul(nn.Module):
    def __init__(self):
        super(FMul, self).__init__()

    def forward(self, x, y):
        x = x * y
        x = x * 10.0
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_mul(change_ordering):
    model = FMul()
    model.eval()

    input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, (input_np1, input_np2), verbose=False, change_ordering=change_ordering)
