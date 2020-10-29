import numpy as np
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class FAdd(nn.Module):
    def __init__(self):
        super(FAdd, self).__init__()

    def forward(self, x, y):
        x = x + y + np.float32(0.1)
        # x = x
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_add(change_ordering):
    model = FAdd()
    model.eval()

    input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, (input_np1, input_np2), verbose=False, change_ordering=change_ordering)
