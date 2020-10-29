import torch.nn as nn
import numpy as np
import torch
import pytest

from test.utils import convert_and_test


class FCastTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FCastTest, self).__init__()

    def forward(self, x):
        return x.type(torch.DoubleTensor).type(torch.BoolTensor).type(torch.uint8)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_cast(change_ordering):
    model = FCastTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
