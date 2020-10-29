import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class FFloorTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FFloorTest, self).__init__()

    def forward(self, x):
        return x.floor()


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_floor(change_ordering):
    model = FFloorTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
