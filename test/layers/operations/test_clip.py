import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class FClipTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        self.low = np.random.uniform(-1, 1)
        self.high = np.random.uniform(1, 2)
        super(FClipTest, self).__init__()

    def forward(self, x):
        return x.clamp(self.low, self.high)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_clip(change_ordering):
    model = FClipTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
