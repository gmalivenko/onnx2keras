import torch.nn as nn
import torch
import numpy as np
import pytest


from test.utils import convert_and_test


class FNormTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self, dim, keepdim):
        super(FNormTest, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        x = torch.norm(x, p=2, dim=self.dim, keepdim=self.keepdim)
        return x


# TODO: Not working with dim=[2,3] and change_ordering=False ???? error about 0.0001-0.001
@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('dim', [[1, 2], [1, 3]])
@pytest.mark.parametrize('epsilon', [5e-5])
@pytest.mark.parametrize('keepdim', [True, False])
def test_norm(change_ordering, dim, epsilon, keepdim):
    model = FNormTest(dim, keepdim)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))

    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering, epsilon=epsilon)
