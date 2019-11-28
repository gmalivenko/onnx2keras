import torch.nn as nn
import torch


class FNormTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FNormTest, self).__init__()

    def forward(self, x):
        x = torch.norm(x, p=2, dim=[1, 2])
        return x