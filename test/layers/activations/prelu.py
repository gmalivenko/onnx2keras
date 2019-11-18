import torch.nn as nn
import torch


class LayerPReLUTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self, num_params=3):
        super(LayerPReLUTest, self).__init__()
        self.num_params = num_params
        self.prelu = nn.PReLU(num_params)

    def forward(self, x):
        x = self.prelu(x)
        return x


class FPReLUTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self, num_params=3):
        super(FPReLUTest, self).__init__()
        self.num_params = num_params

    def forward(self, x):
        from torch.nn import functional as F
        weights = torch.FloatTensor(torch.rand(self.num_params).numpy())
        return F.prelu(x, weight=weights)
