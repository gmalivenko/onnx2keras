import torch.nn as nn
import random


class LayerLeakyReLUTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerLeakyReLUTest, self).__init__()
        self.negative_slope = random.random()
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        x = self.leaky_relu(x)
        return x


class FLeakyReLUTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FLeakyReLUTest, self).__init__()
        self.negative_slope = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.leaky_relu(x, self.negative_slope)
