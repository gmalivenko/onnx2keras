import torch.nn as nn
import random


class LayerELUTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerELUTest, self).__init__()
        self.alpha = random.random()
        self.elu = nn.ELU(alpha=self.alpha)

    def forward(self, x):
        x = self.elu(x)
        return x


class FPELUTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FPELUTest, self).__init__()
        self.alpha = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.elu(x, alpha=self.alpha)
