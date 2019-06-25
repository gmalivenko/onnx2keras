import torch.nn as nn
import random


class LayerHardtanhTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerHardtanhTest, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()
        self.htanh = nn.Hardtanh(min_val=self.min_val, max_val=self.max_val)

    def forward(self, x):
        x = self.htanh(x)
        return x


class FHardtanhTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FHardtanhTest, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)
