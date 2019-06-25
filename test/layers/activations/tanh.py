import torch.nn as nn


class LayerTanhTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerTanhTest, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        return x


class FTanhTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FTanhTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.tanh(x)
