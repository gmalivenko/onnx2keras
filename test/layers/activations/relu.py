import torch.nn as nn


class LayerReLUTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerReLUTest, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x


class FReLUTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FReLUTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.relu(x)
