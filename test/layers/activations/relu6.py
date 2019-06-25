import torch.nn as nn


class LayerReLU6Test(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerReLU6Test, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.relu(x)
        return x


class FReLU6Test(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FReLU6Test, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.relu6(x)
