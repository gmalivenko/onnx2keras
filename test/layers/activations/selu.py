import torch.nn as nn


class LayerSELUTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerSELUTest, self).__init__()
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.selu(x)
        return x


class FSELUTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FSELUTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.selu(x)
