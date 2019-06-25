import torch.nn as nn


class LayerSigmoidTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerSigmoidTest, self).__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.sig(x)
        return x


class FSigmoidTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FSigmoidTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.sigmoid(x)
