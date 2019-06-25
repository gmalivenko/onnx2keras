import torch.nn as nn


class LayerLogSigmoidTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerLogSigmoidTest, self).__init__()
        self.sig = nn.LogSigmoid()

    def forward(self, x):
        x = self.sig(x)
        return x


class FLogSigmoidTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FLogSigmoidTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.logsigmoid(x)
