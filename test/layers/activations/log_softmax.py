import torch.nn as nn
import numpy as np


class LayerLogSoftmaxTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerLogSoftmaxTest, self).__init__()
        self.dim = np.random.randint(0, 3)
        self.softmax = nn.LogSoftmax(dim=self.dim)

    def forward(self, x):
        x = self.softmax(x)
        return x


class FLogSoftmaxTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FLogSoftmaxTest, self).__init__()
        self.dim = np.random.randint(0, 3)

    def forward(self, x):
        from torch.nn import functional as F
        return F.softmax(x, self.dim)

