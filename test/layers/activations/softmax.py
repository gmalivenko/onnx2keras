import torch.nn as nn
import numpy as np


class LayerSoftmaxTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerSoftmaxTest, self).__init__()
        self.dim = np.random.randint(0, 3)
        self.softmax = nn.Softmax(dim=self.dim)

    def forward(self, x):
        x = self.softmax(x)
        return x


class FSoftmaxTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FSoftmaxTest, self).__init__()
        self.dim = np.random.randint(0, 3)

    def forward(self, x):
        from torch.nn import functional as F
        return F.softmax(x, self.dim)

