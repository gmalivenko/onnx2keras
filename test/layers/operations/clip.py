import torch.nn as nn
import numpy as np


class FClipTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        self.low = np.random.uniform(-1, 1)
        self.high = np.random.uniform(1, 2)
        super(FClipTest, self).__init__()

    def forward(self, x):
        return x.clamp(self.low, self.high)
