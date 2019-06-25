import torch.nn as nn
import random


class LayerThresholdTest(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerThresholdTest, self).__init__()
        self.threshold = random.random()
        self.value = self.threshold + random.random()
        self.thresh = nn.Threshold(self.threshold, self.value)

    def forward(self, x):
        x = self.thresh(x)
        return x


class FThresholdTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FThresholdTest, self).__init__()
        self.threshold = random.random()
        self.value = self.threshold + random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.threshold(x, threshold=self.threshold, value=self.value)
