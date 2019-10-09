import torch.nn as nn
import numpy as np


class FFloorTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FFloorTest, self).__init__()

    def forward(self, x):
        return x.floor()
