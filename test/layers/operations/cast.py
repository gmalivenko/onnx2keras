import torch.nn as nn
import numpy as np
import torch


class FCastTest(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FCastTest, self).__init__()

    def forward(self, x):
        return x.type(torch.DoubleTensor).type(torch.BoolTensor).type(torch.uint8)


