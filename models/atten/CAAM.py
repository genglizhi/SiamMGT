import matplotlib.pyplot as plt
# import seaborn as sns
import os
import scipy.io
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time
import sys

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_v = nn.Conv2d(in_planes, in_planes, 1)  # MLp
        self.fc1_i = nn.Conv2d(in_planes, in_planes, 1)  # MLp

        self.softmax = nn.Softmax(dim=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_v, x_i):
        b, c, _, _ = x_v.size()

        avg_out_v = self.fc1_v(self.avg_pool(x_v))
        avg_out_i = self.fc1_i(self.avg_pool(x_i))

        w_v = self.sigmoid(avg_out_v)
        w_i = self.sigmoid(avg_out_i)
        fuse_w = torch.cat((w_v, w_i), dim=1)
        out = fuse_w.view(b, c, 2, -1)
        w = self.softmax(out)
        w = w.chunk(2, dim=2)
        w1 = w[0]
        w2 = w[1]

        x_v1 = x_v * w1
        x_i1 = x_i * w2

        wv = w1.mean()
        wi = w2.mean()
        # wv = w1.sum()
        # wi = w2.sum()

        return x_v, x_i, wv, wi