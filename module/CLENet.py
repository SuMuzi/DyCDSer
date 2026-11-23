import torch
import math
from torch import nn
import torch.nn.functional as F
from blocks import DepthwiseConvFFN, depthwise_separable_conv_block, CBAM

class CLENet(nn.Module):
    def __init__(self, in_channels,hidden_channels=64):
        super(CLENet, self).__init__()
        self.conv1  = depthwise_separable_conv_block(in_channels, hidden_channels)
        self.conv_ffn = DepthwiseConvFFN(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        )
