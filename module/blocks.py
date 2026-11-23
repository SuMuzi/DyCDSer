import torch
import math
from torch import nn
import torch.nn.functional as F
def depthwise_separable_conv_block(in_c, out_c, kernel_size=3, padding=1):
    return nn.Sequential(
        # Depthwise Convolution
        nn.Conv2d(in_c, in_c, kernel_size=kernel_size, padding=padding, padding_mode='replicate',groups=in_c, bias=False),
        nn.GroupNorm(in_c, in_c),
        nn.LeakyReLU(0.1, inplace=True),

        # Pointwise Convolution
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.GroupNorm(4, out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8,act=None,norm=None):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self,act=None,norm=None):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3,padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, ssd_config, split_nums,act=None,norm=None):
        super(CBAM, self).__init__()
        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.channel_attention = ChannelAttentionModule(channel,act=act, norm=norm)
        self.spatial_attention = SpatialAttentionModule(act=act, norm=norm)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class DepthwiseConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_groups=8, negative_slope=0.01):
        super(DepthwiseConvFFN, self).__init__()


        # 深度可分离卷积：通道扩展
        self.conv1 = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                padding_mode='reflect',
                groups=in_channels,
                bias=False
            ),
            nn.GroupNorm(in_channels, in_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            # Pointwise Convolution
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

        # 深度可分离卷积：通道压缩
        self.conv2 = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode='reflect',
                groups=num_groups,
                bias=False
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            # Pointwise Convolution
            nn.Conv2d(
                hidden_channels,
                in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(in_channels, in_channels)
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity  # 残差连接
