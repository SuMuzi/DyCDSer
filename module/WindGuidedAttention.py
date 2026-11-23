import torch
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class WindGuidedAttention(nn.Module):
    """
    利用风场生成空间注意力，引导网络关注关键区域
    """

    def __init__(self, channels=64):
        super().__init__()
        # 风场编码器
        self.wind_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # u, v, speed
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid()  # 输出注意力权重
        )

        # 特征融合
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, features, u, v):
        """
        features: 降水特征 (B, C, H, W)
        u, v: 风场分量 (B, 1, H, W)
        """
        # 计算风速
        speed = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        wind_input = torch.cat([u, v, speed], dim=1)

        # 生成注意力图
        attention = self.wind_encoder(wind_input)  # (B, C, H, W)

        # 风场强的地方给予更多关注（如辐合区）
        attended_features = features * attention

        # 融合
        out = self.fusion(torch.cat([features, attended_features], dim=1))

        return out, attention


class WindAttnUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(...)
        self.wind_attn = WindGuidedAttention(64)

    def forward(self, lr_precip, u, v):
        # 编码
        feat = self.unet.encoder(lr_precip)

        # 风场引导
        feat, attn_map = self.wind_attn(feat, u, v)

        # 解码
        out = self.unet.decoder(feat)

        return out, attn_map  # 返回注意力图用于可视化