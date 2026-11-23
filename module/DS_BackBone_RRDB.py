import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .blocks import depthwise_separable_conv_block

# ==================== RRDB 模块 ====================
class UpsampleBlock(nn.Module):
    """2x 上采样：Conv -> PixelShuffle -> LeakyReLU"""
    def __init__(self, in_channels: int = 64, up_scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2),
                              kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.pixel_shuffle(self.conv(x)))

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB)
    密集连接 + 残差连接，增强特征复用能力
    """

    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate,
                          kernel_size=3, padding=1, padding_mode='replicate'),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        # 局部特征融合
        self.local_fusion = nn.Conv2d(in_channels + num_layers * growth_rate,
                                      in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.local_fusion(torch.cat(features, dim=1))
        return fused + x  # 残差连接


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB)
    RDB 的堆叠 + 残差连接，形成更深的特征提取网络
    """

    def __init__(self, in_channels: int, num_blocks: int = 3, growth_rate: int = 32):
        super().__init__()
        self.rdb_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdb_blocks.append(ResidualDenseBlock(in_channels, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for rdb in self.rdb_blocks:
            out = rdb(out)
        return out + x  # 全局残差连接

class RRDBNet(nn.Module):
    """
    RRDBNet超分辨率网络

    Args:
        in_channels:   输入通道数（降水=1，RGB=3）
        out_channels:  输出通道数
        num_features:   基础特征数（默认64）
        num_blocks:     RRDB 数量（默认23，ESRGAN 配置）
        num_growth:     RDB 中间增长通道数（默认32）
        num_upsample:   上采样块数量（2x=1, 4x=2, 8x=3）
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 23,
        num_growth: int = 32,
        num_upsample: int = 2,  # 4x
    ):
        super().__init__()

        # 1. 初始卷积（低分辨率特征提取）
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        # 2. RRDB  trunk（主体）
        self.trunk = nn.Sequential(
            *[RRDB(num_features, num_growth, beta=0.2) for _ in range(num_blocks)]
        )

        # 3. trunk 后卷积（用于跳跃连接）
        self.conv_trunk = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        # 4. 上采样模块
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(num_features, up_scale=2) for _ in range(num_upsample)]
        )

        # 5. 重建卷积（高分辨率精修）
        self.conv_hr = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 6. 输出卷积
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W] 低分辨率输入
        Returns:
            [B, out_channels, H*2^N, W*2^N] 高分辨率输出
        """
        # 初始特征
        feat = self.conv_first(x)

        # RRDB trunk + 长跳跃连接
        trunk_feat = self.trunk(feat)
        trunk_feat = self.conv_trunk(trunk_feat)
        feat = feat + trunk_feat  # 全局残差连接

        # 上采样
        feat = self.upsample_blocks(feat)

        # 重建输出
        feat = self.conv_hr(feat)
        out = self.conv_last(feat)

        return out