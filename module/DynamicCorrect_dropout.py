import torch
import torch.nn as nn
from typing import Optional, Dict

def depthwise_separable_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )


class DropPath(nn.Module):
    """随机深度 DropPath"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class RRDB(nn.Module):
    """带DropPath的RRDB"""
    def __init__(self, in_channels, num_blocks=1, growth_rate=32, drop_path_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(growth_rate, in_channels + i * growth_rate, 3, padding=1),
                DropPath(drop_path_rate)
            )
            self.blocks.append(block)
        
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class WindErrorExtractor(nn.Module):
    def __init__(self, base_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, base_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_dim, base_dim, 3, padding=1)
        )
        
    def forward(self, u_diff, v_diff):
        x = torch.cat([u_diff, v_diff], dim=1)
        feat = self.conv(x)
        divergence = self._compute_divergence(u_diff, v_diff)
        return feat, divergence
    
    def _compute_divergence(self, u, v):
        dx_u = nn.functional.pad(u[:, :, :, 2:] - u[:, :, :, :-2], (1, 1, 0, 0), mode='replicate') / 2
        dy_v = nn.functional.pad(v[:, :, 2:, :] - v[:, :, :-2, :], (0, 0, 1, 1), mode='replicate') / 2
        return dx_u + dy_v


class WindGuidedPrecipitationAdjustment(nn.Module):
    def __init__(self, precip_dim=64, wind_feat_dim=64):
        super().__init__()
        self.adjust_conv = nn.Sequential(
            nn.Conv2d(wind_feat_dim + 1, precip_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(precip_dim, 1, 1)
        )
        
    def forward(self, precip_feat, wind_feat, div_error, raw_precip):
        x = torch.cat([wind_feat, div_error], dim=1)
        adjustment = self.adjust_conv(x)
        adjusted = raw_precip + adjustment
        return adjusted, {'adjustment': adjustment, 'div_error': div_error}


class MultiVariableFusionWithWindCorrection(nn.Module):
    """
    改进版：添加正则化和防止过拟合机制
    """
    
    def __init__(
            self,
            precip_channels: int = 1,
            other_channels: int = 17,
            wind_channels: int = 2,
            base_dim: int = 64,
            dropout_rate: float = 0.1,
            drop_path_rate: float = 0.1,
            reduce_complexity: bool = True  # 简化模型
    ):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        # 降水编码器（简化版）
        if reduce_complexity:
            self.precip_encoder = nn.Sequential(
                depthwise_separable_conv_block(precip_channels, base_dim),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(base_dim, base_dim, 1),
            )
        else:
            self.precip_encoder = nn.Sequential(
                depthwise_separable_conv_block(precip_channels, base_dim),
                nn.Dropout2d(dropout_rate),
                depthwise_separable_conv_block(base_dim, 2 * base_dim),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(2 * base_dim, base_dim, 1),
            )

        # 其他变量编码器
        self.other_encoder = nn.Sequential(
            depthwise_separable_conv_block(other_channels, base_dim),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(base_dim, base_dim, 1),
        )

        # 风速差异提取
        self.wind_error_extractor = WindErrorExtractor()

        # 风速引导的降水调整
        self.wind_adjustment = WindGuidedPrecipitationAdjustment(
            precip_dim=base_dim,
            wind_feat_dim=base_dim
        )

        # 多变量融合
        self.fusion = nn.Sequential(
            depthwise_separable_conv_block(base_dim + base_dim, base_dim),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(base_dim, base_dim, 1),
            nn.LeakyReLU(0.1)
        )

        # RRDB（简化为1个block）
        self.rrdb = RRDB(
            in_channels=base_dim, 
            num_blocks=1 if reduce_complexity else 3, 
            growth_rate=32,
            drop_path_rate=drop_path_rate
        )

        # 输出头
        self.precip_head = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            depthwise_separable_conv_block(base_dim, base_dim // 2),
            nn.Conv2d(base_dim // 2, 1, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(
            self,
            precip_inputs: torch.Tensor,
            other_vars: torch.Tensor,
            wind_model: torch.Tensor,
            wind_obs: Optional[torch.Tensor] = None,
            return_adjustments: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # 编码各变量
        precip_feat = self.precip_encoder(precip_inputs)
        other_feat = self.other_encoder(other_vars)

        adjustments = None

        # 计算风速差异
        if wind_obs is not None:
            u_diff = wind_model[:, 0:1] - wind_obs[:, 0:1]
            v_diff = wind_model[:, 1:2] - wind_obs[:, 1:2]

            wind_feat, div_error = self.wind_error_extractor(u_diff, v_diff)

            raw_precip = precip_inputs[:, 0:1]
            adjusted_precip, adjustments = self.wind_adjustment(
                precip_feat, wind_feat, div_error, raw_precip
            )

            # 调整降水特征
            precip_feat_adjusted = precip_feat + self.precip_encoder(
                torch.cat([adjusted_precip, torch.zeros_like(precip_inputs[:, 1:])], dim=1)
            ) - self.precip_encoder(
                torch.cat([raw_precip, torch.zeros_like(precip_inputs[:, 1:])], dim=1)
            )
        else:
            precip_feat_adjusted = precip_feat

        # 多变量融合
        fused = self.fusion(torch.cat([precip_feat_adjusted, other_feat], dim=1))

        # RRDB深层特征提取
        enhanced_feat = self.rrdb(fused)

        # 输出
        precip_out = self.precip_head(enhanced_feat)

        outputs = {
            'precipitation': precip_out,
            'precip_adjusted': adjusted_precip if 'adjusted_precip' in locals() else precip_out
        }

        if return_adjustments and adjustments is not None:
            outputs['adjustments'] = adjustments

        return outputs