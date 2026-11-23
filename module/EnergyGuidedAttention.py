import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import depthwise_separable_conv_block,depthwise_separable_conv_block2

class EnergyGuidedAttention(nn.Module):
    """
    湿静力能（MSE）梯度引导

    物理基础：
    - MSE = Cp*T + g*z + Lv*q （单位: J/kg）
    - MSE 水平梯度指示能量锋区
    - ∇MSE 大值区：潜在不稳定边界/干侵入
    """

    def __init__(self, num_heads):
        super().__init__()

        # MSE梯度编码
        self.mse_encoder = nn.Sequential(
            depthwise_separable_conv_block(in_c=1, out_c=32),
            depthwise_separable_conv_block2(in_c=32, out_c=num_heads),
            nn.Sigmoid()
        )

    def compute_mse(self, T, q, z):
        """
        湿静力能
        T: K, q: kg/kg, z: m
        """
        Cp = 1004  # J/(kg·K)
        g = 9.81  # m/s²
        Lv = 2.5e6  # J/kg

        mse = Cp * T + g * z + Lv * q  # [B, L, H, W]
        return mse

    def forward(self, features, T, q, z_levels):
        # 计算低层MSE（通常取850hPa或边界层）
        mse = self.compute_mse(T[:, 0], q[:, 0], z_levels[0])

        # MSE水平梯度（用Sobel算子）
        grad_mse = self.compute_gradient(mse)

        # 归一化
        mse_norm = (mse - mse.mean()) / (mse.std() + 1e-8)
        grad_norm = torch.clamp(grad_mse / 1000, 0, 5)  # 梯度阈值

        energy_input = torch.cat([mse_norm.unsqueeze(1), grad_norm], dim=1)
        energy_attention = self.mse_encoder(energy_input)

        return features * energy_attention, energy_attention