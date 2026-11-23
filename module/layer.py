import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DivergenceGuidedAttention(nn.Module):
    """
    散度引导的注意力机制

    将物理散度场转换为注意力权重，引导模型关注动力学活跃区域
    """

    def __init__(self, dim, num_heads=8, divergence_weight=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.divergence_weight = divergence_weight  # 散度引导强度

        # 标准注意力投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 散度编码网络
        self.divergence_encoder = nn.Sequential(
            nn.Linear(1, 64),  # 输入：散度标量
            nn.ReLU(),
            nn.Linear(64, num_heads),  # 输出：每个头的散度调制
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        # 散度-注意力融合门控
        self.divergence_gate = nn.Sequential(
            nn.Linear(dim + num_heads, num_heads),
            nn.Sigmoid()
        )

    def compute_divergence(self, velocity_field):
        """
        计算速度场的散度

        velocity_field: [B, 2, H, W] 或 [B, 3, H, W, D]
            包含 u, v (w) 分量

        Returns:
            divergence: [B, 1, H, W] 散度场
        """
        B, C, *spatial = velocity_field.shape

        if len(spatial) == 2:  # 2D
            u, v = velocity_field[:, 0], velocity_field[:, 1]

            # 有限差分计算散度
            du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2  # 中心差分
            dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2

            # 填充保持尺寸
            du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')
            dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')

            divergence = du_dx + dv_dy

        elif len(spatial) == 3:  # 3D
            u, v, w = velocity_field[:, 0], velocity_field[:, 1], velocity_field[:, 2]

            du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2
            dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2
            dw_dz = (w[:, 2:, :, :] - w[:, :-2, :, :]) / 2

            du_dx = F.pad(du_dx, (1, 1, 0, 0, 0, 0), mode='replicate')
            dv_dy = F.pad(dv_dy, (0, 0, 1, 1, 0, 0), mode='replicate')
            dw_dz = F.pad(dw_dz, (0, 0, 0, 0, 1, 1), mode='replicate')

            divergence = du_dx + dv_dy + dw_dz

        return divergence.unsqueeze(1)  # [B, 1, H, W] 或 [B, 1, D, H, W]

    def divergence_to_attention(self, divergence, features):
        """
        将散度场转换为注意力调制

        策略：
        - 强负散度（辐合）：增强注意力（重要区域）
        - 强正散度（辐散）：适度增强（高层抽吸也重要）
        - 近零散度：标准注意力
        """
        B = divergence.shape[0]

        # 全局散度统计
        div_mean = divergence.mean(dim=[2, 3], keepdim=True)
        div_std = divergence.std(dim=[2, 3], keepdim=True) + 1e-6

        # 标准化
        div_norm = (divergence - div_mean) / div_std  # [B, 1, H, W]

        # 编码为注意力调制
        # 负散度（辐合）→ 正值调制
        div_modulation = -div_norm  # 反转：负散度变成高注意力

        # 每个空间位置的调制系数 [B, num_heads, H, W]
        div_encoded = self.divergence_encoder(
            div_modulation.permute(0, 2, 3, 1)  # [B, H, W, 1]
        ).permute(0, 3, 1, 2)  # [B, num_heads, H, W]

        return div_encoded

    def forward(self, x, velocity_field=None, return_divergence=False):
        """
        前向传播

        Parameters:
        -----------
        x : [B, L, D] 或 [B, C, H, W] - 输入特征
        velocity_field : [B, 2/3, H, W] - 速度场（可选，可从x推导）
        """
        # 处理输入维度
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        else:
            B, L, D = x.shape
            H = W = int(np.sqrt(L))
            x_flat = x

        # 标准注意力计算
        q = self.q_proj(x_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, L, d]

        # 计算散度引导（如果提供速度场）
        if velocity_field is not None:
            # 计算散度
            divergence = self.compute_divergence(velocity_field)  # [B, 1, H, W]

            # 转换为注意力调制
            div_attention = self.divergence_to_attention(divergence, x)  # [B, num_heads, H, W]
            div_attention_flat = div_attention.view(B, self.num_heads, -1, 1)  # [B, H, L, 1]
        else:
            # 从特征隐式估计散度（学习得到）
            # 或使用零调制
            divergence = None
            div_attention_flat = 0

        # 标准注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]

        # 散度引导：增强辐合区域的注意力
        # 方法1：直接加到分数上
        divergence_bias = self.divergence_weight * div_attention_flat
        scores = scores + divergence_bias

        # 方法2：门控融合（可选）
        # gate = self.divergence_gate(torch.cat([...], dim=-1))
        # scores = gate * scores + (1-gate) * divergence_bias

        # Softmax归一化
        attn = F.softmax(scores, dim=-1)

        # 应用注意力
        out = torch.matmul(attn, v)  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)

        if x.dim() == 4:
            out = out.permute(0, 2, 1).view(B, self.dim, H, W)

        if return_divergence and divergence is not None:
            return out, divergence

        return out