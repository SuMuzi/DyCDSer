import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # 门控机制：计算门控权重 (Gate)
        # 输入是 query 和 key/value 的拼接，输出 0 到 1 之间的权重
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        """
        Args:
            query: 主输入 (Batch, Seq_Len_Q, D_Model)
            key: 辅助输入 (Batch, Seq_Len_K, D_Model)
            value: 辅助输入 (Batch, Seq_Len_K, D_Model)
        """
        # 1. 标准交叉注意力
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)

        # 2. 门控计算 (Gating Mechanism)
        # 将 query 和 注意力输出 拼接，决定保留多少注意力信息
        # 也可以选择拼接 query 和 key 的均值，视具体业务逻辑而定
        gate_input = torch.cat([query, attn_output], dim=-1)
        gate = self.gate_net(gate_input)

        # 3. 应用门控并融合
        # gate 决定了注意力输出的贡献度
        gated_output = gate * attn_output

        # 4. 残差连接
        output = self.norm(query + gated_output)
        return output

class CAPECINGatingModule(nn.Module):
    def __init__(self, feature_dim):
        """
        feature_dim: 输入特征图的通道数
        """
        super(CAPECINGatingModule, self).__init__()

        # 1. 数据驱动的门控生成路径
        self.data_gate_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # 2. 物理先验（CAPE & CIN）的特征提取路径
        # 输入通道为 2 (CAPE 和 CIN)，输出通道与特征图一致
        # 使用 3x3 卷积可以感受周围的能量梯度，比 1x1 效果更好
        self.physics_extractor = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.GroupNorm(num_groups=16, num_channels=4),
            nn.SiLU(),  # 平滑激活函数
            nn.Conv2d(16, feature_dim, kernel_size=1)
        )

        # 3. 可学习的物理权重系数 gamma
        # 初始化为 0，让模型在训练初期依赖数据驱动，随后逐渐引入物理先验
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, cape, cin, t2m, t850, t500, z, z850, z500):

        dz_mid = z500 - z850 + self.eps
        gamma_mid = -(t500 - t850) / dz_mid

        dz_low = z850 - z + self.eps
        gamma_low = -(t850 - t2m) / dz_low
        #
        # dz_deep = z500 - z + self.eps
        # gamma_deep = -(t500 - t2m) / dz_deep

        # --- 第二步：物理特征融合 ---
        # 将归一化后的 CAPE 和 CIN 在通道维度拼接 [Batch, 2, H, W]
        physics_input = torch.cat([cape, cin,gamma_low,gamma_mid], dim=1)

        # 提取物理门控偏置 [Batch, feature_dim, H, W]
        physics_bias = self.physics_extractor(physics_input)

        # --- 第三步：数据特征提取 ---
        # [Batch, feature_dim, H, W]
        data_gate = self.data_gate_conv(x)

        # --- 第四步：动态门控融合 ---
        # Gate = Sigmoid(数据驱动 + gamma * 物理先验)
        gate = torch.sigmoid(data_gate + self.gamma * physics_bias)

        # --- 第五步：应用门控 ---
        out = x * gate

        return out


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    batch_size = 4
    channels = 64
    H, W = 32, 32

    # 模拟输入数据
    x_features = torch.randn(batch_size, channels, H, W)
    cape_data = torch.rand(batch_size, 1, H, W) * 2000  # 模拟 0-2000 J/kg 的 CAPE
    cin_data = torch.rand(batch_size, 1, H, W) * 200  # 模拟 0-200 J/kg 的 CIN

    # 初始化模块
    gating_module = CAPECINGatingModule(feature_dim=channels)

    # 前向传播
    output_features = gating_module(x_features, cape_data, cin_data)

    print(f"输入特征维度: {x_features.shape}")
    print(f"输出特征维度: {output_features.shape}")
    print(f"当前可学习物理权重 gamma 的值: {gating_module.gamma.item()}")