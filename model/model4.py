import torch
import torch.nn as nn
import torch.nn.functional as F
# ==========================================
# 新增：要素间注意力模块 (Variable/Channel Attention)
# ==========================================
class VariableAttention(nn.Module):
    """
    要素间注意力机制 (基于通道注意力)
    物理意义：动态评估同一层内不同气象要素（如 T, q, u, v）对当前降水任务的重要性。
    """

    def __init__(self, num_variables, reduction=2):
        super(VariableAttention, self).__init__()
        # 确保 reduction 后的通道数至少为 1
        reduced_dim = max(1, num_variables // reduction)

        # Squeeze-and-Excitation 结构
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze: 提取每个要素的全局空间统计信息
            nn.Conv2d(num_variables, reduced_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, num_variables, kernel_size=1),
            nn.Sigmoid()  # Excitation: 生成 0~1 之间的要素权重
        )

    def forward(self, x):
        # x shape: (B, num_variables, H, W)
        # weight shape: (B, num_variables, 1, 1)
        weight = self.mlp(x)
        # 将权重乘回原特征，重要要素被增强，次要要素被抑制
        return x * weight


# (保留之前的 DynamicThermodynamicAttention 和 StabilityGating 模块...)
class DynamicThermodynamicAttention(nn.Module):
    def __init__(self, cir_channels, thermo_channels, hidden_dim):
        super(DynamicThermodynamicAttention, self).__init__()
        self.query_conv = nn.Conv2d(cir_channels, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(thermo_channels, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(thermo_channels, hidden_dim, kernel_size=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, f_cir, f_thermo):
        B, C, H, W = f_cir.size()
        Q = self.query_conv(f_cir).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key_conv(f_thermo).view(B, -1, H * W)
        V = self.value_conv(f_thermo).view(B, -1, H * W).permute(0, 2, 1)

        attention = torch.bmm(Q, K) * self.scale
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(attention, V)
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        return out


class StabilityGating(nn.Module):
    def __init__(self, vp_channels, hidden_dim):
        super(StabilityGating, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(vp_channels, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_vp):
        return self.mlp(f_vp)

# ==========================================
# 更新后的主模型
# ==========================================
class PhysicallyConstrainedFusion(nn.Module):
    def __init__(self, bl_ch, trop_ch, cir_ch, vp_ch, hidden_dim=64):
        super(PhysicallyConstrainedFusion, self).__init__()

        # 1. 为每种数据源添加【要素间注意力模块】
        self.var_attn_bl = VariableAttention(bl_ch)
        self.var_attn_trop = VariableAttention(trop_ch)
        self.var_attn_cir = VariableAttention(cir_ch)
        self.var_attn_vp = VariableAttention(vp_ch)

        # 2. 特征提取层
        self.extract_bl = nn.Conv2d(bl_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_trop = nn.Conv2d(trop_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_cir = nn.Conv2d(cir_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_vp = nn.Conv2d(vp_ch, hidden_dim, kernel_size=3, padding=1)

        # 3. 跨层融合与门控模块
        thermo_channels = hidden_dim * 2
        self.cross_attention = DynamicThermodynamicAttention(hidden_dim, thermo_channels, hidden_dim)
        self.stability_gate = StabilityGating(hidden_dim, hidden_dim)

        # 4. 降尺度重建层
        self.reconstruction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, bl, trop, cir, vp):
        # 步骤 1：先进行【要素间注意力】加权，筛选重要气象变量
        bl_weighted = self.var_attn_bl(bl)
        trop_weighted = self.var_attn_trop(trop)
        cir_weighted = self.var_attn_cir(cir)
        vp_weighted = self.var_attn_vp(vp)

        # 步骤 2：独立特征提取 (将不同数量的要素映射到统一的 hidden_dim)
        f_bl = F.relu(self.extract_bl(bl_weighted))
        f_trop = F.relu(self.extract_trop(trop_weighted))
        f_cir = F.relu(self.extract_cir(cir_weighted))
        f_vp = F.relu(self.extract_vp(vp_weighted))

        # 步骤 3：拼接热力/水汽特征 (边界层 + 对流层)
        f_thermo = torch.cat([f_bl, f_trop], dim=1)

        # 步骤 4: 动力-热力交叉注意力
        f_cross = self.cross_attention(f_cir, f_thermo)

        # 步骤 5: 稳定度门控
        gate = self.stability_gate(f_vp)
        f_fused = f_cross * gate

        # 步骤 6: 高分辨率降水重建
        hr_precip = self.reconstruction(f_fused)

        return hr_precip
