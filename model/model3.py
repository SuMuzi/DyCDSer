import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicThermodynamicAttention(nn.Module):
    """
    模块 A: 动力-热力交叉注意力模块
    环流层 (Circulation) 作为 Query (寻找动力强迫区)
    边界层 (BL) + 对流层 (Trop) 作为 Key 和 Value (提供水汽和热力条件)
    """

    def __init__(self, cir_channels, thermo_channels, hidden_dim):
        super(DynamicThermodynamicAttention, self).__init__()
        # 将输入映射到相同的隐藏层维度
        self.query_conv = nn.Conv2d(cir_channels, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(thermo_channels, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(thermo_channels, hidden_dim, kernel_size=1)

        self.scale = hidden_dim ** -0.5

    def forward(self, f_cir, f_thermo):
        B, C, H, W = f_cir.size()

        # 1. 生成 Q, K, V 并展平空间维度 (B, C, H*W)
        Q = self.query_conv(f_cir).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C)
        K = self.key_conv(f_thermo).view(B, -1, H * W)  # (B, C, H*W)
        V = self.value_conv(f_thermo).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # 2. 计算交叉注意力权重 (寻找动力与热力配合的区域)
        # attention shape: (B, H*W, H*W)
        attention = torch.bmm(Q, K) * self.scale
        attention = F.softmax(attention, dim=-1)

        # 3. 聚合特征
        out = torch.bmm(attention, V)  # (B, H*W, C)

        out = out.permute(0, 2, 1).view(B, -1, H, W)  # 恢复空间形状 (B, C, H, W)

        return out


class StabilityGating(nn.Module):
    """
    模块 B: 层结稳定度门控模块
    利用垂直廓线 (Vertical Profile) 生成通道级或空间级的门控权重，控制对流触发
    """

    def __init__(self, vp_channels, hidden_dim):
        super(StabilityGating, self).__init__()
        # 使用全局平均池化提取整层稳定度特征，再通过 MLP 生成门控向量
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Conv2d(vp_channels, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1),
            nn.Sigmoid()  # 输出 0~1 之间的门控权重
        )

    def forward(self, f_vp):
        # 输出 shape: (B, hidden_dim, 1, 1)
        gate = self.mlp(f_vp)
        return gate


class PhysicallyConstrainedFusion(nn.Module):
    """
    主模型：融合四种气象数据进行降水降尺度
    """

    def __init__(self, bl_ch, trop_ch, cir_ch, vp_ch, hidden_dim=64):
        super(PhysicallyConstrainedFusion, self).__init__()

        # 简单的特征提取器 (实际应用中可替换为 ResNet 块等)
        self.extract_bl = nn.Conv2d(bl_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_trop = nn.Conv2d(trop_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_cir = nn.Conv2d(cir_ch, hidden_dim, kernel_size=3, padding=1)
        self.extract_vp = nn.Conv2d(vp_ch, hidden_dim, kernel_size=3, padding=1)

        # 注意力与门控模块
        thermo_channels = hidden_dim * 2  # 边界层和对流层拼接
        self.cross_attention = DynamicThermodynamicAttention(hidden_dim, thermo_channels, hidden_dim)
        self.stability_gate = StabilityGating(hidden_dim, hidden_dim)

        # 降尺度重建层 (例如通过反卷积或上采样+卷积提升分辨率)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 假设降尺度2倍
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.ReLU()  # 降水不能为负
        )

    def forward(self, bl, trop, cir, vp):
        # 1. 独立特征提取
        f_bl = F.relu(self.extract_bl(bl))
        f_trop = F.relu(self.extract_trop(trop))
        f_cir = F.relu(self.extract_cir(cir))
        f_vp = F.relu(self.extract_vp(vp))

        # 2. 拼接热力/水汽特征 (边界层 + 对流层)
        f_thermo = torch.cat([f_bl, f_trop], dim=1)

        # 3. 模块 A: 动力-热力交叉注意力
        f_cross = self.cross_attention(f_cir, f_thermo)

        # 4. 模块 B: 稳定度门控
        gate = self.stability_gate(f_vp)

        # 5. 门控调制 (物理意义：如果层结极其稳定，gate 趋近于0，抑制降水特征)
        f_fused = f_cross * gate

        # 6. 高分辨率降水重建
        hr_precip = self.reconstruction(f_fused)

        return hr_precip


# ==========================================
# 物理约束损失函数定义
# ==========================================
def physical_loss(hr_precip_pred, hr_precip_true, lr_precip_true, lambda_phy=0.1):
    """
    结合了 MSE 和水汽守恒物理约束的损失函数
    """
    # 1. 像素级重建损失 (常规的深度学习 Loss)
    mse_loss = F.mse_loss(hr_precip_pred, hr_precip_true)

    # 2. 物理约束损失 (水汽守恒：高分辨率降水的空间平均应等于低分辨率降水)
    # 假设 hr_precip_pred 是降尺度 2 倍的结果，我们用 AvgPool 退化回低分辨率
    hr_to_lr_pred = F.avg_pool2d(hr_precip_pred, kernel_size=2, stride=2)
    phy_loss = F.mse_loss(hr_to_lr_pred, lr_precip_true)

    # 总损失
    total_loss = mse_loss + lambda_phy * phy_loss
    return total_loss


# ==========================================
# 测试代码 (Dummy Data)
# ==========================================
if __name__ == "__main__":
    # 假设 Batch Size = 2, 空间分辨率为 32x32
    B, H, W = 2, 32, 32

    # 模拟输入数据 (通道数根据你的实际变量数设定)
    bl_data = torch.randn(B, 4, H, W)  # 边界层: 比如 u10, v10, t2m, d2m
    trop_data = torch.randn(B, 6, H, W)  # 对流层: 比如 850hPa/700hPa 的 q, t, w
    cir_data = torch.randn(B, 3, H, W)  # 环流层: 比如 500hPa 的 z, u, v
    vp_data = torch.randn(B, 2, H, W)  # 垂直廓线: 比如 CAPE, CIN

    # 模拟真实标签
    hr_label = torch.rand(B, 1, H * 2, W * 2)  # 64x64 高分辨率降水
    lr_label = F.avg_pool2d(hr_label, 2)  # 32x32 低分辨率降水 (用于物理约束)

    # 实例化模型
    model = PhysicallyConstrainedFusion(bl_ch=4, trop_ch=6, cir_ch=3, vp_ch=2, hidden_dim=64)

    # 前向传播
    pred_hr_precip = model(bl_data, trop_data, cir_data, vp_data)

    # 计算损失
    loss = physical_loss(pred_hr_precip, hr_label, lr_label, lambda_phy=0.5)

    print(f"输出高分辨率降水形状: {pred_hr_precip.shape}")  # 预期: [2, 1, 64, 64]
    print(f"计算得到的总损失: {loss.item():.4f}")
