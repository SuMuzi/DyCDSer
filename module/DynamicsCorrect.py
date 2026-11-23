import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .blocks import depthwise_separable_conv_block

class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合：precip_feat作为Query，other_feat作为Key/Value
    适用于：让降水特征主动查询其他变量中的相关信息
    """
    def __init__(self, precip_dim=64, other_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = precip_dim // num_heads
        
        # Q from precip, K/V from other
        self.q_proj = nn.Conv2d(precip_dim, precip_dim, 1)
        self.k_proj = nn.Conv2d(other_dim, precip_dim, 1)
        self.v_proj = nn.Conv2d(other_dim, precip_dim, 1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(precip_dim, precip_dim, 1)
        
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(precip_dim * 2, precip_dim, 3, padding=1),
            nn.BatchNorm2d(precip_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, precip_feat, other_feat):
        B, C, H, W = precip_feat.shape
        
        # 投影
        Q = self.q_proj(precip_feat)  # [B, C, H, W]
        K = self.k_proj(other_feat)
        V = self.v_proj(other_feat)
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.other_dim // self.num_heads, H * W)  # 修正
        
        # 注意力计算
        attn = torch.einsum('bhdn,bhem->bdhem', Q, K) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bhdnm,bhfm->bhdfn', attn, V)
        out = out.reshape(B, -1, H, W)
        
        out = self.out_proj(out)
        
        # 残差融合
        fused = self.fusion(torch.cat([precip_feat, out], dim=1))
        return fused
# ==================== RRDB 模块 ====================
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


class WindErrorExtractor(nn.Module):
    """
    提取风速差异的关键物理特征：
    - 散度误差（辐合/辐散偏差）→ 直接影响降水
    - 涡度误差（旋转特征）
    - 垂直风切变误差 → 对流触发条件
    """

    def __init__(self, in_ch: int = 2):  # 输入u,v风速差异
        super().__init__()

        # 散度计算核 (∂u/∂x + ∂v/∂y)
        self.div_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.div_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        div_kernel_x = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).view(1, 1, 3, 3)
        div_kernel_y = torch.tensor([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]).view(1, 1, 3, 3)
        self.div_x.weight = nn.Parameter(div_kernel_x, requires_grad=False)
        self.div_y.weight = nn.Parameter(div_kernel_y, requires_grad=False)

        # 涡度计算核 (∂v/∂x - ∂u/∂y)
        self.vort_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.vort_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.vort_x.weight = nn.Parameter(div_kernel_x, requires_grad=False)  # ∂/∂x
        self.vort_y.weight = nn.Parameter(div_kernel_y, requires_grad=False)  # ∂/∂y

        # 特征编码
        self.encoder = nn.Sequential(
            depthwise_separable_conv_block(4, 64),
            depthwise_separable_conv_block(64, 128),
            nn.Conv2d(128, 64, 1),
        )

    def forward(self, u_diff: torch.Tensor, v_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_diff: 纬向风速差异 (模型 - 观测) [B, 1, H, W]
            v_diff: 经向风速差异 [B, 1, H, W]
        Returns:
            error_features: 动力误差特征 [B, 32, H, W]
        """
        # 计算散度误差（关键：正散度→辐散→抑制降水，负散度→辐合→增强降水）
        div_error = self.div_x(u_diff) + self.div_y(v_diff)  # [B, 1, H, W]

        # 计算涡度误差
        vort_error = self.vort_x(v_diff) - self.vort_y(u_diff)  # [B, 1, H, W]

        # 组合特征
        concat = torch.cat([u_diff, v_diff, div_error, vort_error], dim=1)  # [B, 4, H, W]
        features = self.encoder(concat)  # [B, 32, H, W]

        return features, div_error  # 返回div_error用于显式约束


class WindGuidedPrecipitationAdjustment(nn.Module):
    """
    根据风速差异动态调整降水融合权重和强度
    核心逻辑：散度误差修正 → 降水强度调整
    """

    def __init__(self, precip_dim: int = 64, wind_feat_dim: int = 32):
        super().__init__()

        # 门控生成：根据动力误差调整降水融合策略
        # 输入：降水特征 + 动力误差特征
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(precip_dim + wind_feat_dim, precip_dim, 3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(precip_dim, 2, 3, padding=1, padding_mode='replicate'),  # 2通道：强度调整因子 + 位置偏移
            nn.Tanh()  # 输出范围[-1, 1]，表示增减
        )

        # 强度修正头：基于散度误差直接计算降水强度修正量
        # 物理假设：散度误差ΔD → 垂直速度误差Δω → 降水误差ΔP
        self.intensity_corrector = nn.Sequential(
            nn.Conv2d(wind_feat_dim, wind_feat_dim, 1, padding_mode='replicate'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, 1),
            nn.Tanh()  # 修正比例 [-1, 1]，-1表示减半，1表示加倍
        )

        # 位置偏移头（光流风格）：风误差导致平流位置偏移
        self.position_offset = nn.Sequential(
            nn.Conv2d(wind_feat_dim, wind_feat_dim, 3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(wind_feat_dim, 2, 3, padding=1, padding_mode='replicate'),  # 2通道：dx, dy偏移
            nn.Tanh()
        )

    def forward(
            self,
            precip_feat: torch.Tensor,  # 降水特征（如多源融合特征）
            wind_error_feat: torch.Tensor,  # 风速差异特征
            div_error: torch.Tensor,  # 散度误差（显式物理量）
            raw_precip: torch.Tensor  # 原始降水输入（用于修正）
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            adjusted_precip: 调整后的降水 [B, 1, H, W]
            adjustments: 包含各调整量的字典（用于可视化/诊断）
        """
        # 1. 空间自适应门控（哪些区域需要调整）
        combined = torch.cat([precip_feat, wind_error_feat], dim=1)
        gate = self.spatial_gate(combined)  # [B, 2, H, W]

        intensity_gate = gate[:, 0:1]  # 强度调整倾向
        position_gate = gate[:, 1:2]  # 位置调整倾向

        # 2. 强度修正（基于散度误差）
        # 散度<0（辐合不足）→ 降水应增强；散度>0（辐散）→ 降水应减弱
        # 注意div_error是模型-观测，正表示模式辐散过强（应减降水）
        intensity_delta = self.intensity_corrector(wind_error_feat)  # [-1, 1]

        # 物理关系：负散度误差（辐合不足）→ 增加降水
        # 映射：div_error ∈ [-max, +max] → 修正因子 ∈ [0.5, 2.0]
        # 负相关：div_error越大（越正），降水修正应越小
        phys_correction = torch.exp(-div_error * 2.0)  # 经验系数，可学习
        phys_correction = torch.clamp(phys_correction, 0.5, 2.0)

        # 门控融合：数据驱动修正 + 物理驱动修正
        correction_factor = 1.0 + 0.5 * intensity_gate * intensity_delta + 0.5 * (phys_correction - 1.0)
        correction_factor = torch.clamp(correction_factor, 0.3, 3.0)  # 安全范围

        adjusted_intensity = raw_precip * correction_factor

        # 3. 位置修正（风场平流误差导致降水位置偏移）
        # 使用grid_sample进行空间扭曲补偿
        offset = self.position_offset(wind_error_feat)  # [B, 2, H, W]
        offset = offset * position_gate  # 门控控制是否启用位置修正

        # 生成采样网格（基于offset）
        B, C, H, W = adjusted_intensity.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=adjusted_intensity.device),
            torch.linspace(-1, 1, W, device=adjusted_intensity.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # 应用偏移（归一化到[-1,1]坐标系）
        offset_norm = offset.permute(0, 2, 3, 1)  # [B, H, W, 2]
        offset_norm[:, :, :, 0] /= W / 2  # x方向归一化
        offset_norm[:, :, :, 1] /= H / 2  # y方向归一化

        # 反向采样（补偿风场误差导致的平流偏移）
        adjusted_position = F.grid_sample(
            adjusted_intensity,
            grid + offset_norm,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        adjustments = {
            'intensity_factor': correction_factor,
            'position_offset': offset,
            'div_error': div_error,
            'gate_intensity': intensity_gate,
            'gate_position': position_gate
        }

        return adjusted_position, adjustments


class MultiVariableFusionWithWindCorrection(nn.Module):
    """
    完整的多变量融合网络：融合降水、温度、湿度等，并利用风速差异校正
    """

    def __init__(
            self,
            precip_channels: int = 1,  # 多源降水输入（如雷达、卫星、模式）
            other_channels: int = 17,  # 其他变量（温度、湿度）
            wind_channels: int = 2,  # u, v风
            base_dim: int = 64
    ):
        super().__init__()

        # 各变量编码器
        self.precip_encoder = nn.Sequential(
            depthwise_separable_conv_block(precip_channels, base_dim),
            depthwise_separable_conv_block(base_dim, 2 * base_dim),
            nn.Conv2d(2 * base_dim, base_dim, 1),
        )

        self.other_encoder = nn.Sequential(
            depthwise_separable_conv_block(other_channels, base_dim),
            depthwise_separable_conv_block(base_dim, 2 * base_dim),
            nn.Conv2d(2 * base_dim, base_dim, 1),
        )

        # 风速差异提取
        self.wind_error_extractor = WindErrorExtractor()

        # 风速引导的降水调整
        self.wind_adjustment = WindGuidedPrecipitationAdjustment(
            precip_dim=base_dim,
            wind_feat_dim=base_dim
        )

        # 多变量融合（降水+温度+湿度）
        self.fusion = nn.Sequential(
            depthwise_separable_conv_block(base_dim + base_dim, base_dim),
            depthwise_separable_conv_block(base_dim, 2 * base_dim),
            nn.Conv2d(2 * base_dim, base_dim, 1),
            nn.LeakyReLU(0.1)
        )

        # RRDB 深层特征提取（在融合后进一步增强特征表达）
        self.rrdb = RRDB(in_channels=base_dim, num_blocks=3, growth_rate=32)

        # 输出头
        self.precip_head = nn.Sequential(
            depthwise_separable_conv_block(base_dim, base_dim // 2),
            nn.Conv2d(base_dim // 2, 1, 1),
            nn.LeakyReLU(0.1)
        )

        # 其他变量输出（可选，保持多变量一致性）
        self.temp_head = nn.Conv2d(base_dim, 1, 1)
        self.rh_head = nn.Conv2d(base_dim, 1, 1)

    def forward(
            self,
            precip_inputs: torch.Tensor,  # [B, C_p, H, W] 多源降水
            other_vars: torch.Tensor,  # [B, C_o, H, W] 温度、湿度等
            wind_model: torch.Tensor,  # [B, 2, H, W] 模式风场
            wind_obs: Optional[torch.Tensor] = None,  # 观测风场（用于计算差异）
            return_adjustments: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # 编码各变量
        precip_feat = self.precip_encoder(precip_inputs)  # [B, base_dim, H, W]
        other_feat = self.other_encoder(other_vars)  # [B, base_dim/2, H, W]

        adjustments = None


        # 计算风速差异
        u_diff = wind_model[:, 0:1] - wind_obs[:, 0:1]  # 模式-观测
        v_diff = wind_model[:, 1:2] - wind_obs[:, 1:2]

        # 提取动力误差特征
        wind_feat, div_error = self.wind_error_extractor(u_diff, v_diff)

        # 调整降水特征和原始降水（取precip_inputs第一通道作为代表）
        raw_precip = precip_inputs[:, 0:1]  # 假设第一通道是主要降水源

        adjusted_precip, adjustments = self.wind_adjustment(
            precip_feat, wind_feat, div_error, raw_precip
        )

        # 将调整后的降水重新编码（或与原特征融合）
        # 这里简单替换第一通道特征，实际可设计更复杂的融合
        precip_feat_adjusted = precip_feat + self.precip_encoder(
            torch.cat([adjusted_precip, torch.zeros_like(precip_inputs[:, 1:])], dim=1)
        ) - self.precip_encoder(
            torch.cat([raw_precip, torch.zeros_like(precip_inputs[:, 1:])], dim=1)
        )


        # 多变量融合
        fused = self.fusion(torch.cat([precip_feat_adjusted, other_feat], dim=1))

        # RRDB 深层特征提取（增强融合后的特征表达能力）
        enhanced_feat = self.rrdb(fused)

        # 输出
        precip_out = self.precip_head(enhanced_feat)
        # temp_out = self.temp_head(enhanced_feat)
        # rh_out = self.rh_head(enhanced_feat)

        outputs = {
            'precipitation': precip_out,
            # 'temperature': temp_out,
            # 'humidity': rh_out,
            'precip_adjusted': adjusted_precip if adjusted_precip is not None else precip_out
        }

        if return_adjustments and adjustments is not None:
            outputs['adjustments'] = adjustments

        return outputs


# ==================== 使用示例 ====================

def demo_wind_guided_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("风速差异引导的降水融合与校正")
    print("=" * 70)

    B, H, W = 2, 64, 64

    # 模拟输入
    # 1. 多源降水（模式+卫星+雷达）
    precip_inputs = torch.rand(B, 3, H, W).to(device) * 10  # 3个来源

    # 2. 其他变量（温度距平、相对湿度）
    other_vars = torch.rand(B, 2, H, W).to(device)
    other_vars[:, 1] = other_vars[:, 1] * 100  # RH 0-100%

    # 3. 风场（模式 vs 观测）
    # 模拟模式风场（有系统性偏差：风速偏强，方向偏东）
    wind_model = torch.zeros(B, 2, H, W).to(device)
    wind_model[:, 0] = 5.0 + torch.randn(B, H, W).to(device) * 2  # u风，均值5m/s
    wind_model[:, 1] = 2.0 + torch.randn(B, H, W).to(device) * 1  # v风

    # 观测风场（真值，模式偏差：模式辐合偏强）
    wind_obs = wind_model.clone()
    wind_obs[:, 0] -= 1.5  # 模式u风偏强1.5m/s
    wind_obs[:, 1] -= 0.8  # 模式v风偏强0.8m/s
    # 添加位置偏移导致的散度差异（模式在中心有虚假辐合）
    center_mask = torch.zeros(H, W).to(device)
    center_mask[H // 4:3 * H // 4, W // 4:3 * W // 4] = 1.0
    wind_obs[:, 0] -= center_mask.unsqueeze(0) * 2.0  # 观测在中心辐合更弱

    print(f"\n输入统计:")
    print(f"  模式风场u: 均值={wind_model[:, 0].mean():.2f}, 观测u: 均值={wind_obs[:, 0].mean():.2f}")
    print(f"  风速差异: 均值={(wind_model - wind_obs).mean():.2f}m/s")

    # 初始化模型
    model = MultiVariableFusionWithWindCorrection(
        precip_channels=3,
        other_channels=2,
        wind_channels=2,
        base_dim=64,
        use_wind_correction=True
    ).to(device)

    # 前向传播
    outputs = model(
        precip_inputs,
        other_vars,
        wind_model,
        wind_obs,
        return_adjustments=True
    )

    print(f"\n输出结果:")
    print(
        f"  融合降水: {outputs['precipitation'].shape}, 范围[{outputs['precipitation'].min():.2f}, {outputs['precipitation'].max():.2f}]")
    print(f"  温度: {outputs['temperature'].shape}")
    print(f"  湿度: {outputs['humidity'].shape}")

    # 检查风速校正效果
    if 'adjustments' in outputs:
        adj = outputs['adjustments']
        print(f"\n风速校正细节:")
        print(f"  散度误差范围: [{adj['div_error'].min():.4f}, {adj['div_error'].max():.4f}]")
        print(
            f"  强度修正因子: 均值={adj['intensity_factor'].mean():.3f}, 范围[{adj['intensity_factor'].min():.2f}, {adj['intensity_factor'].max():.2f}]")
        print(f"  位置偏移量: 均值={adj['position_offset'].abs().mean():.3f}像素")

        # 诊断：散度误差大的区域，修正因子应显著偏离1
        high_div_error_mask = (adj['div_error'] > adj['div_error'].mean() + adj['div_error'].std()).float()
        avg_factor_high_error = (adj['intensity_factor'] * high_div_error_mask).sum() / (
                high_div_error_mask.sum() + 1e-6)
        print(f"  高散度误差区修正因子: {avg_factor_high_error:.3f} (应显著偏离1.0)")

    # 计算损失
    targets = {
        'precipitation': torch.rand(B, 1, H, W).to(device) * 15,
        'temperature': torch.rand(B, 1, H, W).to(device) * 30,
        'humidity': torch.rand(B, 1, H, W).to(device) * 100
    }

    criterion = PhysicsConstrainedLoss(lambda_mass=0.1, lambda_continuity=0.2, lambda_budget=0.5)
    losses = criterion(outputs, targets, wind_model, wind_obs)

    print(f"\n损失分解:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    return model, outputs, losses


if __name__ == "__main__":
    model, outputs, losses = demo_wind_guided_fusion()