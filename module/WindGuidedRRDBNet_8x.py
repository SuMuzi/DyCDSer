"""
WindGuidedRRDBNet_8x.py
=======================
8倍降尺度的风场引导 RRDBNet

针对 ERA5/模式降水 8x 超分辨率任务设计，利用高分辨率风场 (u,v)
在多尺度渐进式上采样过程中进行物理引导。

核心设计:
    - 多尺度风场编码 (1x / 2x / 4x / 8x)
    - 渐进式风场引导 (每步上采样后空间注意力修正)
    - FiLM 全局调制 (RRDB trunk 前天气形势嵌入)
    - 物理约束 (散度注意力: 辐合区增强降水细节)
    - MC-Dropout 不确定性估计 (注意力方差量化风场-降水关系不确定度)
    - 注意力数据保存 (不绘图，直接输出 .npy)

Author: Auto-generated
Date: 2026-05-26
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


# ==================== 数据收集器 (用于测试循环累积保存) ====================

class AttentionDataCollector:
    """
    注意力数据收集器：在测试循环中累积数据，最后一次性保存
    
    使用方式:
        collector = AttentionDataCollector()
        
        for batch in test_loader:
            outputs = model(batch['lr'], batch['wind_hr'])
            mc_result = model.estimate_uncertainty(batch['lr'], batch['wind_hr'])
            collector.accumulate_attention(outputs)
            collector.accumulate_mc_dropout(mc_result)
        
        collector.save_all(save_dir='./test_results')
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """清空所有累积数据"""
        self.attention_data = {
            'attn_2x': [], 'attn_4x': [], 'attn_8x': [],
        }
        self.mc_dropout_data = {
            'sr_mean': [], 'sr_var': [], 'sr_std': [],
            'epistemic': [], 'aleatoric': [], 'total_unc': [], 'reliability': [],
            'attn_2x_mean': [], 'attn_2x_var': [], 'attn_2x_std': [],
            'attn_4x_mean': [], 'attn_4x_var': [], 'attn_4x_std': [],
            'attn_8x_mean': [], 'attn_8x_var': [], 'attn_8x_std': [],
        }

    @staticmethod
    def _to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        """Tensor → numpy (detach + cpu + numpy)"""
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy()

    def accumulate_attention(
        self,
        outputs: Dict[str, torch.Tensor],
    ):
        """
        累积单次前向传播的注意力数据
        
        Args:
            outputs: 模型 forward 返回的 dict
        """
        for k in ['attn_2x', 'attn_4x', 'attn_8x']:
            if k in outputs:
                self.attention_data[k].append(self._to_numpy(outputs[k]))

    def accumulate_mc_dropout(self, mc_result: Dict[str, torch.Tensor]):
        """
        累积 MC-Dropout 不确定性估计数据
        
        Args:
            mc_result: estimate_uncertainty 返回的 dict
        """
        for key in self.mc_dropout_data.keys():
            if key in mc_result:
                self.mc_dropout_data[key].append(self._to_numpy(mc_result[key]))

    def save_all(self, save_dir: str, prefix: str = 'test'):
        """
        一次性保存所有累积数据为单个 .npy 文件
        
        Args:
            save_dir: 保存目录
            prefix:   文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = {}

        # 保存注意力数据
        for key, values in self.attention_data.items():
            if values:
                arr = np.concatenate(values, axis=0)
                path = os.path.join(save_dir, f'{prefix}_{key}.npy')
                np.save(path, arr)
                saved_paths[key] = path
                print(f"  Saved {key}: shape={arr.shape}")

        # 保存 MC-Dropout 数据
        for key, values in self.mc_dropout_data.items():
            if values:
                arr = np.concatenate(values, axis=0)
                path = os.path.join(save_dir, f'{prefix}_{key}.npy')
                np.save(path, arr)
                saved_paths[key] = path
                print(f"  Saved {key}: shape={arr.shape}")

        # 保存元信息
        meta = {
            'total_batches_attention': len(self.attention_data['sr']),
            'total_batches_mc': len(self.mc_dropout_data['sr_mean']),
            'total_samples': len(self.attention_data['sr']) * (self.attention_data['sr'][0].shape[0] if self.attention_data['sr'] else 0),
        }
        meta_path = os.path.join(save_dir, f'{prefix}_meta.npy')
        np.save(meta_path, meta)
        saved_paths['meta'] = meta_path

        print(f"\n[AttentionDataCollector] All data saved to {save_dir}")
        return saved_paths


# ==================== 基础模块 (与 DS_BackBone_RRDB 保持一致) ===================="""



class ChannelCrossAttentionFusion(nn.Module):
    """
    通道维度的交叉注意力融合

    计算复杂度: O(C^2 * H * W)  (C=64时远低于空间注意力)

    物理意义:
        - Query: 降水特征 (我要修正什么?)
        - Key/Value: 其他气象要素 (温度/湿度/气压能提供什么信息?)
        - 注意力权重: 其他变量中哪些通道与降水修正最相关
    """

    def __init__(self, precip_dim: int = 64, other_dim: int = 64,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert precip_dim % num_heads == 0, "precip_dim 必须被 num_heads 整除"

        self.precip_dim = precip_dim
        self.other_dim = other_dim
        self.num_heads = num_heads
        self.head_dim = precip_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 投影层 (1x1卷积 = 逐像素线性投影)
        self.q_proj = nn.Conv2d(precip_dim, precip_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(other_dim, precip_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(other_dim, precip_dim, kernel_size=1)

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(precip_dim, precip_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

        # 门控：控制 other_feat 的融合比例
        self.gate = nn.Sequential(
            nn.Conv2d(precip_dim + other_dim, precip_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # LayerNorm (在通道维度)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=precip_dim)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=precip_dim)

        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Conv2d(precip_dim, precip_dim * 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(precip_dim * 2, precip_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, precip_feat: torch.Tensor, other_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            precip_feat: [B, C_p, H, W] 降水特征 (已校正)
            other_feat:  [B, C_o, H, W] 其他气象要素特征
        Returns:
            fused: [B, C_p, H, W] 融合后的特征
        """
        B, C, H, W = precip_feat.shape

        # ==== Cross-Attention ====
        # 投影
        Q = self.q_proj(precip_feat)  # [B, C, H, W]
        K = self.k_proj(other_feat)  # [B, C, H, W]
        V = self.v_proj(other_feat)  # [B, C, H, W]

        # 重塑为多头: [B, heads, C//heads, HW]
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)

        # 通道注意力: Q @ K^T -> [B, heads, C//heads, C//heads]
        # 注意: 这里是在"通道组"之间做注意力，空间维度作为batch
        attn = torch.einsum('bhdi,bhdj->bhij', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)  # 对 Key 维度归一化

        # 加权: attn @ V -> [B, heads, C//heads, HW]
        out = torch.einsum('bhij,bhdj->bhdi', attn, V)
        out = out.reshape(B, C, H, W)

        # 输出投影 + 残差
        out = self.out_proj(out)
        out = self.norm1(precip_feat + out)  # 残差连接1

        # ==== FFN ====
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)  # 残差连接2

        # ==== 门控融合 ====
        # gate=1 时完全使用 cross-attn 输出, gate=0 时保留原始降水特征
        gate = self.gate(torch.cat([precip_feat, other_feat], dim=1))
        fused = gate * out + (1 - gate) * precip_feat

        return fused
class UpsampleBlock(nn.Module):
    """2x 上采样：Conv -> PixelShuffle -> LeakyReLU"""

    def __init__(self, in_channels: int = 64, up_scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * (up_scale ** 2),
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB)"""

    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels + i * growth_rate, growth_rate,
                    kernel_size=3, padding=1, padding_mode='replicate'
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.local_fusion = nn.Conv2d(
            in_channels + num_layers * growth_rate,
            in_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.local_fusion(torch.cat(features, dim=1))
        return fused + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, in_channels: int, growth_rate: int = 32, beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.rdb1 = ResidualDenseBlock(in_channels, growth_rate)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_rate)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x


# ==================== 风场引导模块 (含 Dropout，用于 MC-Dropout) ====================

class MultiscaleWindEncoder(nn.Module):
    """
    多尺度风场编码器

    对 HR 风场进行多尺度下采样和编码，为 8x 渐进上采样的每个阶段
    提供对应分辨率的风场特征。

    输出尺度:
        1x : 相对 LR  (用于 trunk 前 FiLM 调制)
        2x : 2H x 2W  (用于第1次上采样后)
        4x : 4H x 4W  (用于第2次上采样后)
        8x : 8H x 8W  (用于第3次上采样后)
    """

    def __init__(self, wind_channels: int = 2, num_features: int = 64):
        super().__init__()

        # 1x 尺度 (相对 LR，8x 下采样)
        self.enc_1x = nn.Sequential(
            nn.Conv2d(wind_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_features, 3, padding=1),
        )

        # 2x 尺度 (4x 下采样)
        self.enc_2x = nn.Sequential(
            nn.Conv2d(wind_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_features, 3, padding=1),
        )

        # 4x 尺度 (2x 下采样)
        self.enc_4x = nn.Sequential(
            nn.Conv2d(wind_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_features, 3, padding=1),
        )

        # 8x 尺度 (原分辨率)
        self.enc_8x = nn.Sequential(
            nn.Conv2d(wind_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_features, 3, padding=1),
        )

    def forward(self, wind_hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            wind_hr: [B, 2, 8H, 8W] 高分辨率风场 (u, v)
        Returns:
            dict: {
                '1x': [B, C, H, W],
                '2x': [B, C, 2H, 2W],
                '4x': [B, C, 4H, 4W],
                '8x': [B, C, 8H, 8W]
            }
        """
        # 多尺度下采样
        wind_8x = wind_hr                                      # [B,2,8H,8W]
        wind_4x = F.avg_pool2d(wind_hr, kernel_size=2)         # [B,2,4H,4W]
        wind_2x = F.avg_pool2d(wind_hr, kernel_size=4)         # [B,2,2H,2W]
        wind_1x = F.avg_pool2d(wind_hr, kernel_size=8)         # [B,2,H,W]

        return {
            '1x': self.enc_1x(wind_1x),
            '2x': self.enc_2x(wind_2x),
            '4x': self.enc_4x(wind_4x),
            '8x': self.enc_8x(wind_8x),
        }


class ProgressiveWindGuidance(nn.Module):
    """
    渐进式风场引导模块 (含 Dropout2d，支持 MC-Dropout)

    在每次 2x 上采样后，利用对应尺度的风场特征进行:
        1) 散度注意力调制 (物理约束)
        2) 学习空间注意力
        3) 特征精修融合

    Attributes:
        spatial_attn:  降水特征 + 风场特征 -> 空间权重 [B,1,H,W]
        divergence_conv: 散度 -> 物理注意力 [B,1,H,W]
        refinement:    融合调制后的特征 [B,C,H,W]
    """

    def __init__(self, num_features: int = 64, dropout_rate: float = 0.1):
        super().__init__()

        # 空间注意力：融合降水特征和风场特征
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),  # MC-Dropout 位置1
            nn.Conv2d(num_features, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 散度 -> 注意力 (辐合区高注意力)
        self.divergence_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=dropout_rate),  # MC-Dropout 位置2
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 特征精修
        self.refinement = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),  # MC-Dropout 位置3
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

    @staticmethod
    def compute_divergence(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        计算二维风场散度: div = du/dx + dv/dy
        使用中心差分，边缘使用 replicate 填充。

        Args:
            u: [B, 1, H, W] 东西风分量
            v: [B, 1, H, W] 南北风分量
        Returns:
            div: [B, 1, H, W] 散度场
        """
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')

        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
        dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')

        return du_dx + dv_dy

    def forward(
        self,
        precip_feat: torch.Tensor,    # [B, C, H, W]
        wind_feat: torch.Tensor,      # [B, C, H, W]
        wind_uv: torch.Tensor         # [B, 2, H, W] 用于散度计算
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_feat: 修正后的特征 [B, C, H, W]
            attn_map:     融合注意力图 [B, 1, H, W] (用于可视化/不确定性估计)
        """
        # 1. 散度注意力 (物理约束)
        u, v = wind_uv[:, 0:1], wind_uv[:, 1:2]
        div = self.compute_divergence(u, v)           # [B,1,H,W]
        div_attn = self.divergence_conv(-div)         # 负散度=辐合 -> 高注意力

        # 2. 学习空间注意力
        concat = torch.cat([precip_feat, wind_feat], dim=1)
        learned_attn = self.spatial_attn(concat)      # [B,1,H,W]

        # 3. 融合注意力 (物理0.5 + 学习0.5)
        attn_map = 0.5 * div_attn + 0.5 * learned_attn

        # 4. 应用注意力调制
        #   高注意力区 -> 保留更多降水特征 (增强细节)
        #   低注意力区 -> 倾向风场特征 (修正位置/强度)
        modulated = precip_feat * attn_map + wind_feat * (1.0 - attn_map)

        # 5. 特征精修
        refined = self.refinement(torch.cat([modulated, wind_feat], dim=1))

        return refined, attn_map


# ==================== 主网络 ====================

class WindGuidedRRDBNet_8x(nn.Module):
    """
    8倍降尺度的风场引导 RRDBNet

    特点:
        - 3次 PixelShuffle 上采样 (2x -> 4x -> 8x)
        - 多尺度风场编码 (1x/2x/4x/8x)
        - 渐进式风场引导 (每步上采样后空间注意力修正)
        - 物理约束 (散度注意力调制辐合区)
        - FiLM 全局调制 (trunk 前嵌入天气形势)
        - MC-Dropout 不确定性估计
        - 注意力数据直接保存 (.npy)

    Args:
        in_channels:       输入降水通道数 (默认1)
        out_channels:      输出降水通道数 (默认1)
        wind_channels:     风场通道数 (默认2: u,v)
        num_features:      基础特征维度 (默认64)
        num_blocks:        RRDB 残差块数量 (默认23, ESRGAN配置)
        num_growth:        RDB 增长通道 (默认32)
        use_wind_guidance: 是否启用风场引导 (默认True)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        wind_channels: int = 2,
        num_features: int = 64,
        num_blocks: int = 23,
        num_growth: int = 32,
        use_wind_guidance: bool = True,
    ):
        super().__init__()

        self.use_wind_guidance = use_wind_guidance
        self.num_features = num_features

        # ========== 1. 初始特征提取 ==========
        self.conv_first = nn.Conv2d(
            in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=True
        )

        # ========== 2. 多尺度风场编码器 ==========
        if use_wind_guidance:
            self.wind_encoders = MultiscaleWindEncoder(wind_channels, num_features)

            # FiLM 生成器 (用于 trunk 前全局调制)
            self.film_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_features, num_features // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features // 4, num_features * 2, kernel_size=1),
            )

        # ========== 3. RRDB trunk ==========
        self.trunk = nn.Sequential(*[
            RRDB(num_features, num_growth, beta=0.2) for _ in range(num_blocks)
        ])
        self.conv_trunk = nn.Conv2d(
            num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True
        )

        # ========== 4. 渐进式上采样 + 风场引导 ==========
        # 第1次上采样: H,W -> 2H,2W
        self.upsample_1 = UpsampleBlock(num_features, up_scale=2)
        if use_wind_guidance:
            self.wind_guide_1 = ProgressiveWindGuidance(num_features, dropout_rate=0.1)

        # 第2次上采样: 2H,2W -> 4H,4W
        self.upsample_2 = UpsampleBlock(num_features, up_scale=2)
        if use_wind_guidance:
            self.wind_guide_2 = ProgressiveWindGuidance(num_features, dropout_rate=0.1)

        # 第3次上采样: 4H,4W -> 8H,8W
        self.upsample_3 = UpsampleBlock(num_features, up_scale=2)
        if use_wind_guidance:
            self.wind_guide_3 = ProgressiveWindGuidance(num_features, dropout_rate=0.1)

        # ========== 5. 输出重建 ==========
        self.conv_hr = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_last = nn.Conv2d(
            num_features, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """He 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def apply_film(self, feat: torch.Tensor, wind_feat_1x: torch.Tensor) -> torch.Tensor:
        """
        FiLM (Feature-wise Linear Modulation) 调制

        利用 1x 尺度的风场特征生成全局 gamma/beta，
        对降水初始特征进行缩放和平移，嵌入大尺度天气形势信息。

        Args:
            feat:         [B, C, H, W] 降水特征
            wind_feat_1x: [B, C, H, W] LR尺度风场特征
        Returns:
            modulated:    [B, C, H, W] 调制后的特征
        """
        film_params = self.film_gen(wind_feat_1x)          # [B, 2*C, 1, 1]
        gamma, beta = torch.chunk(film_params, 2, dim=1)   # 各 [B, C, 1, 1]

        gamma = torch.sigmoid(gamma) * 2.0                 # [0, 2]
        beta = torch.tanh(beta)                            # [-1, 1]

        return feat * gamma + beta

    def forward(
        self,
        x: torch.Tensor,                           # LR降水 [B, in_channels, H, W]
        wind_hr: Optional[torch.Tensor] = None     # HR风场 [B, wind_channels, 8H, 8W]
    ) -> Dict[str, torch.Tensor]:
        """
        标准前向传播

        Args:
            x:       低分辨率降水输入
            wind_hr: 高分辨率风场 (u, v)，空间尺寸为目标 HR 尺寸

        Returns:
            dict: {
                'sr':        超分辨率降水 [B, 1, 8H, 8W]
                'attn_2x':   2x 阶段注意力图 (可选)
                'attn_4x':   4x 阶段注意力图 (可选)
                'attn_8x':   8x 阶段注意力图 (可选)
            }
        """
        # ===== 1. 初始特征提取 =====
        feat = self.conv_first(x)  # [B, 64, H, W]

        wind_features = None
        if self.use_wind_guidance and wind_hr is not None:
            # 多尺度风场编码
            wind_features = self.wind_encoders(wind_hr)

            # FiLM 全局调制 (LR 尺度)
            feat = self.apply_film(feat, wind_features['1x'])

        # ===== 2. RRDB trunk =====
        trunk_feat = self.trunk(feat)
        trunk_feat = self.conv_trunk(trunk_feat)
        feat = feat + trunk_feat  # 长跳跃连接

        # ===== 3. 第1次上采样 (2x) =====
        feat = self.upsample_1(feat)  # [B, 64, 2H, 2W]

        attn_2x = None
        if self.use_wind_guidance and wind_features is not None:
            # 风场下采样到 2x 尺度用于散度计算
            wind_2x_uv = F.avg_pool2d(wind_hr, kernel_size=4)  # [B, 2, 2H, 2W]
            feat, attn_2x = self.wind_guide_1(
                feat, wind_features['2x'], wind_2x_uv
            )

        # ===== 4. 第2次上采样 (4x) =====
        feat = self.upsample_2(feat)  # [B, 64, 4H, 4W]

        attn_4x = None
        if self.use_wind_guidance and wind_features is not None:
            wind_4x_uv = F.avg_pool2d(wind_hr, kernel_size=2)  # [B, 2, 4H, 4W]
            feat, attn_4x = self.wind_guide_2(
                feat, wind_features['4x'], wind_4x_uv
            )

        # ===== 5. 第3次上采样 (8x) =====
        feat = self.upsample_3(feat)  # [B, 64, 8H, 8W]

        attn_8x = None
        if self.use_wind_guidance and wind_features is not None:
            feat, attn_8x = self.wind_guide_3(
                feat, wind_features['8x'], wind_hr  # 原分辨率风场
            )

        # ===== 6. 输出重建 =====
        feat = self.conv_hr(feat)
        out = self.conv_last(feat)

        # 降水非负约束
        out = torch.clamp(out, min=0.0)

        result = {'sr': out}

        # 返回注意力图用于可视化/诊断/不确定性估计
        if self.use_wind_guidance:
            if attn_2x is not None:
                result['attn_2x'] = attn_2x
            if attn_4x is not None:
                result['attn_4x'] = attn_4x
            if attn_8x is not None:
                result['attn_8x'] = attn_8x

        return result

    # ==================== MC-Dropout 不确定性估计 ====================

    @staticmethod
    def enable_mc_dropout(model: nn.Module):
        """
        启用模型中的 Dropout 层 (即使在 eval 模式下)
        这是 MC-Dropout 的核心：训练和测试时都使用 Dropout 进行贝叶斯近似
        """
        def _enable_dropout(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

        model.eval()
        model.apply(_enable_dropout)

    @torch.no_grad()
    def forward_mc_dropout(
        self,
        x: torch.Tensor,
        wind_hr: Optional[torch.Tensor],
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        MC-Dropout 多次采样前向传播

        Args:
            x:           LR降水 [B, 1, H, W]
            wind_hr:     HR风场 [B, 2, 8H, 8W]
            num_samples: MC采样次数 (默认30)

        Returns:
            dict: {
                'sr_mean':     预测均值 [B, 1, 8H, 8W]
                'sr_var':      预测方差 [B, 1, 8H, 8W]
                'sr_std':      预测标准差 [B, 1, 8H, 8W]
                'attn_2x_mean': 2x注意力均值 (可选)
                'attn_2x_var':  2x注意力方差 (可选)
                'attn_4x_mean': 4x注意力均值 (可选)
                'attn_4x_var':  4x注意力方差 (可选)
                'attn_8x_mean': 8x注意力均值 (可选)
                'attn_8x_var':  8x注意力方差 (可选)
            }
        """
        self.enable_mc_dropout(self)

        sr_samples = []
        attn_samples = {'attn_2x': [], 'attn_4x': [], 'attn_8x': []}

        for _ in range(num_samples):
            outputs = self.forward(x, wind_hr)
            sr_samples.append(outputs['sr'])

            for k in ['attn_2x', 'attn_4x', 'attn_8x']:
                if k in outputs:
                    attn_samples[k].append(outputs[k])

        # 预测统计量
        sr_stack = torch.stack(sr_samples, dim=0)  # [T, B, 1, H, W]
        result = {
            'sr_mean': sr_stack.mean(dim=0),
            'sr_var': sr_stack.var(dim=0),
            'sr_std': torch.sqrt(sr_stack.var(dim=0) + 1e-8),
        }

        # 注意力统计量 (方差大 = 风场-降水关系不确定度高)
        for k in ['attn_2x', 'attn_4x', 'attn_8x']:
            if attn_samples[k]:
                stack = torch.stack(attn_samples[k], dim=0)  # [T, B, 1, H, W]
                result[f'{k}_mean'] = stack.mean(dim=0)
                result[f'{k}_var'] = stack.var(dim=0)
                result[f'{k}_std'] = torch.sqrt(stack.var(dim=0) + 1e-8)

        return result

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        wind_hr: Optional[torch.Tensor],
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        对外接口：MC-Dropout 不确定性估计
        包含认知不确定性、偶然不确定性和可信度图
        """
        mc_result = self.forward_mc_dropout(x, wind_hr, num_samples)

        # 认知不确定性 = 预测方差 (模型参数不确定)
        epistemic = mc_result['sr_var']

        # 偶然不确定性 = 局部梯度变化 (数据噪声)
        sr_mean = mc_result['sr_mean']
        grad_x = torch.abs(sr_mean[:, :, :, 1:] - sr_mean[:, :, :, :-1])
        grad_y = torch.abs(sr_mean[:, :, 1:, :] - sr_mean[:, :, :-1, :])
        aleatoric_val = (grad_x.mean() + grad_y.mean()) / 2.0
        aleatoric = torch.ones_like(sr_mean) * aleatoric_val

        # 总不确定性
        total_unc = epistemic + aleatoric

        # 可信度图 (不确定性越大 -> 可信度越低)
        tau = sr_mean.mean() * 0.5 + 1e-6
        reliability = torch.exp(-total_unc / tau)
        reliability = torch.clamp(reliability, 0, 1)

        return {
            **mc_result,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total_unc': total_unc,
            'reliability': reliability,
        }

    # ==================== 注意力数据保存 (不绘图) ====================

    @torch.no_grad()
    def save_attention_data(
        self,
        x: torch.Tensor,
        wind_hr: Optional[torch.Tensor],
        save_dir: str,
        prefix: str = 'sample',
        target_hr: Optional[torch.Tensor] = None
    ) -> Dict[str, str]:
        """
        保存注意力相关数据为 .npy 文件 (不绘制任何图像)

        Args:
            x:         LR降水 [B, 1, H, W]
            wind_hr:   HR风场 [B, 2, 8H, 8W]
            save_dir:  保存目录
            prefix:    文件名前缀
            target_hr: HR降水真值 (可选，用于对比)

        Returns:
            saved_paths: 保存的文件路径字典
        """
        os.makedirs(save_dir, exist_ok=True)
        self.eval()

        outputs = self.forward(x, wind_hr)
        saved_paths = {}

        # 1. 输入/输出
        lr_path = os.path.join(save_dir, f'{prefix}_lr.npy')
        np.save(lr_path, x.cpu().numpy())
        saved_paths['lr'] = lr_path

        sr_path = os.path.join(save_dir, f'{prefix}_sr.npy')
        np.save(sr_path, outputs['sr'].cpu().numpy())
        saved_paths['sr'] = sr_path

        if wind_hr is not None:
            wind_path = os.path.join(save_dir, f'{prefix}_wind_hr.npy')
            np.save(wind_path, wind_hr.cpu().numpy())
            saved_paths['wind_hr'] = wind_path

        if target_hr is not None:
            gt_path = os.path.join(save_dir, f'{prefix}_gt.npy')
            np.save(gt_path, target_hr.cpu().numpy())
            saved_paths['gt'] = gt_path

        # 2. 多尺度注意力
        for k in ['attn_2x', 'attn_4x', 'attn_8x']:
            if k in outputs:
                attn_path = os.path.join(save_dir, f'{prefix}_{k}.npy')
                np.save(attn_path, outputs[k].cpu().numpy())
                saved_paths[k] = attn_path

        # 3. 统计信息
        stats = {}
        if 'attn_8x' in outputs:
            attn = outputs['attn_8x']
            stats['attn_8x_mean'] = attn.mean().item()
            stats['attn_8x_std'] = attn.std().item()
            stats['attn_8x_min'] = attn.min().item()
            stats['attn_8x_max'] = attn.max().item()

        stats_path = os.path.join(save_dir, f'{prefix}_stats.npy')
        np.save(stats_path, stats)
        saved_paths['stats'] = stats_path

        print(f"[save_attention_data] Saved to {save_dir}, prefix={prefix}")
        return saved_paths

    @torch.no_grad()
    def save_mc_dropout_data(
        self,
        x: torch.Tensor,
        wind_hr: Optional[torch.Tensor],
        save_dir: str,
        num_samples: int = 30,
        prefix: str = 'mc'
    ) -> Dict[str, str]:
        """
        保存 MC-Dropout 不确定性估计数据 (不绘图)

        Args:
            x:           LR降水
            wind_hr:     HR风场
            save_dir:    保存目录
            num_samples: MC采样次数
            prefix:      文件名前缀

        Returns:
            saved_paths: 保存的文件路径字典
        """
        os.makedirs(save_dir, exist_ok=True)

        result = self.estimate_uncertainty(x, wind_hr, num_samples)
        saved_paths = {}

        # 核心预测
        for key in ['sr_mean', 'sr_var', 'sr_std', 'epistemic', 'aleatoric', 'total_unc', 'reliability']:
            if key in result:
                path = os.path.join(save_dir, f'{prefix}_{key}.npy')
                np.save(path, result[key].cpu().numpy())
                saved_paths[key] = path

        # 注意力不确定性 (核心：attn_var 大 = 风场-降水关系不确定)
        for k in ['attn_2x', 'attn_4x', 'attn_8x']:
            mean_key = f'{k}_mean'
            var_key = f'{k}_var'
            std_key = f'{k}_std'
            if mean_key in result:
                np.save(os.path.join(save_dir, f'{prefix}_{mean_key}.npy'), result[mean_key].cpu().numpy())
                saved_paths[mean_key] = os.path.join(save_dir, f'{prefix}_{mean_key}.npy')
            if var_key in result:
                np.save(os.path.join(save_dir, f'{prefix}_{var_key}.npy'), result[var_key].cpu().numpy())
                saved_paths[var_key] = os.path.join(save_dir, f'{prefix}_{var_key}.npy')
            if std_key in result:
                np.save(os.path.join(save_dir, f'{prefix}_{std_key}.npy'), result[std_key].cpu().numpy())
                saved_paths[std_key] = os.path.join(save_dir, f'{prefix}_{std_key}.npy')

        # 保存输入参考
        np.save(os.path.join(save_dir, f'{prefix}_lr.npy'), x.cpu().numpy())
        saved_paths['lr'] = os.path.join(save_dir, f'{prefix}_lr.npy')

        if wind_hr is not None:
            np.save(os.path.join(save_dir, f'{prefix}_wind_hr.npy'), wind_hr.cpu().numpy())
            saved_paths['wind_hr'] = os.path.join(save_dir, f'{prefix}_wind_hr.npy')

        # 保存元信息
        meta = {
            'num_samples': num_samples,
            'sr_mean_range': [result['sr_mean'].min().item(), result['sr_mean'].max().item()],
            'sr_var_range': [result['sr_var'].min().item(), result['sr_var'].max().item()],
            'reliability_mean': result['reliability'].mean().item(),
        }
        meta_path = os.path.join(save_dir, f'{prefix}_meta.npy')
        np.save(meta_path, meta)
        saved_paths['meta'] = meta_path

        print(f"[save_mc_dropout_data] Saved MC-Dropout data to {save_dir}, prefix={prefix}")
        return saved_paths


# ==================== 损失函数 ====================

class WindGuidedLoss(nn.Module):
    """
    风场引导 8x 降尺度任务的复合损失函数

    包含:
        1) 像素级 L1 损失
        2) 梯度损失 (保留降水边缘)
        3) 风场注意力正则化 (防止注意力退化)
        4) 物理一致性损失 (辐合区降水正相关)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_grad: float = 0.1,
        lambda_attn: float = 0.01,
        lambda_phys: float = 0.05,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_attn = lambda_attn
        self.lambda_phys = lambda_phys

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: torch.Tensor,
        wind_hr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred:   模型输出 (含 'sr' 和注意力图)
            target: 高分辨率降水真值 [B,1,8H,8W]
            wind_hr: 高分辨率风场 (用于物理损失)
        Returns:
            total_loss: 总损失
            loss_dict:  各分项损失字典
        """
        sr = pred['sr']
        loss_dict = {}

        # 1. L1 像素损失
        loss_l1 = F.l1_loss(sr, target)
        loss_dict['l1'] = loss_l1.item()

        # 2. 梯度损失 (保留空间细节)
        grad_pred_x = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1])
        grad_pred_y = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :])
        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        loss_grad = (
            F.l1_loss(grad_pred_x, grad_target_x) +
            F.l1_loss(grad_pred_y, grad_target_y)
        )
        loss_dict['grad'] = loss_grad.item()

        total_loss = self.lambda_l1 * loss_l1 + self.lambda_grad * loss_grad

        # 3. 注意力正则化 (鼓励注意力在 0.5 附近，避免退化)
        if 'attn_8x' in pred:
            attn = pred['attn_8x']
            loss_attn = torch.mean((attn - 0.5) ** 2)
            loss_dict['attn'] = loss_attn.item()
            total_loss += self.lambda_attn * loss_attn

        # 4. 物理一致性损失 (辐合区应对应高降水)
        if wind_hr is not None and 'attn_8x' in pred:
            u, v = wind_hr[:, 0:1], wind_hr[:, 1:2]
            du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
            du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')
            dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
            dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')
            div = du_dx + dv_dy  # 散度

            # 辐合区 (div < 0) 应有高降水
            convergence_mask = (div < 0).float()
            if convergence_mask.sum() > 0:
                conv_precip = (sr * convergence_mask).sum() / (convergence_mask.sum() + 1e-6)
                div_precip = (sr * (1 - convergence_mask)).sum() / ((1 - convergence_mask).sum() + 1e-6)
                loss_phys = F.relu(div_precip - conv_precip + 0.5)  # margin=0.5
                loss_dict['phys'] = loss_phys.item()
                total_loss += self.lambda_phys * loss_phys

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ==================== 便捷函数 (不依赖 matplotlib) ====================

def compute_divergence_numpy(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    numpy 版散度计算 (用于后处理分析)

    Args:
        u: [H, W] 东西风分量
        v: [H, W] 南北风分量
    Returns:
        div: [H, W] 散度场
    """
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    return du_dx + dv_dy


# ==================== 使用示例 ====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = 2
    H_lr, W_lr = 32, 32       # LR 尺寸 (如 ERA5 0.25度)
    H_hr, W_hr = 256, 256     # HR 尺寸 (如雷达 0.03度)  8x

    # 模拟输入
    lr_precip = torch.rand(B, 1, H_lr, W_lr).to(device) * 5.0
    hr_wind = torch.randn(B, 2, H_hr, W_hr).to(device) * 3.0   # u,v ~ N(0,3)
    target_hr = torch.rand(B, 1, H_hr, W_hr).to(device) * 10.0

    # 模型
    model = WindGuidedRRDBNet_8x(
        in_channels=1,
        out_channels=1,
        wind_channels=2,
        num_features=64,
        num_blocks=23,
        num_growth=32,
        use_wind_guidance=True,
    ).to(device)

    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # ========== 1. 标准前向 + 保存注意力数据 ==========
    print("\n--- Standard Forward ---")
    outputs = model(lr_precip, hr_wind)
    print(f"  sr shape:        {outputs['sr'].shape}")
    print(f"  attn_2x shape:   {outputs['attn_2x'].shape}")
    print(f"  attn_4x shape:   {outputs['attn_4x'].shape}")
    print(f"  attn_8x shape:   {outputs['attn_8x'].shape}")

    # 保存为 .npy (不绘图)
    saved = model.save_attention_data(
        lr_precip, hr_wind,
        save_dir='./attention_data',
        prefix='test_standard',
        target_hr=target_hr
    )
    print(f"  Saved files: {list(saved.keys())}")

    # ========== 2. MC-Dropout 不确定性估计 + 保存 ==========
    print("\n--- MC-Dropout Uncertainty ---")
    mc_result = model.estimate_uncertainty(lr_precip, hr_wind, num_samples=10)
    print(f"  sr_mean shape:   {mc_result['sr_mean'].shape}")
    print(f"  sr_var range:    [{mc_result['sr_var'].min():.4f}, {mc_result['sr_var'].max():.4f}]")
    print(f"  reliability mean: {mc_result['reliability'].mean():.4f}")

    if 'attn_8x_var' in mc_result:
        print(f"  attn_8x_var range: [{mc_result['attn_8x_var'].min():.6f}, {mc_result['attn_8x_var'].max():.6f}]")

    # 保存 MC-Dropout 数据
    saved_mc = model.save_mc_dropout_data(
        lr_precip, hr_wind,
        save_dir='./mc_dropout_data',
        num_samples=10,
        prefix='test_mc'
    )
    print(f"  Saved MC files: {list(saved_mc.keys())}")

    # ========== 3. 损失计算 ==========
    print("\n--- Loss Computation ---")
    criterion = WindGuidedLoss(lambda_l1=1.0, lambda_grad=0.1, lambda_attn=0.01, lambda_phys=0.05)
    loss, loss_dict = criterion(outputs, target_hr, hr_wind)
    print(f"  Loss dict: {loss_dict}")

    print("\nDone. All data saved as .npy files (no figures plotted).")
