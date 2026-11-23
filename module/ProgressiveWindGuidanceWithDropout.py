import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


# ==================== 1. 带 Dropout 的引导模块 ====================

class ProgressiveWindGuidanceWithDropout(nn.Module):
    """
    在 ProgressiveWindGuidance 基础上加入 MC-Dropout
    用于不确定性估计
    """

    def __init__(self, num_features: int = 64, dropout_rate: float = 0.1):
        super().__init__()

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),  # MC-Dropout 位置1
            nn.Conv2d(num_features, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 散度注意力
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
    def compute_divergence(u, v):
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')
        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
        dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')
        return du_dx + dv_dy

    def forward(self, precip_feat, wind_feat, wind_uv):
        # 散度注意力
        u, v = wind_uv[:, 0:1], wind_uv[:, 1:2]
        div = self.compute_divergence(u, v)
        div_attn = self.divergence_conv(-div)

        # 学习空间注意力
        concat = torch.cat([precip_feat, wind_feat], dim=1)
        learned_attn = self.spatial_attn(concat)

        # 融合注意力
        attn_map = 0.5 * div_attn + 0.5 * learned_attn

        # 特征调制
        modulated = precip_feat * attn_map + wind_feat * (1.0 - attn_map)
        refined = self.refinement(torch.cat([modulated, wind_feat], dim=1))

        return refined, attn_map


# ==================== 2. MC-Dropout 推理引擎 ====================

class WindGuidedUncertaintyEstimator:
    """
    风场引导降水超分辨率的不确定性估计器

    使用 MC-Dropout 进行贝叶斯近似推理:
        - 多次前向传播 (T次)
        - 收集每次的注意力图和预测结果
        - 计算均值 (预测) 和方差 (不确定性)
    """

    def __init__(
            self,
            model: nn.Module,
            num_samples: int = 50,  # MC采样次数
            dropout_layers: Optional[List[str]] = None
    ):
        self.model = model
        self.num_samples = num_samples
        self.dropout_layers = dropout_layers or ['wind_guide_1', 'wind_guide_2', 'wind_guide_3']

    def _enable_mc_dropout(self):
        """启用模型中的 Dropout 层 (即使在 eval 模式下)"""

        def enable_dropout(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()  # 强制 Dropout 保持训练模式

        self.model.eval()
        self.model.apply(enable_dropout)

    @torch.no_grad()
    def estimate(
            self,
            lr_precip: torch.Tensor,
            wind_hr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        MC-Dropout 不确定性估计

        Args:
            lr_precip: [B, 1, H, W] 低分辨率降水
            wind_hr:   [B, 2, 8H, 8W] 高分辨率风场

        Returns:
            dict: {
                'sr_mean':      预测均值 [B, 1, 8H, 8W]
                'sr_var':       预测方差 [B, 1, 8H, 8W]
                'sr_std':       预测标准差 [B, 1, 8H, 8W]
                'attn_mean':    注意力均值 [B, 1, 8H, 8W]
                'attn_var':     注意力方差 [B, 1, 8H, 8W]
                'epistemic':    认知不确定性 (模型不确定)
                'aleatoric':    偶然不确定性 (数据噪声)
                'total_unc':    总不确定性
                'reliability':  可信度图 [B, 1, 8H, 8W] (0-1)
            }
        """
        self._enable_mc_dropout()
        device = lr_precip.device

        # 收集多次采样结果
        sr_samples = []
        attn_2x_samples = []
        attn_4x_samples = []
        attn_8x_samples = []

        for _ in range(self.num_samples):
            outputs = self.model(lr_precip, wind_hr)

            sr_samples.append(outputs['sr'])
            if 'attn_2x' in outputs:
                attn_2x_samples.append(outputs['attn_2x'])
            if 'attn_4x' in outputs:
                attn_4x_samples.append(outputs['attn_4x'])
            if 'attn_8x' in outputs:
                attn_8x_samples.append(outputs['attn_8x'])

        # 堆叠采样结果
        sr_stack = torch.stack(sr_samples, dim=0)  # [T, B, 1, H, W]
        attn_8x_stack = torch.stack(attn_8x_samples, dim=0) if attn_8x_samples else None

        # ========== 预测不确定性 ==========
        sr_mean = sr_stack.mean(dim=0)  # [B, 1, H, W]
        sr_var = sr_stack.var(dim=0)  # [B, 1, H, W]
        sr_std = torch.sqrt(sr_var + 1e-8)

        result = {
            'sr_mean': sr_mean,
            'sr_var': sr_var,
            'sr_std': sr_std,
        }

        # ========== 注意力不确定性 (风场-降水关系不确定度) ==========
        if attn_8x_stack is not None:
            attn_mean = attn_8x_stack.mean(dim=0)  # [B, 1, H, W]
            attn_var = attn_8x_stack.var(dim=0)  # [B, 1, H, W]
            attn_std = torch.sqrt(attn_var + 1e-8)

            result['attn_mean'] = attn_mean
            result['attn_var'] = attn_var
            result['attn_std'] = attn_std

            # 注意力熵: 衡量注意力分布的混乱程度
            # H = -p*log(p) - (1-p)*log(1-p), p = attn_mean
            p = torch.clamp(attn_mean, 1e-8, 1 - 1e-8)
            attn_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            result['attn_entropy'] = attn_entropy

        # ========== 分解不确定性 ==========
        # 认知不确定性 (Epistemic): 模型参数不确定
        # 偶然不确定性 (Aleatoric): 数据固有噪声

        # 使用概率分布建模: 假设预测是 N(mean, var_data)
        # 认知不确定性 = 均值方差
        epistemic = sr_var  # [B, 1, H, W]

        # 偶然不确定性 = 平均预测方差 (需要模型输出 log_var)
        # 这里简化为 sr_mean 的局部梯度变化
        grad_x = torch.abs(sr_mean[:, :, :, 1:] - sr_mean[:, :, :, :-1])
        grad_y = torch.abs(sr_mean[:, :, 1:, :] - sr_mean[:, :, :-1, :])
        aleatoric = (grad_x.mean() + grad_y.mean()) / 2.0
        aleatoric = torch.ones_like(sr_mean) * aleatoric

        result['epistemic'] = epistemic
        result['aleatoric'] = aleatoric
        result['total_unc'] = epistemic + aleatoric

        # ========== 可信度图 ==========
        # 不确定性越大 -> 可信度越低
        # reliability = exp(-total_unc / tau), tau 为温度系数
        tau = sr_mean.mean() * 0.5 + 1e-6
        reliability = torch.exp(-result['total_unc'] / tau)
        result['reliability'] = torch.clamp(reliability, 0, 1)

        return result

    def estimate_with_bootstrap(
            self,
            lr_precip: torch.Tensor,
            wind_hr_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Bootstrap风场扰动: 对风场加噪声，测试模型对风场输入的敏感度

        Args:
            wind_hr_list: 多个扰动风场样本 (如不同初始条件的风场)
        """
        self._enable_mc_dropout()

        all_sr = []
        all_attn = []

        for wind_hr in wind_hr_list:
            for _ in range(self.num_samples // len(wind_hr_list)):
                out = self.model(lr_precip, wind_hr)
                all_sr.append(out['sr'])
                if 'attn_8x' in out:
                    all_attn.append(out['attn_8x'])

        sr_stack = torch.stack(all_sr, dim=0)
        result = {
            'sr_mean': sr_stack.mean(dim=0),
            'sr_var': sr_stack.var(dim=0),
            'sr_std': torch.sqrt(sr_stack.var(dim=0) + 1e-8),
        }

        if all_attn:
            attn_stack = torch.stack(all_attn, dim=0)
            result['attn_var'] = attn_stack.var(dim=0)
            result['attn_std'] = torch.sqrt(attn_stack.var(dim=0) + 1e-8)

        return result


# ==================== 3. 可视化 ====================

def visualize_uncertainty(
        uncertainty_result: Dict[str, torch.Tensor],
        lr_precip: torch.Tensor,
        wind_hr: torch.Tensor,
        target_hr: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
):
    """
    可视化不确定性分析结果

    子图布局:
        1. LR输入降水
        2. HR风场散度 (物理参考)
        3. 预测均值 (sr_mean)
        4. 注意力均值 (attn_mean)
        5. 注意力方差 (attn_var)  <-- 风场-降水关系不确定度
        6. 预测标准差 (sr_std)     <-- 预报可信度
        7. 认知不确定性 (epistemic)
        8. 可信度图 (reliability)
    """
    import matplotlib.pyplot as plt

    batch_idx = 0  # 可视化第一个样本
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # 1. LR 降水
    ax = axes[0]
    im = ax.imshow(lr_precip[batch_idx, 0].cpu().numpy(), cmap='Blues')
    ax.set_title('LR Precipitation Input')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. 风场散度
    ax = axes[1]
    u = wind_hr[batch_idx, 0].cpu().numpy()
    v = wind_hr[batch_idx, 1].cpu().numpy()
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    div = du_dx + dv_dy
    im = ax.imshow(-div, cmap='RdBu_r', vmin=-5, vmax=5)  # 负散度=辐合
    ax.set_title('Wind Convergence (-div)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. 预测均值
    ax = axes[2]
    im = ax.imshow(uncertainty_result['sr_mean'][batch_idx, 0].cpu().numpy(), cmap='YlOrRd')
    ax.set_title('SR Mean Prediction')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. 注意力均值
    if 'attn_mean' in uncertainty_result:
        ax = axes[3]
        im = ax.imshow(uncertainty_result['attn_mean'][batch_idx, 0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title('Attention Mean (Wind-Precip Focus)')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 5. 注意力方差 (核心: 风场-降水关系不确定度)
    if 'attn_var' in uncertainty_result:
        ax = axes[4]
        attn_var = uncertainty_result['attn_var'][batch_idx, 0].cpu().numpy()
        im = ax.imshow(attn_var, cmap='hot')
        ax.set_title(f'Attention Variance (Uncertainty)\\nMax={attn_var.max():.4f}')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 6. 预测标准差
    ax = axes[5]
    sr_std = uncertainty_result['sr_std'][batch_idx, 0].cpu().numpy()
    im = ax.imshow(sr_std, cmap='hot')
    ax.set_title(f'Prediction Std (Forecast Uncertainty)\\nMax={sr_std.max():.2f}')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 7. 认知不确定性
    ax = axes[6]
    epis = uncertainty_result['epistemic'][batch_idx, 0].cpu().numpy()
    im = ax.imshow(epis, cmap='hot')
    ax.set_title('Epistemic Uncertainty (Model Uncertain)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 8. 可信度图
    ax = axes[7]
    rel = uncertainty_result['reliability'][batch_idx, 0].cpu().numpy()
    im = ax.imshow(rel, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Reliability Map (Green=High Trust)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # GT 对比
    if target_hr is not None:
        mae = torch.abs(uncertainty_result['sr_mean'] - target_hr).mean().item()
        fig.suptitle(f'MC-Dropout Uncertainty Analysis | MAE={mae:.3f}', fontsize=16)
    else:
        fig.suptitle('MC-Dropout Uncertainty Analysis', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# ==================== 4. 使用示例 ====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, H_lr = 2, 32
    H_hr = H_lr * 8  # 256

    # 模拟数据
    lr_precip = torch.rand(B, 1, H_lr, H_lr).to(device) * 5.0
    wind_hr = torch.randn(B, 2, H_hr, H_hr).to(device) * 3.0
    target_hr = torch.rand(B, 1, H_hr, H_hr).to(device) * 10.0

    # 初始化模型 (假设 WindGuidedRRDBNet_8x 已定义)
    # model = WindGuidedRRDBNet_8x(...).to(device)

    # 创建不确定性估计器
    # estimator = WindGuidedUncertaintyEstimator(model, num_samples=30)

    # 推理
    # result = estimator.estimate(lr_precip, wind_hr)

    # 打印统计信息
    # print(f"预测均值范围: [{result['sr_mean'].min():.2f}, {result['sr_mean'].max():.2f}]")
    # print(f"注意力方差范围: [{result['attn_var'].min():.6f}, {result['attn_var'].max():.6f}]")
    # print(f"平均可信度: {result['reliability'].mean():.3f}")

    # 可视化
    # visualize_uncertainty(result, lr_precip, wind_hr, target_hr, save_path='uncertainty.png')