import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数：确保风速校正后的降水满足动力一致性
    """

    def __init__(
            self,
            lambda_mass: float = 0.1,  # 质量守恒约束
            lambda_continuity: float = 0.2,  # 连续方程约束
            lambda_budget: float = 0.3  # 水汽收支约束
    ):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_continuity = lambda_continuity
        self.lambda_budget = lambda_budget

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            wind_model: torch.Tensor,
            wind_obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # 1. 标准MSE损失
        losses['precip_mse'] = F.mse_loss(outputs['precipitation'], targets['precipitation'])

        # 2. 质量守恒约束：调整前后的降水总量应与模式风场散度匹配
        if 'precip_adjusted' in outputs and 'adjustments' in outputs:
            # 计算调整前后的散度变化
            u_mod, v_mod = wind_model[:, 0:1], wind_model[:, 1:2]
            u_obs, v_obs = wind_obs[:, 0:1], wind_obs[:, 1:2]

            # 简化：假设降水与低层散度成正比（深对流参数化思路）
            # 模式散度
            div_mod = self.compute_divergence(u_mod, v_mod)
            # 观测散度（真值）
            div_obs = self.compute_divergence(u_obs, v_obs)

            # 调整后的降水应与观测散度匹配
            # 线性关系假设：P ≈ -α * div（辐合→降水）
            precip_pred = outputs['precip_adjusted']
            # 归一化后比较形态（避免强度量级差异）
            precip_norm = (precip_pred - precip_pred.mean()) / (precip_pred.std() + 1e-6)
            div_obs_norm = (div_obs - div_obs.mean()) / (div_obs.std() + 1e-6)

            # 负相关约束（辐合对应降水）
            correlation = (precip_norm * div_obs_norm).mean()
            losses['physics_consistency'] = -correlation * self.lambda_budget  # 最大化负相关（辐合→降水）

            # 3. 连续性约束：调整量应平滑（避免非物理的跳跃）
            adj = outputs['adjustments']
            if 'position_offset' in adj:
                offset_smooth = self.smoothness_loss(adj['position_offset'])
                losses['offset_smooth'] = offset_smooth * self.lambda_continuity

        losses['total'] = sum(losses.values())
        return losses

    def compute_divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算散度"""
        # 使用有限差分
        du_dx = u[:, :, :, 2:] - u[:, :, :, :-2]
        dv_dy = v[:, :, 2:, :] - v[:, :, :-2, :]
        # 填充到原尺寸
        du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')
        dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')
        return du_dx + dv_dy

    def smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算空间平滑性（梯度的L1范数）"""
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return (grad_x.mean() + grad_y.mean()) / 2


class WeightedMAEForExtreme(nn.Module):
    """
    面向极端降水的加权 MAE 损失函数。

    核心思想：让大雨/强降水样本在损失中占更大权重，避免模型为了压低整体 MAE
    而牺牲稀有但重要的极端降水事件。

    假设输入 pred / target 都是物理空间降水（单位：mm/h），非负。
    """

    def __init__(self, mode='hybrid', alpha=5.0, beta=0.5, threshold=10.0, eps=1e-6, normalize_weight=True):
        """
        Args:
            mode: 权重计算方式
                - 'log'      : w = log(1 + target) + 1，随雨强单调增加
                - 'power'    : w = (target + eps)^beta，可精细控制尾部权重
                - 'threshold': w = alpha 当 target >= threshold，否则 1
                - 'hybrid'   : log 权重 + threshold 额外加权（推荐）
            alpha: threshold/hybrid 模式下极端降水的额外权重倍数
            beta:  power 模式下控制尾部压缩/展开程度
            threshold: 极端降水阈值（mm/h），默认 10 mm/h
            eps: 数值稳定性
            normalize_weight: 是否把权重归一化到均值为 1，避免整体 loss 量级剧烈变化
        """
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.eps = eps
        self.normalize_weight = normalize_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 保证非负，避免异常输入
        target = torch.clamp(target, min=0.0)
        pred = torch.clamp(pred, min=0.0)

        error = torch.abs(pred - target)

        if self.mode == 'log':
            weight = torch.log1p(target) + 1.0

        elif self.mode == 'power':
            weight = torch.pow(target + self.eps, self.beta)

        elif self.mode == 'threshold':
            weight = torch.where(
                target >= self.threshold,
                torch.full_like(target, self.alpha),
                torch.full_like(target, 1.0)
            )

        elif self.mode == 'hybrid':
            log_w = torch.log1p(target) + 1.0
            extreme_w = torch.where(
                target >= self.threshold,
                torch.full_like(target, self.alpha),
                torch.full_like(target, 1.0)
            )
            weight = log_w * extreme_w

        else:
            weight = torch.ones_like(target)

        if self.normalize_weight:
            weight = weight / (weight.mean() + self.eps)

        return (weight * error).mean()
