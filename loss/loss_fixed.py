import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PhysicsConstrainedLoss_fixed(nn.Module):
    """
    物理约束损失函数（修正版）

    主要修正点：
    1. physics_consistency 改为非负损失，避免负值抵消 MSE。
    2. 补全 mass_conservation 项（lambda_mass 原来定义但未使用）。
    3. offset_smooth 保持非负。
    4. 各项增加 eps，防止除零。
    """

    def __init__(
            self,
            lambda_mse: float = 1.0,            # MSE 权重
            lambda_mass: float = 0.1,           # 质量守恒约束
            lambda_continuity: float = 0.2,     # 连续性/平滑约束
            lambda_budget: float = 0.3,         # 水汽收支/物理一致性约束
            eps: float = 1e-6
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_mass = lambda_mass
        self.lambda_continuity = lambda_continuity
        self.lambda_budget = lambda_budget
        self.eps = eps

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            wind_model: torch.Tensor,
            wind_obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # 1. 标准 MSE 损失
        losses['precip_mse'] = F.mse_loss(
            outputs['precipitation'], targets['precipitation']
        ) * self.lambda_mse

        # 2. 质量守恒：调整后降水总量应与目标降水总量接近
        if 'precip_adjusted' in outputs:
            pred_mean = outputs['precip_adjusted'].mean()
            target_mean = targets['precipitation'].mean()
            losses['mass_conservation'] = torch.abs(pred_mean - target_mean) * self.lambda_mass

        # 3. 物理一致性：降水应与观测散度呈负相关（辐合→降水）
        if 'precip_adjusted' in outputs and 'adjustments' in outputs:
            u_mod, v_mod = wind_model[:, 0:1], wind_model[:, 1:2]
            u_obs, v_obs = wind_obs[:, 0:1], wind_obs[:, 1:2]

            div_obs = self.compute_divergence(u_obs, v_obs)
            precip_pred = outputs['precip_adjusted']

            # 标准化后比较形态
            precip_norm = (precip_pred - precip_pred.mean()) / (precip_pred.std() + self.eps)
            div_obs_norm = (div_obs - div_obs.mean()) / (div_obs.std() + self.eps)

            correlation = (precip_norm * div_obs_norm).mean()

            # 修正：只惩罚正相关（辐散区降水偏多，违反物理）
            # 负相关或零相关时损失为 0
            losses['physics_consistency'] = F.relu(correlation) * self.lambda_budget

            # 4. 连续性约束：调整量应平滑
            adj = outputs['adjustments']
            if 'position_offset' in adj:
                offset_smooth = self.smoothness_loss(adj['position_offset'])
                losses['offset_smooth'] = offset_smooth * self.lambda_continuity

        losses['total'] = sum(losses.values())
        return losses

    def compute_divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算散度"""
        du_dx = u[:, :, :, 2:] - u[:, :, :, :-2]
        dv_dy = v[:, :, 2:, :] - v[:, :, :-2, :]
        du_dx = F.pad(du_dx, (1, 1, 0, 0), mode='replicate')
        dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode='replicate')
        return du_dx + dv_dy

    def smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """空间平滑性损失"""
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return (grad_x.mean() + grad_y.mean()) / 2


if __name__ == "__main__":
    # 简单验证
    B, H, W = 2, 64, 64
    criterion = PhysicsConstrainedLoss_fixed()

    outputs = {
        'precipitation': torch.rand(B, 1, H, W),
        'precip_adjusted': torch.rand(B, 1, H, W),
        'adjustments': {
            'position_offset': torch.rand(B, 2, H, W) * 0.1
        }
    }
    targets = {
        'precipitation': torch.rand(B, 1, H, W)
    }
    wind_model = torch.rand(B, 2, H, W)
    wind_obs = torch.rand(B, 2, H, W)

    losses = criterion(outputs, targets, wind_model, wind_obs)
    print("Loss breakdown:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
