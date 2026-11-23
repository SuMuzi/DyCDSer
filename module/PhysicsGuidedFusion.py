import torch
import torch.nn as nn
import torch.nn.functional as F
class DivergenceCalculator(nn.Module):
    """
    用 nn.Conv2d 实现散度/梯度计算
    卷积核为不可训练的 Sobel 风格差分核
    """
    def __init__(self):
        super().__init__()

        # du/dx 核: 中心差分, 水平方向
        # [[0, 0, 0],
        #  [-0.5, 0, 0.5],
        #  [0, 0, 0]]
        kernel_dx = torch.tensor([
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0]
        ]).view(1, 1, 3, 3)

        # dv/dy 核: 中心差分, 垂直方向
        # [[0, 0.5, 0],
        #  [0, 0, 0],
        #  [0, -0.5, 0]]
        kernel_dy = torch.tensor([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0]
        ]).view(1, 1, 3, 3)

        # 注册为不可训练参数
        self.register_buffer('kernel_dx', kernel_dx)
        self.register_buffer('kernel_dy', kernel_dy)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, 1, H, W] 或 [B, C, H, W]
            v: [B, 1, H, W] 或 [B, C, H, W]
        Returns:
            div: [B, C, H, W] 散度场
        """
        B, C, H, W = u.shape

        # 扩展核到多通道 (group=C，每个通道独立卷积)
        # 或者直接对每组通道分别计算
        if C > 1:
            # 方式1: 对每个通道分别计算（推荐，逻辑清晰）
            div_list = []
            for c in range(C):
                du_dx = F.conv2d(
                    u[:, c:c+1], self.kernel_dx,
                    padding=1, stride=1
                )
                dv_dy = F.conv2d(
                    v[:, c:c+1], self.kernel_dy,
                    padding=1, stride=1
                )
                div_list.append(du_dx + dv_dy)
            return torch.cat(div_list, dim=1)
        else:
            # 单通道直接计算
            du_dx = F.conv2d(u, self.kernel_dx, padding=1, stride=1)
            dv_dy = F.conv2d(v, self.kernel_dy, padding=1, stride=1)
            return du_dx + dv_dy


# ==================== 更高效的 group conv 版本 ====================

class DivergenceCalculatorGroup(nn.Module):
    """
    使用 group convolution 高效处理多通道
    所有通道共享同一组差分核
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels

        # 为每个通道复制相同的核
        kernel_dx = torch.tensor([
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0]
        ]).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        kernel_dx = kernel_dx.repeat(in_channels, 1, 1, 1)  # [C, 1, 3, 3]

        kernel_dy = torch.tensor([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_dy = kernel_dy.repeat(in_channels, 1, 1, 1)

        # 作为 buffer 注册（不训练，随模型保存）
        self.register_buffer('weight_dx', kernel_dx)
        self.register_buffer('weight_dy', kernel_dy)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, C, H, W]
            v: [B, C, H, W]
        Returns:
            div: [B, C, H, W]
        """
        # group=C: 每个通道用自己的核独立计算
        du_dx = F.conv2d(u, self.weight_dx, padding=1, groups=self.in_channels)
        dv_dy = F.conv2d(v, self.weight_dy, padding=1, groups=self.in_channels)
        return du_dx + dv_dy
class PhysicsGuidedFusion(nn.Module):
    """
    基于大气物理的显式融合
    整合环流、湿度、不稳定度、垂直运动生成降水潜势
    """

    def __init__(self, dim=256):
        super().__init__()

        # 物理量解码头
        self.q_decoder = nn.Sequential(  # 整层可降水量
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()  # 归一化湿度
        )

        self.omega_decoder = nn.Sequential(  # 垂直速度（诊断）
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Tanh()  # -1(下沉) ~ 1(上升)
        )

        self.cape_decoder = nn.Sequential(  # 对流有效位能
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.ReLU()  # >= 0
        )

        self.lifted_index_decoder = nn.Sequential(  # 抬升指数（稳定性）
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1)
            # 负值表示不稳定
        )

        # 降水生成网络（物理参数化风格）
        self.precip_generator = ConvectiveParametrizationModule()

    def forward(self, multi_view_features):
        """
        显式解码物理量，物理一致地生成降水
        """
        # 合并多视图特征
        combined = torch.cat([
            multi_view_features['circulation'],
            multi_view_features['lower_troposphere'],
            multi_view_features['boundary_layer'],
            multi_view_features['vertical_profile']
        ], dim=1)  # [B, 256, H, W]

        # 解码关键物理量
        Q = self.q_decoder(combined)  # 可降水量 [0,1]
        omega = self.omega_decoder(combined)  # 垂直速度 [-1,1]
        CAPE = self.cape_decoder(combined)  # 对流能量 [0,∞)
        LI = self.lifted_index_decoder(combined)  # 稳定性 (-∞,∞)

        # 物理一致的降水生成
        precipitation = self.precip_generator(Q, omega, CAPE, LI)

        return precipitation, {
            'Q': Q, 'omega': omega, 'CAPE': CAPE, 'LI': LI
        }


class ConvectiveParametrizationModule(nn.Module):
    """
    模拟对流参数化方案的神经网络版本
    降水 = f(湿度, 上升运动, 不稳定度, 大尺度强迫)
    """

    def __init__(self):
        super().__init__()

        # 对流触发函数（类似Kuo方案）
        self.trigger = nn.Sequential(
            nn.Linear(4, 64),  # Q, omega, CAPE, LI
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 对流概率
        )

        # 降水效率（给定对流强度下的降水转化率）
        self.efficiency = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, Q, omega, CAPE, LI):
        B, C, H, W = Q.shape

        # 堆叠物理量
        physics = torch.cat([Q, omega, CAPE, LI], dim=1)  # [B, 4, H, W]
        physics_flat = physics.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]

        # 对流触发概率
        conv_prob = self.trigger(physics_flat).view(B, 1, H, W)

        # 降水效率
        efficiency = self.efficiency(physics_flat).view(B, 1, H, W)

        # 物理降水公式：
        # 降水 = 触发概率 × 效率 × 可降水量 × 上升运动强度
        upward_motion = F.relu(-omega)  # 只考虑上升(omega<0)

        precipitation = (
                conv_prob *  # 是否对流 [0,1]
                efficiency *  # 转化效率 [0,1]
                Q * 100 *  # 可降水量 (归一化后反归一化，单位mm)
                upward_motion * 10  # 上升运动强度
        )

        # 大尺度强迫调制（从环流视图）
        large_scale_forcing = 1.0  # 可添加天气尺度强迫项

        return precipitation * large_scale_forcing