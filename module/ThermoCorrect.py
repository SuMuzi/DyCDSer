import torch
import torch.nn as nn
import torch.nn.functional as F
from .DivergenceLayer import DivergenceLayer


class PhysicsConstrainedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'):
        """
        Physics-constrained convolution block

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            padding_mode: Padding mode
        """
        super().__init__()

        # 主卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # 物理约束参数
        self.physics_weight = nn.Parameter(torch.tensor(0.1))

        # 温度梯度计算层
        self.gradient_layer = TemperatureGradientLayer()

    def forward(self, x, temperature=None):
        """
        Forward pass with physics constraints

        Args:
            x: Input feature map
            temperature: Temperature field (optional)

        Returns:
            Output feature map with physics constraints
        """
        # 常规卷积操作
        out = self.conv(x)
        out = self.bn(out)

        # 应用物理约束（如果提供温度数据）
        if temperature is not None:
            # 计算温度梯度
            _, grad_magnitude = self.gradient_layer(temperature)

            # 调整梯度大小以匹配特征图
            grad_magnitude = F.interpolate(grad_magnitude, size=out.shape[2:], mode='bilinear')

            # 将温度梯度信息融入特征
            out = out + self.physics_weight * grad_magnitude

        # 激活函数
        out = self.act(out)

        return out


class WindConstrainedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'):
        """
        Wind-constrained convolution block

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            padding_mode: Padding mode
        """
        super().__init__()

        # 主卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # 物理约束参数
        self.physics_weight = nn.Parameter(torch.tensor(0.1))

        # 散度计算层
        self.divergence_layer = DivergenceLayer()

    def forward(self, x, u_wind=None, v_wind=None):
        """
        Forward pass with wind constraints

        Args:
            x: Input feature map
            u_wind: Zonal wind component (optional)
            v_wind: Meridional wind component (optional)

        Returns:
            Output feature map with wind constraints
        """
        # 常规卷积操作
        out = self.conv(x)
        out = self.bn(out)

        # 应用风场约束（如果提供风场数据）
        if u_wind is not None and v_wind is not None:
            # 计算散度
            divergence = self.divergence_layer(u_wind, v_wind)

            # 调整散度大小以匹配特征图
            divergence = F.interpolate(divergence, size=out.shape[2:], mode='bilinear')

            # 将散度信息融入特征（负散度对应辐合，有利于降水）
            out = out - self.physics_weight * divergence

        # 激活函数
        out = self.act(out)

        return out


class PhysicsInformedNetwork(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, c_list=[32, 64, 128]):
        """
        Physics-informed network with explicit physics constraints

        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            c_list: List of channel sizes
        """
        super().__init__()

        # 输入投影
        self.input_proj = nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1, padding_mode='reflect')

        # 物理约束块
        self.blocks = nn.ModuleList()
        for i in range(len(c_list)):
            in_ch = c_list[i - 1] if i > 0 else c_list[0]
            out_ch = c_list[i]

            # 交替使用温度和风场约束块
            if i % 2 == 0:
                self.blocks.append(PhysicsConstrainedBlock(in_ch, out_ch))
            else:
                self.blocks.append(WindConstrainedBlock(in_ch, out_ch))

        # 输出投影
        self.output_proj = nn.Conv2d(c_list[-1], output_channels, kernel_size=3, padding=1, padding_mode='reflect')

        # 降水非负性约束
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, temperature=None, u_wind=None, v_wind=None):
        """
        Forward pass with physics constraints

        Args:
            x: Input feature map
            temperature: Temperature field (optional)
            u_wind: Zonal wind component (optional)
            v_wind: Meridional wind component (optional)

        Returns:
            Output precipitation prediction
        """
        # 输入投影
        x = self.input_proj(x)

        # 处理通过物理约束块
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x, temperature)
            else:
                x = block(x, u_wind, v_wind)

        # 输出投影
        x = self.output_proj(x)

        # 应用非负性约束
        x = self.relu(x)

        return x


class TemperatureGradientLayer(nn.Module):
    def __init__(self, grid_spacing=1.0, padding_mode='reflect'):
        """
        Compute temperature gradient

        Args:
            grid_spacing: Grid spacing (in meters) for physical dimension calibration.
                         Set to 1.0 for feature extraction only.
            padding_mode: Boundary padding mode, 'reflect', 'replicate', or 'constant'
        """
        super().__init__()
        self.grid_spacing = grid_spacing
        self.padding_mode = padding_mode

        # Define convolution kernels for partial derivatives (central difference)
        # d/dx: [-0.5, 0, 0.5] (1×3 kernel)
        # d/dy: [-0.5, 0, 0.5]^T (3×1 kernel)

        # Kernel for x-direction derivative (horizontal difference)
        self.register_buffer("dx_kernel", torch.tensor([[[[-0.5, 0, 0.5]]]], dtype=torch.float32))
        # Kernel for y-direction derivative (vertical difference)
        self.register_buffer("dy_kernel", torch.tensor([[[[-0.5], [0], [0.5]]]], dtype=torch.float32))

    def forward(self, temperature):
        """
        Compute temperature gradient

        Args:
            temperature: (B, 1, H, W) - Temperature field

        Returns:
            gradient: (B, 2, H, W) - Temperature gradient field (dx, dy)
            gradient_magnitude: (B, 1, H, W) - Gradient magnitude
        """
        # Compute dT/dx (x-direction derivative)
        dT_dx = F.conv2d(temperature, self.dx_kernel, padding=(0, 1),padding_mode=self.padding_mode) / self.grid_spacing

        # Compute dT/dy (y-direction derivative)
        dT_dy = F.conv2d(temperature, self.dy_kernel, padding=(1, 0), padding_mode=self.padding_mode) / self.grid_spacing

        # Combine gradients
        gradient = torch.cat([dT_dx, dT_dy], dim=1)

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(dT_dx ** 2 + dT_dy ** 2)

        return gradient, gradient_magnitude


# --- 使用示例 ---#
if __name__ == "__main__":
    # 测试物理约束网络
    B, C, H, W = 2, 1, 32, 32

    # 输入数据
    input_data = torch.randn(B, C, H, W)
    temperature = torch.randn(B, 1, H, W)
    u_wind = torch.randn(B, 1, H, W)
    v_wind = torch.randn(B, 1, H, W)

    # 初始化模型
    model = PhysicsInformedNetwork()

    # 前向传播
    output = model(input_data, temperature, u_wind, v_wind)

    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出最小值: {output.min().item()}")  # 应该非负
    print("物理约束网络测试成功！")