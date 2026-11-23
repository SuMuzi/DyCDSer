import torch
import torch.nn as nn
import torch.nn.functional as F


class DivergenceLayer(nn.Module):
    def __init__(self, grid_spacing=1.0, padding_mode='reflect'):
        """
        Compute divergence of wind field

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

    def forward(self, u, v):
        """
        Compute divergence of wind field

        Args:
            u: (B, 1, H, W) - Zonal wind component
            v: (B, 1, H, W) - Meridional wind component

        Returns:
            divergence: (B, 1, H, W) - Divergence field
                       div > 0: Divergence (air spreading out)
                       div < 0: Convergence (air coming together)
        """
        # Compute du/dx (x-direction derivative)
        du_dx = F.conv2d(u, self.dx_kernel, padding=(0, 1), padding_mode=self.padding_mode) / self.grid_spacing

        # Compute dv/dy (y-direction derivative)
        dv_dy = F.conv2d(v, self.dy_kernel, padding=(1, 0), padding_mode=self.padding_mode) / self.grid_spacing

        # Divergence = du/dx + dv/dy
        div = du_dx + dv_dy

        return div


# --- 使用示例 ---
# 假设输入风场 (B, 1, H, W)
B, H, W = 2, 32, 32
u_wind = torch.randn(B, 1, H, W)
v_wind = torch.randn(B, 1, H, W)

div_layer = DivergenceLayer()
divergence = div_layer(u_wind, v_wind)

print(f"输入风场形状: {u_wind.shape}")
print(f"计算出的散度形状: {divergence.shape}")