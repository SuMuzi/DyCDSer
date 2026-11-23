import matplotlib.pyplot as plt
import numpy as np

def compute_divergence(u, v):
    """计算水平散度"""
    dx = np.gradient(u.longitude) * 111320  # m
    dy = np.gradient(u.latitude) * 110540  # m

    du_dx = np.gradient(u, axis=-1) / dx[np.newaxis, np.newaxis, :]
    dv_dy = np.gradient(v, axis=-2) / dy[np.newaxis, :, np.newaxis]

    return du_dx + dv_dy

def visualize_wind_guidance(pred, target, u, v, save_path):
    """
    可视化风场引导效果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 降水对比
    axes[0, 0].imshow(target[0, 0].cpu(), cmap='Blues')
    axes[0, 0].set_title('HR Target')

    axes[0, 1].imshow(pred[0, 0].detach().cpu(), cmap='Blues')
    axes[0, 1].set_title('Prediction')

    # 风场叠加
    x = np.arange(u.shape[-1])
    y = np.arange(u.shape[-2])
    X, Y = np.meshgrid(x, y)

    axes[0, 2].imshow(pred[0, 0].detach().cpu(), cmap='Blues', alpha=0.6)
    axes[0, 2].quiver(X[::5, ::5], Y[::5, ::5],
                      u[0, 0, ::5, ::5].cpu(),
                      v[0, 0, ::5, ::5].cpu(),
                      scale=50)
    axes[0, 2].set_title('Pred + Wind Overlay')

    # 误差分析
    error = (pred - target)[0, 0].detach().cpu()
    im = axes[1, 0].imshow(error, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, 0].set_title('Error')
    plt.colorbar(im, ax=axes[1, 0])

    # 散度场
    div = compute_divergence(u[0, 0], v[0, 0])
    axes[1, 1].imshow(div.cpu(), cmap='RdBu_r')
    axes[1, 1].set_title('Wind Divergence')

    # 误差 vs 散度散点（检验物理一致性）
    axes[1, 2].scatter(div.cpu().flatten(), error.abs().cpu().flatten(), alpha=0.1)
    axes[1, 2].set_xlabel('Divergence')
    axes[1, 2].set_ylabel('|Error|')
    axes[1, 2].set_title('Error vs Divergence')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)