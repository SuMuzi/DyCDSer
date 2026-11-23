import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from utils.utils import compute_divergence


def visualize_wind_precip_coupling(pred, target, u, v, save_path, case_name=''):
    """
    综合可视化：风场、降水、误差的空间关系
    
    Args:
        pred: 预测降水
        target: 观测降水
        u: 风场U分量
        v: 风场V分量
        save_path: 保存路径
        case_name: 案例名称（用于标题）
    """
    # 计算特征
    wind_speed = np.sqrt(u ** 2 + v ** 2)
    divergence = compute_divergence(u, v)
    error = pred - target
    rel_error = np.abs(error) / (target + 1e-6)

    fig = plt.figure(figsize=(20, 16))

    # 1. 降水对比
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(target, cmap='Blues', vmin=0, vmax=20)
    ax1.set_title('Target Precipitation (mm/h)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(pred, cmap='Blues', vmin=0, vmax=20)
    ax2.set_title('Predicted Precipitation (mm/h)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(np.abs(error), cmap='Reds', vmin=0, vmax=10)
    ax3.set_title('Absolute Error (mm/h)')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 2. 风场特征
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(wind_speed, cmap='YlOrRd')
    ax4.set_title('Wind Speed (m/s)')
    # 叠加风矢量（稀疏采样）
    y, x = np.mgrid[0:u.shape[0]:5, 0:u.shape[1]:5]
    ax4.quiver(x, y, u[::5, ::5], v[::5, ::5], scale=50, color='black', alpha=0.7)
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(divergence, cmap='RdBu_r', vmin=-5e-4, vmax=5e-4)
    ax5.set_title('Divergence (1/s)')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # 3. 条件误差分析（散度分箱）
    ax6 = plt.subplot(3, 3, 6)
    div_bins = np.linspace(divergence.min(), divergence.max(), 20)
    bin_errors = []
    for i in range(len(div_bins) - 1):
        mask = (divergence >= div_bins[i]) & (divergence < div_bins[i + 1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(np.abs(error[mask])))
        else:
            bin_errors.append(np.nan)

    ax6.bar(range(len(bin_errors)), bin_errors, color='steelblue', alpha=0.7)
    ax6.set_xlabel('Divergence Bin (left=convergence, right=divergence)')
    ax6.set_ylabel('Mean Absolute Error (mm/h)')
    ax6.set_title('Error by Divergence Regime')

    # 4. 强风区叠加
    ax7 = plt.subplot(3, 3, 7)
    strong_wind = wind_speed > np.percentile(wind_speed, 90)
    im7 = ax7.imshow(target, cmap='Blues', alpha=0.6)
    ax7.contour(strong_wind, colors='red', linewidths=2)
    ax7.set_title('Observed Precip + Strong Wind Regions')

    ax8 = plt.subplot(3, 3, 8)
    im8 = ax8.imshow(pred, cmap='Blues', alpha=0.6)
    ax8.contour(strong_wind, colors='red', linewidths=2)
    ax8.set_title('Predicted Precip + Strong Wind Regions')

    # 5. 误差-风场散点图
    ax9 = plt.subplot(3, 3, 9)
    total_samples = len(wind_speed.flatten())
    sample_size = min(5000, total_samples)
    sample_idx = np.random.choice(total_samples, sample_size, replace=False)
    ax9.scatter(wind_speed.flatten()[sample_idx],
                np.abs(error).flatten()[sample_idx],
                alpha=0.3, s=1, c=divergence.flatten()[sample_idx],
                cmap='RdBu_r', vmin=-1e-4, vmax=1e-4)
    ax9.set_xlabel('Wind Speed (m/s)')
    ax9.set_ylabel('Absolute Error (mm/h)')
    ax9.set_title('Error vs Wind Speed (color=Divergence)')

    plt.suptitle(f'Wind-Precipitation Coupling Analysis: {case_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
