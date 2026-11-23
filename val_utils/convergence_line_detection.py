from skimage.morphology import skeletonize
import numpy as np
from skimage.measure import label, regionprops
from utils.utils import compute_divergence
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cfeat


def convergence_line_detection(u, v, threshold=1e-4):
    """
    检测辐合线（类似锋面/飑线）
    
    Args:
        u: 风场U分量
        v: 风场V分量
        threshold: 辐合阈值（散度 < -threshold 为强辐合）
    
    Returns:
        lines: 检测到的辐合线列表，每条线是坐标点列表
        convergence_mask: 辐合区域掩码
    """
    divergence = compute_divergence(u, v)
    
    # 强辐合区域（散度 < -threshold）
    convergence_mask = divergence < -threshold
    
    # 骨架提取（需要转换为uint8）
    skeleton = skeletonize(convergence_mask.astype(np.uint8))
    
    # 提取线段
    lines = extract_lines_from_skeleton(skeleton)
    
    return lines, convergence_mask


def extract_lines_from_skeleton(skeleton):
    """
    从骨架图中提取线段
    
    Args:
        skeleton: 骨架化后的二值图像
    
    Returns:
        lines: 线段列表，每条线包含坐标点列表
    """
    lines = []
    
    # 标记连通区域
    labeled = label(skeleton)
    props = regionprops(labeled)
    
    for prop in props:
        # 获取区域的所有像素坐标
        coords = prop.coords.tolist()
        if len(coords) >= 2:  # 至少2个点才构成线段
            lines.append(coords)
    
    return lines


def evaluate_precip_along_convergence(pred, target, u, v, lines, buffer=3):
    """
    评估辐合线附近降水精度
    
    Args:
        pred: 预测降水
        target: 观测降水
        u: 风场U分量
        v: 风场V分量
        lines: 检测到的辐合线
        buffer: 缓冲区宽度
    
    Returns:
        results: 包含沿辐合线和远离辐合线的评估指标
    """
    # 创建辐合线缓冲区
    line_mask = np.zeros_like(pred, dtype=bool)
    for line in lines:
        for y, x in line:
            y1, y2 = max(0, y-buffer), min(pred.shape[0], y+buffer+1)
            x1, x2 = max(0, x-buffer), min(pred.shape[1], x+buffer+1)
            line_mask[y1:y2, x1:x2] = True
    
    # 评估
    results = {
        'along_convergence': {
            'rmse': np.sqrt(np.mean((pred[line_mask] - target[line_mask])**2)),
            'mae': np.mean(np.abs(pred[line_mask] - target[line_mask])),
            'sample_size': line_mask.sum()
        },
        'away_convergence': {
            'rmse': np.sqrt(np.mean((pred[~line_mask] - target[~line_mask])**2)),
            'mae': np.mean(np.abs(pred[~line_mask] - target[~line_mask])),
            'sample_size': (~line_mask).sum()
        }
    }
    
    # 提升度
    results['improvement_ratio'] = (
        results['away_convergence']['rmse'] / results['along_convergence']['rmse']
    )
    
    return results


def plot_convergence_and_precip(pred, target, u, v, lines, convergence_mask, save_path=None):
    """
    可视化辐合线和降水对比
    
    Args:
        pred: 预测降水
        target: 观测降水
        u: 风场U分量
        v: 风场V分量
        lines: 检测到的辐合线
        convergence_mask: 辐合区域掩码
        save_path: 保存路径（可选）
    """
    divergence = compute_divergence(u, v)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 散度场 + 辐合线
    im1 = axes[0, 0].imshow(divergence, cmap='RdBu_r', 
                            norm=colors.Normalize(vmin=-0.001, vmax=0.001))
    axes[0, 0].contour(convergence_mask, colors='green', levels=[0.5], linewidths=2)
    for line in lines:
        coords = np.array(line)
        axes[0, 0].plot(coords[:, 1], coords[:, 0], 'w-', linewidth=2)
    axes[0, 0].set_title('Divergence + Convergence Lines')
    plt.colorbar(im1, ax=axes[0, 0], label='Divergence')
    
    # 2. 观测降水 + 辐合线
    im2 = axes[0, 1].imshow(target, cmap='Blues', vmin=0, vmax=50)
    for line in lines:
        coords = np.array(line)
        axes[0, 1].plot(coords[:, 1], coords[:, 0], 'r-', linewidth=2)
    axes[0, 1].set_title('Observed Precipitation + Convergence Lines')
    plt.colorbar(im2, ax=axes[0, 1], label='Precipitation (mm/h)')
    
    # 3. 预测降水 + 辐合线
    im3 = axes[1, 0].imshow(pred, cmap='Blues', vmin=0, vmax=50)
    for line in lines:
        coords = np.array(line)
        axes[1, 0].plot(coords[:, 1], coords[:, 0], 'r-', linewidth=2)
    axes[1, 0].set_title('Predicted Precipitation + Convergence Lines')
    plt.colorbar(im3, ax=axes[1, 0], label='Precipitation (mm/h)')
    
    # 4. 降水误差（预测-观测）
    error = pred - target
    im4 = axes[1, 1].imshow(error, cmap='RdBu_r', 
                            norm=colors.Normalize(vmin=-20, vmax=20))
    for line in lines:
        coords = np.array(line)
        axes[1, 1].plot(coords[:, 1], coords[:, 0], 'w-', linewidth=2)
    axes[1, 1].set_title('Prediction Error (Pred - Obs) + Convergence Lines')
    plt.colorbar(im4, ax=axes[1, 1], label='Error (mm/h)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_convergence_with_coastline(pred, target, u, v, lines, convergence_mask,
                                   lon_min=116.371, lon_max=125.568,
                                   lat_min=19.5483, lat_max=27.8446,
                                   save_path=None):
    """
    可视化辐合线和降水对比（带海岸线）
    
    Args:
        pred: 预测降水
        target: 观测降水
        u: 风场U分量
        v: 风场V分量
        lines: 检测到的辐合线
        convergence_mask: 辐合区域掩码
        lon_min, lon_max: 经度范围
        lat_min, lat_max: 纬度范围
        save_path: 保存路径（可选）
    """
    divergence = compute_divergence(u, v)
    
    # 创建经纬度网格
    lon = np.linspace(lon_min, lon_max, pred.shape[1])
    lat = np.linspace(lat_min, lat_max, pred.shape[0])
    Lon, Lat = np.meshgrid(lon, lat)
    
    # 自定义颜色映射
    bounds_in = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    turbo_camp = get_cmap('turbo')
    colors_list = ['white']
    for i in range(10):
        if i > 0:
            colors_list.append(turbo_camp(i / 9))
    custom_cmap = ListedColormap(colors_list)
    norm_in = BoundaryNorm(bounds_in, custom_cmap.N)
    
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 12), dpi=150)
    
    # 1. 观测降水 + 辐合线 + 海岸线
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)
    im1 = ax1.pcolormesh(Lon, Lat, target, cmap=custom_cmap, shading='auto',
                         transform=proj, norm=norm_in)
    ax1.contour(Lon, Lat, convergence_mask, colors='green', levels=[0.5], 
                linewidths=2, transform=proj)
    for line in lines:
        coords = np.array(line)
        ax1.plot(lon[coords[:, 1]], lat[coords[:, 0]], 'r-', linewidth=2, transform=proj)
    ax1.coastlines(resolution='50m', color='k', linewidth=0.6)
    ax1.add_feature(cfeat.BORDERS, linewidth=0.4)
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax1.set_title('Observed Precipitation')
    cbar1 = fig.colorbar(im1, ax=ax1, label='', ticks=bounds_in[:-1])
    cbar1.set_ticklabels(['0', '0.1', '1', '2.5', '5', '10', '20', '30', '40', '50'])
    
    # 2. 预测降水 + 辐合线 + 海岸线
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    im2 = ax2.pcolormesh(Lon, Lat, pred, cmap=custom_cmap, shading='auto',
                         transform=proj, norm=norm_in)
    ax2.contour(Lon, Lat, convergence_mask, colors='green', levels=[0.5],
                linewidths=2, transform=proj)
    for line in lines:
        coords = np.array(line)
        ax2.plot(lon[coords[:, 1]], lat[coords[:, 0]], 'r-', linewidth=2, transform=proj)
    ax2.coastlines(resolution='50m', color='k', linewidth=0.6)
    ax2.add_feature(cfeat.BORDERS, linewidth=0.4)
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax2.set_title('Predicted Precipitation')
    cbar2 = fig.colorbar(im2, ax=ax2, label='', ticks=bounds_in[:-1])
    cbar2.set_ticklabels(['0', '0.1', '1', '2.5', '5', '10', '20', '30', '40', '50'])
    
    # 3. 散度场 + 辐合线 + 海岸线
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    im3 = ax3.pcolormesh(Lon, Lat, divergence, cmap='RdBu_r', shading='auto',
                         transform=proj, norm=colors.Normalize(vmin=-0.001, vmax=0.001))
    ax3.contour(Lon, Lat, convergence_mask, colors='green', levels=[0.5],
                linewidths=2, transform=proj)
    for line in lines:
        coords = np.array(line)
        ax3.plot(lon[coords[:, 1]], lat[coords[:, 0]], 'w-', linewidth=2, transform=proj)
    ax3.coastlines(resolution='50m', color='k', linewidth=0.6)
    ax3.add_feature(cfeat.BORDERS, linewidth=0.4)
    ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax3.set_title('Divergence')
    fig.colorbar(im3, ax=ax3, label='Divergence')
    
    # 4. 误差场 + 辐合线 + 海岸线
    error = pred - target
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)
    im4 = ax4.pcolormesh(Lon, Lat, error, cmap='RdBu_r', shading='auto',
                         transform=proj, norm=colors.Normalize(vmin=-20, vmax=20))
    for line in lines:
        coords = np.array(line)
        ax4.plot(lon[coords[:, 1]], lat[coords[:, 0]], 'w-', linewidth=2, transform=proj)
    ax4.coastlines(resolution='50m', color='k', linewidth=0.6)
    ax4.add_feature(cfeat.BORDERS, linewidth=0.4)
    ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax4.set_title('Prediction Error (Pred - Obs)')
    fig.colorbar(im4, ax=ax4, label='Error (mm/h)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_convergence_buffer(pred, target, lines, buffer=3, save_path=None):
    """
    可视化辐合线缓冲区和降水误差
    
    Args:
        pred: 预测降水
        target: 观测降水
        lines: 检测到的辐合线
        buffer: 缓冲区宽度
        save_path: 保存路径（可选）
    """
    # 创建缓冲区掩码
    buffer_mask = np.zeros_like(pred, dtype=bool)
    for line in lines:
        for y, x in line:
            y1, y2 = max(0, y-buffer), min(pred.shape[0], y+buffer+1)
            x1, x2 = max(0, x-buffer), min(pred.shape[1], x+buffer+1)
            buffer_mask[y1:y2, x1:x2] = True
    
    # 自定义颜色映射
    bounds_in = [0, 0.1, 1, 2.5, 5, 10, 20, 30, 40, 50, 60]
    turbo_camp = get_cmap('turbo')
    colors_list = ['white']
    for i in range(10):
        if i > 0:
            colors_list.append(turbo_camp(i / 9))
    custom_cmap = ListedColormap(colors_list)
    norm_in = BoundaryNorm(bounds_in, custom_cmap.N)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 观测降水
    im1 = axes[0].imshow(target, cmap=custom_cmap, norm=norm_in)
    axes[0].contour(buffer_mask, colors='red', levels=[0.5], linewidths=2)
    axes[0].set_title('Observed Precipitation')
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=bounds_in[:-1])
    cbar1.set_ticklabels(['0', '0.1', '1', '2.5', '5', '10', '20', '30', '40', '50'])
    
    # 预测降水
    im2 = axes[1].imshow(pred, cmap=custom_cmap, norm=norm_in)
    axes[1].contour(buffer_mask, colors='red', levels=[0.5], linewidths=2)
    axes[1].set_title('Predicted Precipitation')
    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=bounds_in[:-1])
    cbar2.set_ticklabels(['0', '0.1', '1', '2.5', '5', '10', '20', '30', '40', '50'])
    
    # 误差（仅显示缓冲区内外）
    error = pred - target
    im3 = axes[2].imshow(error, cmap='RdBu_r', vmin=-20, vmax=20)
    axes[2].contour(buffer_mask, colors='red', levels=[0.5], linewidths=2)
    axes[2].set_title('Prediction Error')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # 创建模拟数据
    np.random.seed(42)
    u = np.random.randn(100, 100) * 2
    v = np.random.randn(100, 100) * 2
    
    # 添加一条模拟的辐合线
    u[45:55, :] -= 5
    v[:, 45:55] += 5
    
    # 模拟降水
    target = np.zeros((100, 100))
    target[48:52, 40:60] = np.random.rand(4, 20) * 30 + 20
    pred = target.copy() + np.random.randn(100, 100) * 5
    
    # 检测辐合线
    lines, mask = convergence_line_detection(u, v, threshold=5e-4)
    
    # 评估精度
    results = evaluate_precip_along_convergence(pred, target, u, v, lines, buffer=3)
    
    print(f"辐合线附近RMSE: {results['along_convergence']['rmse']:.2f}")
    print(f"远离辐合线RMSE: {results['away_convergence']['rmse']:.2f}")
    print(f"提升度: {results['improvement_ratio']:.2f}")
    
    # 可视化
    plot_convergence_and_precip(pred, target, u, v, lines, mask)
    plot_convergence_buffer(pred, target, lines, buffer=3)
