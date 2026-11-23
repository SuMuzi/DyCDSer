def vector_correlation_analysis(pred, target, u, v):
    """
    分析预测误差与风场的关系
    """
    # 误差场
    error = pred - target
    abs_error = np.abs(error)
    
    # 风场特征
    wind_speed = np.sqrt(u**2 + v**2)
    divergence = compute_divergence(u, v)
    vorticity = compute_vorticity(u, v)
    
    # 分箱统计
    bins = np.percentile(wind_speed, np.linspace(0, 100, 11))  # 10分位
    
    results = []
    for i in range(len(bins)-1):
        mask = (wind_speed >= bins[i]) & (wind_speed < bins[i+1])
        if mask.sum() < 10:
            continue
            
        results.append({
            'wind_speed_range': (bins[i], bins[i+1]),
            'mean_wind': wind_speed[mask].mean(),
            'rmse': np.sqrt(np.mean(error[mask]**2)),
            'mae': np.mean(abs_error[mask]),
            'bias': np.mean(error[mask]),
            'correlation': np.corrcoef(pred[mask], target[mask])[0, 1] if len(pred[mask]) > 1 else 0,
            'sample_size': mask.sum()
        })
    
    return results


def plot_error_vs_wind(error_stats, save_path):
    """
    绘制误差随风速变化
    """
    import matplotlib.pyplot as plt
    
    wind_centers = [r['mean_wind'] for r in error_stats]
    rmses = [r['rmse'] for r in error_stats]
    maes = [r['mae'] for r in error_stats]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 误差随风速
    axes[0].plot(wind_centers, rmses, 'b-o', label='RMSE')
    axes[0].plot(wind_centers, maes, 'r-s', label='MAE')
    axes[0].set_xlabel('Wind Speed (m/s)')
    axes[0].set_ylabel('Error (mm/h)')
    axes[0].set_title('Precipitation Error vs Wind Speed')
    axes[0].legend()
    axes[0].grid(True)
    
    # 样本分布
    samples = [r['sample_size'] for r in error_stats]
    axes[1].bar(range(len(samples)), samples, color='gray', alpha=0.7)
    axes[1].set_xlabel('Wind Speed Bin')
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Sample Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)