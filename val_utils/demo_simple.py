"""
简化版演示：统计显著性检验可视化
不依赖 torch，直接展示统计结果
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def generate_simulation_data(n_samples=20):
    """生成模拟数据"""
    np.random.seed(42)
    
    # 基准模型（较大误差）
    baseline_strong = np.random.normal(5.0, 1.0, n_samples)
    baseline_conv = np.random.normal(4.5, 0.9, n_samples)
    baseline_all = np.random.normal(3.5, 0.8, n_samples)
    
    # 改进模型（较小误差）
    guided_strong = baseline_strong * np.random.uniform(0.65, 0.85, n_samples)
    guided_conv = baseline_conv * np.random.uniform(0.60, 0.80, n_samples)
    guided_all = baseline_all * np.random.uniform(0.85, 0.95, n_samples)
    
    return {
        'strong_wind': {'baseline': baseline_strong, 'guided': guided_strong},
        'convergence': {'baseline': baseline_conv, 'guided': guided_conv},
        'all': {'baseline': baseline_all, 'guided': guided_all}
    }


def significance_test(baseline, guided):
    """执行配对t检验"""
    t_stat, p_value = stats.ttest_rel(baseline, guided)
    diff = baseline - guided
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'improvement_percent': (1 - np.mean(guided)/np.mean(baseline)) * 100,
        'baseline_mean': np.mean(baseline),
        'guided_mean': np.mean(guided),
        'sample_size': len(baseline)
    }


def plot_results(data):
    """可视化结果"""
    fig = plt.figure(figsize=(16, 10))
    
    regions = ['strong_wind', 'convergence', 'all']
    region_names = ['Strong Wind', 'Convergence', 'Overall']
    
    # 1. RMSE对比柱状图
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(3)
    width = 0.35
    
    baseline_means = [np.mean(data[r]['baseline']) for r in regions]
    guided_means = [np.mean(data[r]['guided']) for r in regions]
    
    ax1.bar(x - width/2, baseline_means, width, label='Baseline', color='#1f77b4')
    ax1.bar(x + width/2, guided_means, width, label='Wind-Guided', color='#2ca02c')
    ax1.set_ylabel('RMSE (mm/h)')
    ax1.set_title('RMSE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_names, rotation=15)
    ax1.legend()
    
    # 2. 改进幅度
    ax2 = fig.add_subplot(2, 2, 2)
    improvements = []
    for r in regions:
        res = significance_test(data[r]['baseline'], data[r]['guided'])
        improvements.append(res['improvement_percent'])
    
    bars = ax2.bar(region_names, improvements, color='#ff7f0e', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Improvement Percentage')
    
    # 3. p值和效应量
    ax3 = fig.add_subplot(2, 2, 3)
    p_values = []
    cohens_d = []
    
    for r in regions:
        res = significance_test(data[r]['baseline'], data[r]['guided'])
        p_values.append(-np.log10(res['p_value']))
        cohens_d.append(res['cohens_d'])
    
    ax3.scatter(p_values, cohens_d, s=100, c=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.axhline(y=0.5, color='gray', linestyle='--', label='Effect size')
    ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    
    for i, name in enumerate(region_names):
        ax3.annotate(name, (p_values[i], cohens_d[i]), xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('-log10(p-value)')
    ax3.set_ylabel("Cohen's d")
    ax3.set_title('Significance vs Effect Size')
    ax3.legend()
    
    # 4. 配对散点图（强风区）
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(data['strong_wind']['baseline'], data['strong_wind']['guided'], 
                alpha=0.6, color='#1f77b4')
    ax4.plot([3, 7], [3, 7], 'r--')
    ax4.set_xlabel('Baseline RMSE')
    ax4.set_ylabel('Wind-Guided RMSE')
    ax4.set_title('Paired Comparison (Strong Wind)')
    
    plt.tight_layout()
    plt.savefig('significance_demo.png', dpi=150, bbox_inches='tight')
    print("图像已保存: significance_demo.png")
    plt.show()


def print_report(data):
    """打印统计报告"""
    print("=" * 60)
    print("风场引导降水降尺度 - 显著性检验报告")
    print("=" * 60)
    
    regions = ['strong_wind', 'convergence', 'all']
    region_names = ['Strong Wind', 'Convergence', 'Overall']
    
    for r, name in zip(regions, region_names):
        res = significance_test(data[r]['baseline'], data[r]['guided'])
        
        print(f"\n{name}:")
        print(f"  样本数: {res['sample_size']}")
        print(f"  基准RMSE: {res['baseline_mean']:.3f}")
        print(f"  改进RMSE: {res['guided_mean']:.3f}")
        print(f"  改进幅度: {res['improvement_percent']:.2f}%")
        print(f"  t统计量: {res['t_statistic']:.3f}")
        print(f"  p值: {res['p_value']:.4f}")
        print(f"  Cohen's d: {res['cohens_d']:.3f}")
        print(f"  显著性: {'是' if res['significant'] else '否'}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # 生成数据
    data = generate_simulation_data(n_samples=20)
    
    # 打印报告
    print_report(data)
    
    # 可视化
    plot_results(data)