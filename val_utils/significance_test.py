from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def significance_test(baseline_results, wind_guided_results, test_type='strong_wind'):
    """
    检验风场引导是否显著改善强风/辐合区的降水预测精度
    
    Args:
        baseline_results: 基准模型的结果列表，每个元素包含 {'strong_wind': {'RMSE': ...}, 'convergence': {'RMSE': ...}}
        wind_guided_results: 风场引导模型的结果列表，结构同上
        test_type: 检验类型，可选 'strong_wind' 或 'convergence'
    
    Returns:
        dict: 包含统计检验结果
        
    Raises:
        ValueError: 如果输入数据无效
    """
    # 数据校验
    if len(baseline_results) == 0 or len(wind_guided_results) == 0:
        raise ValueError("输入结果列表不能为空")
    
    if len(baseline_results) != len(wind_guided_results):
        raise ValueError("两个结果列表长度必须一致")
    
    # 提取指定类型的RMSE
    try:
        baseline_values = [r[test_type]['RMSE'] for r in baseline_results]
        guided_values = [r[test_type]['RMSE'] for r in wind_guided_results]
    except KeyError:
        raise ValueError(f"结果字典中缺少 '{test_type}' 或 'RMSE' 键")
    
    baseline_values = np.array(baseline_values)
    guided_values = np.array(guided_values)
    
    # 检查是否有足够的数据点
    if len(baseline_values) < 2:
        raise ValueError("至少需要2个样本进行t检验")
    
    # 检查是否有NaN值
    if np.any(np.isnan(baseline_values)) or np.any(np.isnan(guided_values)):
        raise ValueError("输入数据包含NaN值")
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(baseline_values, guided_values)
    
    # 效应量 (Cohen's d for paired samples)
    diff = baseline_values - guided_values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan
    
    # 计算改进百分比
    baseline_mean = np.mean(baseline_values)
    guided_mean = np.mean(guided_values)
    improvement_percent = (1 - guided_mean / baseline_mean) * 100 if baseline_mean != 0 else np.nan
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'improvement_percent': improvement_percent,
        'baseline_mean_rmse': baseline_mean,
        'guided_mean_rmse': guided_mean,
        'sample_size': len(baseline_values),
        'test_type': test_type
    }


def compare_metrics(baseline_results, wind_guided_results):
    """
    全面比较基准模型和风场引导模型在不同区域的性能
    
    Args:
        baseline_results: 基准模型结果列表
        wind_guided_results: 风场引导模型结果列表
    
    Returns:
        dict: 包含强风区和辐合区的检验结果
    """
    results = {}
    
    for test_type in ['strong_wind', 'convergence']:
        try:
            results[test_type] = significance_test(baseline_results, wind_guided_results, test_type)
        except Exception as e:
            results[test_type] = {
                'error': str(e),
                'test_type': test_type
            }
    
    # 计算整体比较
    try:
        baseline_all = [r['all']['RMSE'] for r in baseline_results]
        guided_all = [r['all']['RMSE'] for r in wind_guided_results]
        t_stat, p_value = stats.ttest_rel(baseline_all, guided_all)
        results['overall'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'improvement_percent': (1 - np.mean(guided_all)/np.mean(baseline_all)) * 100,
            'test_type': 'overall'
        }
    except Exception as e:
        results['overall'] = {'error': str(e), 'test_type': 'overall'}
    
    return results


def print_significance_results(results):
    """
    打印显著性检验结果的可读报告
    
    Args:
        results: compare_metrics 返回的结果字典
    """
    print("=" * 60)
    print("风场引导降水降尺度 - 显著性检验报告")
    print("=" * 60)
    
    for region, stats_dict in results.items():
        if 'error' in stats_dict:
            print(f"\n{region.capitalize()}:")
            print(f"  错误: {stats_dict['error']}")
            continue
        
        print(f"\n{region.capitalize()}:")
        print(f"  样本数: {stats_dict.get('sample_size', 'N/A')}")
        print(f"  基准RMSE: {stats_dict.get('baseline_mean_rmse', 'N/A'):.3f}")
        print(f"  改进RMSE: {stats_dict.get('guided_mean_rmse', 'N/A'):.3f}")
        print(f"  改进幅度: {stats_dict.get('improvement_percent', 'N/A'):.2f}%")
        print(f"  t统计量: {stats_dict.get('t_statistic', 'N/A'):.3f}")
        print(f"  p值: {stats_dict.get('p_value', 'N/A'):.4f}")
        print(f"  Cohen's d: {stats_dict.get('cohens_d', 'N/A'):.3f}")
        print(f"  显著性: {'是' if stats_dict.get('significant', False) else '否'}")
        
        # 解释效应量
        cohens_d = stats_dict.get('cohens_d', 0)
        if cohens_d is not np.nan:
            if abs(cohens_d) < 0.2:
                effect = '极小'
            elif abs(cohens_d) < 0.5:
                effect = '中等'
            elif abs(cohens_d) < 0.8:
                effect = '较大'
            else:
                effect = '很大'
            print(f"  效应量大小: {effect}")
    
    print("\n" + "=" * 60)
