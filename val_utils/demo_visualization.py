"""
演示脚本：降水降尺度评估可视化
展示如何使用所有评估和可视化工具
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from val_utils.object_based_evaluation import object_based_evaluation
from val_utils.convergence_line_detection import (
    convergence_line_detection,
    evaluate_precip_along_convergence,
    plot_convergence_and_precip,
    plot_convergence_buffer
)
from val_utils.visualize_wind_precip_coupling import visualize_wind_precip_coupling
from val_utils.significance_test import (
    compare_metrics,
    print_significance_results,
    plot_significance_results,
    plot_paired_comparison
)


def generate_simulation_data(n_samples=10):
    """生成模拟数据用于演示"""
    results = []
    
    np.random.seed(42)
    for _ in range(n_samples):
        # 模拟强风区RMSE（基准模型较大，改进模型较小）
        baseline_strong = np.random.normal(5.0, 1.0)
        guided_strong = baseline_strong * np.random.uniform(0.65, 0.85)
        
        # 模拟辐合区RMSE
        baseline_conv = np.random.normal(4.5, 0.9)
        guided_conv = baseline_conv * np.random.uniform(0.60, 0.80)
        
        # 模拟整体RMSE
        baseline_all = np.random.normal(3.5, 0.8)
        guided_all = baseline_all * np.random.uniform(0.85, 0.95)
        
        results.append({
            'baseline': {
                'strong_wind': {'RMSE': baseline_strong},
                'convergence': {'RMSE': baseline_conv},
                'all': {'RMSE': baseline_all}
            },
            'guided': {
                'strong_wind': {'RMSE': guided_strong},
                'convergence': {'RMSE': guided_conv},
                'all': {'RMSE': guided_all}
            }
        })
    
    return results


def demo_single_case():
    """演示单个案例的可视化"""
    print("=" * 60)
    print("演示1：单个案例的风场-降水耦合分析")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    u = np.random.randn(100, 100) * 2
    v = np.random.randn(100, 100) * 2
    
    # 添加模拟辐合线
    u[45:55, :] -= 5
    v[:, 45:55] += 5
    
    # 模拟降水
    target = np.zeros((100, 100))
    target[48:52, 40:60] = np.random.rand(4, 20) * 30 + 20
    pred_baseline = target.copy() + np.random.randn(100, 100) * 8
    pred_guided = target.copy() + np.random.randn(100, 100) * 4  # 改进模型误差更小
    
    # 检测辐合线
    lines, mask = convergence_line_detection(u, v, threshold=5e-4)
    
    # 评估精度
    results_baseline = evaluate_precip_along_convergence(pred_baseline, target, u, v, lines)
    results_guided = evaluate_precip_along_convergence(pred_guided, target, u, v, lines)
    
    print(f"基准模型 - 辐合线附近RMSE: {results_baseline['along_convergence']['rmse']:.2f}")
    print(f"改进模型 - 辐合线附近RMSE: {results_guided['along_convergence']['rmse']:.2f}")
    print(f"改进幅度: {(1 - results_guided['along_convergence']['rmse']/results_baseline['along_convergence']['rmse'])*100:.1f}%")
    
    # 可视化
    plot_convergence_and_precip(pred_guided, target, u, v, lines, mask)
    plot_convergence_buffer(pred_guided, target, lines, buffer=3)
    visualize_wind_precip_coupling(pred_guided, target, u, v, 
                                    save_path='demo_wind_precip_coupling.png',
                                    case_name='Simulation Case')


def demo_statistical_test():
    """演示统计检验可视化"""
    print("\n" + "=" * 60)
    print("演示2：统计显著性检验")
    print("=" * 60)
    
    # 生成模拟实验数据
    simulation_data = generate_simulation_data(n_samples=20)
    
    # 提取结果
    baseline_results = [r['baseline'] for r in simulation_data]
    guided_results = [r['guided'] for r in simulation_data]
    
    # 执行检验
    results = compare_metrics(baseline_results, guided_results)
    
    # 打印报告
    print_significance_results(results)
    
    # 可视化
    plot_significance_results(results, save_path='demo_significance.png')
    
    # 配对对比
    baseline_strong = [r['strong_wind']['RMSE'] for r in baseline_results]
    guided_strong = [r['strong_wind']['RMSE'] for r in guided_results]
    plot_paired_comparison(np.array(baseline_strong), np.array(guided_strong),
                           save_path='demo_paired_comparison.png')


def demo_object_based_evaluation():
    """演示基于对象的评估"""
    print("\n" + "=" * 60)
    print("演示3：基于对象的评估")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    target = np.zeros((64, 64))
    # 添加几个降水单体
    for i in range(3):
        y, x = 20 + i*15, 20 + i*10
        target[y-3:y+4, x-3:x+4] = np.random.rand(7, 7) * 40 + 10
    
    pred = target.copy() + np.random.randn(64, 64) * 3
    
    # 添加风场
    u = np.random.randn(64, 64) * 3
    v = np.random.randn(64, 64) * 3
    
    # 对象评估
    results = object_based_evaluation(pred, target, u, v, threshold=5.0)
    
    print(f"检测到降水对象数: {results['all']['match_count']}")
    print(f"平均位移误差: {results['all']['mean_displacement']:.2f} 像素")
    
    if results['strong_wind']:
        strong_wind_displacement = np.mean([r['displacement'] for r in results['strong_wind']])
        print(f"强风区平均位移误差: {strong_wind_displacement:.2f} 像素")
    
    if results['convergence']:
        conv_displacement = np.mean([r['displacement'] for r in results['convergence']])
        print(f"辐合区平均位移误差: {conv_displacement:.2f} 像素")


if __name__ == '__main__':
    # 运行所有演示
    demo_single_case()
    demo_statistical_test()
    demo_object_based_evaluation()
    
    print("\n" + "=" * 60)
    print("所有演示完成！图像已保存。")
    print("=" * 60)
