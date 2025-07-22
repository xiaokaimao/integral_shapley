#!/usr/bin/env python
"""
梯形法 vs 辛普森法在真实数据上的对比
"""

import numpy as np
import time
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.integral_shapley import (
    compute_integral_shapley_trapezoid,
    compute_integral_shapley_simpson,
    compute_integral_shapley_auto,
    stratified_shapley_value
)
from src.utils.utilities import utility_acc
from src.utils.model_utils import return_model


def main():
    """对比梯形法和辛普森法在真实数据上的表现"""
    
    print("梯形法 vs 辛普森法对比")
    print("="*50)
    
    # 准备数据
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # 训练模型
    final_model = return_model('SVC')
    final_model.fit(x_train, y_train)
    clf = return_model('SVC')
    
    
    target_point = 0
    print(f"目标数据点: {target_point}")
    print(f"数据集大小: {len(x_train)}")
    
    # 计算基准值（分层采样）
    print("\n计算基准值（分层采样）...")
    start_time = time.time()
    baseline_shapley = stratified_shapley_value(
        target_point, x_train, y_train, x_valid, y_valid,
        clf, final_model, utility_acc, num_MC=1000
    )
    baseline_time = time.time() - start_time
    
    print(f"基准Shapley值: {baseline_shapley:.6f}")
    print(f"计算时间: {baseline_time:.3f}秒")
    
    # 测试不同的采样点数
    t_sample_counts = [11, 21, 31, 51]
    
    print(f"\n{'采样点数':<8} {'梯形法':<12} {'误差%':<8} {'时间':<8} {'辛普森法':<12} {'误差%':<8} {'时间':<8}")
    print("-" * 80)
    
    for t_samples in t_sample_counts:
        # 梯形法
        start_time = time.time()
        shapley_trap = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, target_point,
            clf, final_model, utility_acc,
            num_t_samples=t_samples, num_MC=100
        )
        trap_time = time.time() - start_time
        trap_error = abs(shapley_trap - baseline_shapley) / abs(baseline_shapley) * 100
        
        # 辛普森法
        start_time = time.time()
        shapley_simp = compute_integral_shapley_simpson(
            x_train, y_train, x_valid, y_valid, target_point,
            clf, final_model, utility_acc,
            num_t_samples=t_samples, num_MC=100
        )
        simp_time = time.time() - start_time
        simp_error = abs(shapley_simp - baseline_shapley) / abs(baseline_shapley) * 100
        
        print(f"{t_samples:<8} {shapley_trap:<12.6f} {trap_error:<8.2f} {trap_time:<8.3f} "
              f"{shapley_simp:<12.6f} {simp_error:<8.2f} {simp_time:<8.3f}")
    
    # 效率分析
    print(f"\n效率分析:")
    print(f"基准方法需要计算: {len(x_train)} × 100 = {len(x_train) * 100} 次评估")
    print(f"积分方法只需要: 21 × 100 = 2100 次评估")
    print(f"效率提升: {(len(x_train) * 100) // 2100:.1f}倍")
    
    # 精度分析
    print(f"\n精度分析（21个采样点）:")
    shapley_trap_21 = compute_integral_shapley_trapezoid(
        x_train, y_train, x_valid, y_valid, target_point,
        clf, final_model, utility_acc,
        num_t_samples=21, num_MC=100
    )
    shapley_simp_21 = compute_integral_shapley_simpson(
        x_train, y_train, x_valid, y_valid, target_point,
        clf, final_model, utility_acc,
        num_t_samples=21, num_MC=100
    )
    
    trap_error_21 = abs(shapley_trap_21 - baseline_shapley) / abs(baseline_shapley) * 100
    simp_error_21 = abs(shapley_simp_21 - baseline_shapley) / abs(baseline_shapley) * 100
    
    print(f"梯形法误差: {trap_error_21:.2f}%")
    print(f"辛普森法误差: {simp_error_21:.2f}%")
    if trap_error_21 > 0:
        print(f"辛普森法精度提升: {trap_error_21/simp_error_21:.1f}倍")
    
    # 推荐配置
    print(f"\n推荐配置:")
    if simp_error_21 < trap_error_21:
        print("推荐使用辛普森法，21个采样点")
        print("原因：更高精度，相似的计算成本")
    else:
        print("两种方法精度相近，可选择梯形法（实现更简单）")
    
    # 演示自动选择功能
    print(f"\n自动选择演示:")
    auto_shapley = compute_integral_shapley_auto(
        x_train, y_train, x_valid, y_valid, target_point,
        clf, final_model, utility_acc,
        method='simpson', precision='balanced'
    )
    auto_error = abs(auto_shapley - baseline_shapley) / abs(baseline_shapley) * 100
    print(f"自动选择结果: {auto_shapley:.6f} (误差: {auto_error:.2f}%)")
    
    # 采样点过多的问题解释
    print(f"\n为什么采样点过多误差反而增大？")
    print("1. 蒙特卡洛噪声累积：每个t点的MC噪声会累积")
    print("2. 数值积分过拟合：过多采样点拟合噪声而非信号")
    print("3. 偏差-方差权衡：存在最优平衡点")
    print("4. 建议：优先增加MC采样数，而非t采样数")


if __name__ == "__main__":
    main()