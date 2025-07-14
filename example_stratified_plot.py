#!/usr/bin/env python
"""
分层采样可视化示例
展示数据点在不同联盟大小下的边际贡献分布
"""

import numpy as np
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.integral_shapley import stratified_shapley_value_with_plot
from utils.utilities import utility_acc
from utils.model_utils import return_model


def main():
    """演示分层采样的可视化功能"""
    
    # 1. 准备数据
    print("加载和准备数据...")
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # 2. 训练模型
    print("训练模型...")
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    print(f"训练集大小: {len(x_train)} 个数据点")
    print(f"验证集大小: {len(x_valid)} 个数据点")
    
    # 3. 选择要分析的数据点
    target_points = [50]  # 分析前3个数据点
    
    for i in target_points:
        print(f"\n{'='*50}")
        print(f"分析数据点 {i}")
        print(f"{'='*50}")
        
        # 计算Shapley值并绘制分层采样图
        shapley_value, layer_sizes, layer_contributions = stratified_shapley_value_with_plot(
            i=i,
            X_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            clf=clf,
            final_model=final_model,
            utility_func=utility_acc,
            num_MC=1000,  # 每层蒙特卡洛采样次数
            plot=True,
            save_path=f'results/plots/stratified_plot_point_{i}.png'
        )
        
        print(f"数据点 {i} 的Shapley值: {shapley_value:.6f}")
        
        # 显示一些统计信息
        print(f"总共 {len(layer_sizes)} 层")
        print(f"最大边际贡献: {max(layer_contributions):.6f} (联盟大小: {layer_sizes[np.argmax(layer_contributions)]})")
        print(f"最小边际贡献: {min(layer_contributions):.6f} (联盟大小: {layer_sizes[np.argmin(layer_contributions)]})")
        print(f"边际贡献标准差: {np.std(layer_contributions):.6f}")


def compare_points():
    """Compare marginal contribution distributions of multiple data points"""
    import matplotlib.pyplot as plt
    
    # Prepare data
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # Train models
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    # Compare several data points
    target_points = [0, 10, 20]
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(12, 8))
    
    for idx, i in enumerate(target_points):
        print(f"Computing for data point {i}...")
        
        # Compute without showing individual plots
        shapley_value, layer_sizes, layer_contributions = stratified_shapley_value_with_plot(
            i=i,
            X_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            clf=clf,
            final_model=final_model,
            utility_func=utility_acc,
            num_MC=30,
            plot=False  # Don't show individual plots
        )
        
        plt.plot(layer_sizes, layer_contributions, 'o-', 
                color=colors[idx], linewidth=2, markersize=4,
                label=f'Data Point {i} (SV={shapley_value:.4f})')
    
    plt.xlabel('Coalition Size')
    plt.ylabel('Expected Marginal Contribution')
    plt.title('Multi-Point Marginal Contribution Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Save comparison plot
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/stratified_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved to: results/plots/stratified_comparison.png")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('results/plots', exist_ok=True)
    
    print("分层采样可视化示例")
    print("="*50)
    
    # 运行主要示例
    main()
    
