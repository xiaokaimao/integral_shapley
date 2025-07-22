#!/usr/bin/env python
"""
演示边际贡献曲线的光滑性
横坐标：层大小，纵坐标：边际贡献期望，MC采样=1000
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.integral_shapley import stratified_shapley_value_with_plot
from src.utils.utilities import utility_acc
from src.utils.model_utils import return_model


def main():
    """对比支持向量和非支持向量的边际贡献曲线"""
    
    print("SVM支持向量 vs 非支持向量的边际贡献曲线对比")
    print("MC采样次数: 1000")
    print("="*60)
    
    # 准备数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # 训练SVM模型
    from sklearn.svm import SVC
    # final_model = SVC(kernel='rbf', gamma='scale', C=1.0)
    # final_model.fit(x_train, y_train)
    # clf = SVC(kernel='rbf', gamma='scale', C=1.0)
    final_model = return_model('logistic')
    final_model.fit(x_train, y_train)
    clf = return_model('logistic')
    
     
    print(f"数据集大小: {len(x_train)} 个训练点")
    print(f"验证集大小: {len(x_valid)} 个验证点")
    
    
    support_vector_indices = final_model.support_
    print(f"支持向量数量: {len(support_vector_indices)}")
    print(f"支持向量比例: {len(support_vector_indices)/len(x_train)*100:.1f}%")
    
    # 选择一个支持向量和一个非支持向量
    support_point = support_vector_indices[18]  # 第一个支持向量
    
    # 找一个非支持向量
    non_support_point = None
    flag = 0
    for i in range(len(x_train)):
        if i not in support_vector_indices:
            non_support_point = i
            flag += 1
            if flag == 18:
                break
    # support_point = 73
    # non_support_point = 26
    print(f"\n选择的点:")
    print(f"  支持向量: 索引 {support_point}")
    print(f"  非支持向量: 索引 {non_support_point}")
    
    # 计算支持向量的边际贡献曲线
    print(f"\n计算支持向量 {support_point} 的边际贡献曲线...")
    sv_shapley, sv_layer_sizes, sv_contributions = stratified_shapley_value_with_plot(
        i=support_point,
        X_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        clf=clf,
        final_model=final_model,
        utility_func=utility_acc,
        num_MC=1000,
        plot=False  # 稍后一起画图
    )
    
    # 计算非支持向量的边际贡献曲线
    print(f"计算非支持向量 {non_support_point} 的边际贡献曲线...")
    nsv_shapley, nsv_layer_sizes, nsv_contributions = stratified_shapley_value_with_plot(
        i=non_support_point,
        X_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        clf=clf,
        final_model=final_model,
        utility_func=utility_acc,
        num_MC=1000,
        plot=False
    )
    
    # 对比分析
    print(f"\n对比分析:")
    print(f"  支持向量 {support_point}:")
    print(f"    Shapley值: {sv_shapley:.6f}")
    print(f"    最大边际贡献: {max(sv_contributions):.6f}")
    print(f"    起始贡献: {sv_contributions[0]:.6f}")
    print(f"    标准差: {np.std(sv_contributions):.6f}")
    
    print(f"  非支持向量 {non_support_point}:")
    print(f"    Shapley值: {nsv_shapley:.6f}")
    print(f"    最大边际贡献: {max(nsv_contributions):.6f}")
    print(f"    起始贡献: {nsv_contributions[0]:.6f}")
    print(f"    标准差: {np.std(nsv_contributions):.6f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 子图1: 原始联盟大小
    plt.subplot(2, 2, 1)
    plt.plot(sv_layer_sizes, sv_contributions, 'r-o', linewidth=2, markersize=3, 
             label=f'Support Vector {support_point} (SV={sv_shapley:.4f})')
    plt.plot(nsv_layer_sizes, nsv_contributions, 'b-s', linewidth=2, markersize=3,
             label=f'Non-Support Vector {non_support_point} (SV={nsv_shapley:.4f})')
    plt.xlabel('Coalition Size')
    plt.ylabel('Expected Marginal Contribution')
    plt.title('Marginal Contribution Curves: SV vs Non-SV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 归一化横坐标 [0,1]
    plt.subplot(2, 2, 2)
    N = len(sv_layer_sizes) - 1
    sv_normalized_x = [s/N for s in sv_layer_sizes]
    nsv_normalized_x = [s/N for s in nsv_layer_sizes]
    
    plt.plot(sv_normalized_x, sv_contributions, 'r-o', linewidth=2, markersize=3,
             label=f'Support Vector {support_point}')
    plt.fill_between(sv_normalized_x, sv_contributions, alpha=0.2, color='red')
    
    plt.plot(nsv_normalized_x, nsv_contributions, 'b-s', linewidth=2, markersize=3,
             label=f'Non-Support Vector {non_support_point}')
    plt.fill_between(nsv_normalized_x, nsv_contributions, alpha=0.2, color='blue')
    
    plt.xlabel('Normalized Coalition Size (t)')
    plt.ylabel('Expected Marginal Contribution E[Δ(t,i)]')
    plt.title('Normalized View: Area = Shapley Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 曲线差异分析
    plt.subplot(2, 2, 3)
    # 计算差异（插值到相同的x轴）
    common_x = np.linspace(0, 1, 100)
    sv_interp = np.interp(common_x, sv_normalized_x, sv_contributions)
    nsv_interp = np.interp(common_x, nsv_normalized_x, nsv_contributions)
    difference = sv_interp - nsv_interp
    
    plt.plot(common_x, difference, 'g-', linewidth=2, label='SV - Non-SV')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.fill_between(common_x, difference, alpha=0.3, color='green')
    plt.xlabel('Normalized Coalition Size (t)')
    plt.ylabel('Contribution Difference')
    plt.title('Difference: Support Vector - Non-Support Vector')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 统计对比
    plt.subplot(2, 2, 4)
    categories = ['Shapley Value', 'Max Contribution', 'Initial Contribution', 'Std Dev']
    sv_stats = [sv_shapley, max(sv_contributions), sv_contributions[0], np.std(sv_contributions)]
    nsv_stats = [nsv_shapley, max(nsv_contributions), nsv_contributions[0], np.std(nsv_contributions)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, sv_stats, width, label=f'Support Vector {support_point}', color='red', alpha=0.7)
    plt.bar(x + width/2, nsv_stats, width, label=f'Non-Support Vector {non_support_point}', color='blue', alpha=0.7)
    
    plt.xlabel('Statistics')
    plt.ylabel('Value')
    plt.title('Statistical Comparison')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    
    # 添加数值标签
    for i, (sv_val, nsv_val) in enumerate(zip(sv_stats, nsv_stats)):
        plt.text(i - width/2, sv_val + max(sv_stats)*0.01, f'{sv_val:.4f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, nsv_val + max(nsv_stats)*0.01, f'{nsv_val:.4f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/plots/svm_sv_vs_nonsv_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 理论解释
    print(f"\n理论解释:")
    print(f"1. 支持向量特征:")
    print(f"   - 位于决策边界附近，对模型关键")
    print(f"   - 起始边际贡献通常较高")
    print(f"   - 曲线可能快速衰减")
    
    print(f"2. 非支持向量特征:")
    print(f"   - 远离决策边界，对当前决策影响小")
    print(f"   - 起始边际贡献较低")
    print(f"   - 曲线相对平缓")
    
    print(f"3. 积分视角:")
    print(f"   - 支持向量的积分面积（Shapley值）通常更大")
    print(f"   - 体现了SVM的稀疏性原理")
    print(f"   - 验证了积分方法能捕捉模型内在特性")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('results/plots', exist_ok=True)
    
    main()