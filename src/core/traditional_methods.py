#!/usr/bin/env python
"""
Traditional Shapley value computation methods.

This module implements classical approaches for Shapley value computation:
- Monte Carlo sampling (permutation-based)
- Exact computation (for small datasets)
- Stratified sampling (layer-wise)
"""

import numpy as np
import itertools
import random
from math import factorial
from tqdm import tqdm
from sklearn.base import clone


def monte_carlo_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                            utility_func, num_samples=10000):
    """
    Traditional Monte Carlo estimation of Shapley values for comparison.
    
    Args:
        i: Target data point index
        X_train, y_train: Training data
        x_valid, y_valid: Validation data
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_samples: Number of random permutations
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    n = X_train.shape[0]
    indices = list(range(n))
    contributions = []
    
    for _ in tqdm(range(num_samples), desc=f"MC sampling for data point {i}", leave=False):
        perm = np.random.permutation(indices)
        pos = np.where(perm == i)[0][0]
        S = list(perm[:pos])
        
        X_S = X_train[S]
        y_S = y_train[S]
        
        try:
            v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
        except:
            v_S = 0.0
        
        X_S_i = np.vstack([X_S, X_train[i]]) if len(S) > 0 else X_train[i].reshape(1, -1)
        y_S_i = np.append(y_S, y_train[i])
        
        try:
            v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
        except:
            v_S_i = 0.0
        
        contributions.append(v_S_i - v_S)
    
    return np.mean(contributions)


def exact_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, utility_func):
    """
    Exact Shapley value computation (for small datasets only).
    
    Args:
        i: Target data point index
        X_train, y_train: Training data
        x_valid, y_valid: Validation data  
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        
    Returns:
        shapley_value: Exact Shapley value for data point i
    """
    n = X_train.shape[0]
    indices = list(range(n))
    indices.remove(i)
    shapley_value = 0.0
    
    for r in tqdm(range(len(indices) + 1), desc=f"Exact Shapley for point {i}"):
        for subset in itertools.combinations(indices, r):
            S = list(subset)
            X_S = X_train[S]
            y_S = y_train[S]
            
            try:
                v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S = 0.0
            
            X_S_i = np.vstack([X_S, X_train[i]]) if len(S) > 0 else X_train[i].reshape(1, -1)
            y_S_i = np.append(y_S, y_train[i])
            
            try:
                v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S_i = 0.0
            
            delta = v_S_i - v_S
            weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            shapley_value += weight * delta
    
    return shapley_value


def stratified_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                            utility_func, num_MC, plot=False, save_path=None, return_details=False):
    """
    使用分层采样方法估计数据点 i 的 Shapley 值。
    
    通过对每个可能的子集大小j进行采样，计算目标数据点 i 的边际贡献：
      Δ = utility(S ∪ {i}) - utility(S)
    
    参数:
      i: 目标数据点在 X_train 中的索引
      X_train, y_train: 训练数据集
      x_valid, y_valid: 验证数据集（用于计算效用）
      clf: 待训练的分类器（例如 SVC）
      final_model: 全量数据训练得到的模型
      utility_func: 效用函数
      num_MC: 每个子集大小的蒙特卡洛采样次数
      plot: 是否生成可视化图表
      save_path: 图表保存路径（可选）
      return_details: 是否返回详细信息 (layer_sizes, layer_contributions)
    
    返回:
      如果 return_details=False: shapley_value（浮点数）
      如果 return_details=True: (shapley_value, layer_sizes, layer_contributions)
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    total = X_train.shape[0]
    # 从训练集中移除目标数据点 i，构建候选池
    indices = [j for j in range(total) if j != i]
    candidate_x = X_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)  # 候选数据数
    
    # 存储每一层(每个大小)的平均边际贡献
    layer_sizes = []
    layer_contributions = []
    strata_values = []
    
    # print("here")
    # 对每个可能的子集大小进行采样
    for j in tqdm(range(N+1), desc=f"Computing stratified Shapley for point {i}"):
        mc_values = []
        # 对大小为j的子集进行num_MC次采样
        for _ in range(num_MC):
            # 如果j=0，则使用空集
            if j == 0:
                sample_indices = []
            else:
                sample_indices = random.sample(range(N), j)
            
            X_sub = candidate_x[sample_indices] if j > 0 else np.empty((0, X_train.shape[1]))
            y_sub = candidate_y[sample_indices] if j > 0 else np.empty(0)
            
            try:
                # 计算效用 v(S)
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except Exception:
                # 如果处理空子集或计算出错，返回 0
                util_S = 0.0
                
            # 构建 S ∪ {i}
            X_sub_i = np.concatenate([X_sub, X_train[i].reshape(1, -1)], axis=0)
            y_sub_i = np.concatenate([y_sub, np.array([y_train[i]])], axis=0)
            
            try:
                # 计算效用 v(S ∪ {i})
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except Exception:
                util_S_i = 0.0
                
            # 计算边际贡献
            mc_values.append(util_S_i - util_S)
        
        # 计算该大小子集的平均边际贡献
        if mc_values:
            avg_contribution = np.mean(mc_values)
            # 分层采样中每层权重相等（类似积分的矩形法则）
            weight = 1/total
            
            layer_sizes.append(j)
            layer_contributions.append(avg_contribution)
            strata_values.append(weight * avg_contribution)
    
    # 总的Shapley值是所有层的加权和
    shapley_value = sum(strata_values)
    
    # 生成可视化图表
    if plot:
        import matplotlib.pyplot as plt
        
        # Create subplot with two x-axes: coalition size and normalized [0,1]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Original coalition size
        ax1.plot(layer_sizes, layer_contributions, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Coalition Size')
        ax1.set_ylabel('Expected Marginal Contribution')
        ax1.set_title(f'Stratified Sampling: Marginal Contribution Distribution for Data Point {i}')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add data labels (subsample for readability)
        step = max(1, len(layer_sizes)//10)
        for x, y in zip(layer_sizes[::step], layer_contributions[::step]):
            ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # Plot 2: Normalized [0,1] scale showing area under curve = Shapley value
        N_plot = len(layer_sizes) - 1  # Max coalition size
        normalized_x = [s/N_plot for s in layer_sizes] if N_plot > 0 else layer_sizes
        
        ax2.plot(normalized_x, layer_contributions, 'go-', linewidth=2, markersize=6)
        ax2.fill_between(normalized_x, layer_contributions, alpha=0.3, color='green', 
                        label=f'Area = Shapley Value = {shapley_value:.6f}')
        ax2.set_xlabel('Normalized Coalition Size (t)')
        ax2.set_ylabel('Expected Marginal Contribution E[Δ(t,i)]')
        ax2.set_title(f'Integral Formulation: ∫₀¹ E[Δ(t,i)] dt = {shapley_value:.6f}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Add mathematical annotation
        max_contrib = max(layer_contributions) if layer_contributions else 1
        ax2.text(0.02, max_contrib*0.8, 
                f'Shapley Value = ∫₀¹ E[Δ(t,i)] dt\n= Area under curve\n= {shapley_value:.6f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    # 返回结果
    if return_details:
        return shapley_value, layer_sizes, layer_contributions
    else:
        return shapley_value


def stratified_shapley_value_with_plot(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                                      utility_func, num_MC, plot=True, save_path=None):
    """
    兼容性包装器：调用统一的 stratified_shapley_value 函数。
    
    此函数已被合并到 stratified_shapley_value 中。保留此包装器以保持向后兼容性。
    
    Args:
      i: Target data point index in X_train
      X_train, y_train: Training dataset
      x_valid, y_valid: Validation dataset (for utility computation)
      clf: Classifier to train on subsets
      final_model: Model trained on full data
      utility_func: Utility function
      num_MC: Monte Carlo samples per coalition size
      plot: Whether to generate plot
      save_path: Plot save path (optional)
    
    Returns:
      tuple: (shapley_value, layer_sizes, layer_contributions)
    """
    return stratified_shapley_value(
        i, X_train, y_train, x_valid, y_valid, clf, final_model,
        utility_func, num_MC, plot=plot, save_path=save_path, return_details=True
    )