#!/usr/bin/env python
"""
Integral Shapley Values Computation

This module implements efficient Shapley value computation using integral formulation:
SV_i = ∫_0^1 E[Δ(t,i)] dt

where Δ(t,i) = v(S_t ∪ {i}) - v(S_t) is the marginal contribution of data point i
when added to a random coalition S_t of size determined by t·(N-1) using configurable rounding.

Key advantages:
1. Computational efficiency through smart sampling of t values
2. Exploitation of smoothness properties for better approximation
3. Multiple integration methods (trapezoid, Gaussian quadrature, adaptive)
"""

import argparse
import numpy as np
import random
import pickle
import multiprocessing as mp
from tqdm import tqdm
import itertools
from math import factorial
from typing import Callable, Optional
from scipy.integrate import simpson, fixed_quad
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Import utility functions and model factory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utilities import utility_RKHS, utility_KL, utility_acc, utility_cosine
from utils.model_utils import return_model
from utils.math_utils import compute_coalition_size



def estimate_smoothness(x_train, y_train, x_valid, y_valid, i, clf, final_model, 
                       utility_func, num_probe_points=10, num_MC_probe=20):
    """
    Estimate the smoothness of the integrand E[Δ(t,i)] by computing second differences.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_probe_points: Number of t values to probe for smoothness
        num_MC_probe: Monte Carlo samples per probe point
        
    Returns:
        max_second_diff: Maximum estimated second difference (smoothness indicator)
    """
    total = x_train.shape[0]
    indices = [j for j in range(total) if j != i]
    candidate_x = x_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)
    
    # Sample t values uniformly in (0,1)
    t_values = np.linspace(0.1, 0.9, num_probe_points)
    integrand_values = []
    
    for t in t_values:
        m = max(int(np.floor(t * N)), 1)
        mc_values = []
        
        for _ in range(num_MC_probe):
            sample_indices = random.sample(range(N), m)
            X_sub = candidate_x[sample_indices]
            y_sub = candidate_y[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S = 0.0
                
            X_sub_i = np.vstack([X_sub, x_train[i]])
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand_values.append(np.mean(mc_values))
    
    # Compute second differences to estimate smoothness
    if len(integrand_values) >= 3:
        second_diffs = []
        dt = t_values[1] - t_values[0]
        for j in range(len(integrand_values) - 2):
            second_diff = abs(integrand_values[j+2] - 2*integrand_values[j+1] + integrand_values[j]) / (dt**2)
            second_diffs.append(second_diff)
        return max(second_diffs) if second_diffs else 0.0
    else:
        return 0.0


def compute_integral_shapley_trapezoid(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                     utility_func, num_t_samples=50, num_MC=100, 
                                     rounding_method='probabilistic'):
    """
    Compute Shapley value using trapezoidal integration.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data  
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_t_samples: Number of t values to sample
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    total = x_train.shape[0]
    N = total  # Total number of data points
    rng = np.random.default_rng()  # Random number generator for probabilistic rounding

    # Sample t values uniformly in [0,1]
    t_values = np.linspace(0, 1, num_t_samples, endpoint=True)
    
    integrand = []
    for t in t_values:
        # Use the new coalition size computation with chosen rounding method
        m = compute_coalition_size(t, N, method=rounding_method, rng=rng)
        mc_values = []
        
        for _ in range(num_MC):
            if m == 0:
                # Empty coalition
                X_sub = np.empty((0, x_train.shape[1]))
                y_sub = np.empty(0)
            else:
                # Sample m points from all candidates except target point i
                candidate_indices = [j for j in range(total) if j != i]
                sample_indices = random.sample(candidate_indices, m)
                X_sub = x_train[sample_indices]
                y_sub = y_train[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S = 0.0
                
            X_sub_i = np.vstack([X_sub, x_train[i]]) if m > 0 else x_train[i].reshape(1, -1)
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand.append(np.mean(mc_values))
    
    # Trapezoidal integration
    shapley_value = np.trapezoid(integrand, t_values)
    return shapley_value


def compute_integral_shapley_gaussian(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                    utility_func, num_nodes=32, num_MC=100, 
                                    rounding_method='probabilistic'):
    """
    Compute Shapley value using Gaussian-Legendre quadrature.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index  
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_nodes: Number of Gaussian quadrature nodes
        num_MC: Monte Carlo samples per node
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    total = x_train.shape[0]
    N = total
    rng = np.random.default_rng()

    def integrand(t):
        """Define integrand function for Gaussian quadrature"""
        t_arr = np.atleast_1d(t)
        out = []
        
        for ti in t_arr:
            m = compute_coalition_size(ti, N, method=rounding_method, rng=rng)
            mc_values = []
            
            for _ in range(num_MC):
                if m == 0:
                    X_sub = np.empty((0, x_train.shape[1]))
                    y_sub = np.empty(0)
                else:
                    candidate_indices = [j for j in range(total) if j != i]
                    sample_indices = random.sample(candidate_indices, m)
                    X_sub = x_train[sample_indices]
                    y_sub = y_train[sample_indices]
                
                try:
                    util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
                except:
                    util_S = 0.0
                    
                X_sub_i = np.vstack([X_sub, x_train[i]]) if m > 0 else x_train[i].reshape(1, -1)
                y_sub_i = np.append(y_sub, y_train[i])
                
                try:
                    util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
                except:
                    util_S_i = 0.0
                    
                mc_values.append(util_S_i - util_S)
            
            out.append(np.mean(mc_values))
        
        return np.array(out)[0] if np.isscalar(t) else np.array(out)

    # Gaussian-Legendre quadrature
    shapley_value, _ = fixed_quad(integrand, 0.0, 1.0, n=num_nodes)
    return shapley_value


def compute_integral_shapley_simpson(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                   utility_func, num_t_samples=50, num_MC=100, 
                                   rounding_method='probabilistic'):
    """
    Compute Shapley value using Simpson's rule for numerical integration.
    Simpson's rule is more accurate than trapezoid for smooth functions,
    especially those with exponential decay or power-law behavior.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        num_t_samples: Number of t samples (must be odd for Simpson's rule)
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    # Ensure odd number of samples for Simpson's rule
    if num_t_samples % 2 == 0:
        num_t_samples += 1
        print(f"Adjusted to {num_t_samples} samples (Simpson's rule requires odd number)")
    
    total = x_train.shape[0]
    N = total - 1  # Exclude target point
    rng = np.random.default_rng()
    
    # Generate t values (excluding endpoints to avoid edge cases)
    t_values = np.linspace(0.001, 0.999, num_t_samples)
    integrand_values = []
    
    print(f"Computing Simpson integral with {num_t_samples} t-samples...")
    
    for t in tqdm(t_values, desc="Simpson integration"):
        mc_values = []
        
        for _ in range(num_MC):
            # Compute coalition size using rounding method
            m = compute_coalition_size(t, N, method=rounding_method, rng=rng)
            
            # Sample coalition S of size m (excluding point i)
            if m == 0:
                X_sub = np.empty((0, x_train.shape[1]))
                y_sub = np.empty(0)
            else:
                candidate_indices = [j for j in range(total) if j != i]
                sample_indices = random.sample(candidate_indices, m)
                X_sub = x_train[sample_indices]
                y_sub = y_train[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S = 0.0
                
            # Coalition S ∪ {i}
            X_sub_i = np.vstack([X_sub, x_train[i]]) if m > 0 else x_train[i].reshape(1, -1)
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand_values.append(np.mean(mc_values))
    
    # Apply Simpson's rule
    from scipy.integrate import simpson
    shapley_value = simpson(integrand_values, t_values)
    
    return shapley_value


def choose_optimal_t_samples(data_size, method='simpson', precision='balanced'):
    """
    Choose optimal number of t sampling points based on data size and precision requirements.
    
    Args:
        data_size: Size of training dataset
        method: 'trapezoid' or 'simpson'
        precision: 'fast', 'balanced', or 'high'
    
    Returns:
        optimal_t_samples: Recommended number of t sampling points
    """
    if method == 'simpson':
        if precision == 'fast':
            base_samples = max(11, min(15, int(np.sqrt(data_size))))
        elif precision == 'balanced':
            base_samples = max(19, min(25, int(np.sqrt(data_size) * 1.5)))
        else:  # high precision
            base_samples = max(25, min(35, int(np.sqrt(data_size) * 2)))
        
        # Ensure odd number for Simpson's rule
        return base_samples if base_samples % 2 == 1 else base_samples + 1
    
    else:  # trapezoid
        if precision == 'fast':
            return max(11, min(17, int(np.sqrt(data_size))))
        elif precision == 'balanced':
            return max(21, min(27, int(np.sqrt(data_size) * 1.5)))
        else:  # high precision
            return max(31, min(41, int(np.sqrt(data_size) * 2)))


def compute_integral_shapley_auto(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                 utility_func, method='simpson', precision='balanced', num_MC=100,
                                 rounding_method='probabilistic'):
    """
    Automatically choose optimal parameters and compute Shapley value.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: 'trapezoid' or 'simpson'
        precision: 'fast', 'balanced', or 'high'
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    # Choose optimal t sampling points
    optimal_t = choose_optimal_t_samples(len(x_train), method, precision)
    
    print(f"Auto-selected {optimal_t} t-samples for {method} method ({precision} precision)")
    
    # Call appropriate integration method
    if method == 'simpson':
        return compute_integral_shapley_simpson(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=optimal_t, num_MC=num_MC, 
            rounding_method=rounding_method
        )
    else:
        return compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=optimal_t, num_MC=num_MC, 
            rounding_method=rounding_method
        )


def compute_integral_shapley_adaptive(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                    utility_func, tolerance=1e-4, max_samples=200, num_MC=100,
                                    rounding_method='probabilistic'):
    """
    Compute Shapley value using adaptive sampling based on smoothness detection.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        tolerance: Convergence tolerance
        max_samples: Maximum number of t samples
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    # Start with a coarse estimate
    coarse_estimate = compute_integral_shapley_trapezoid(
        x_train, y_train, x_valid, y_valid, i, clf, final_model, 
        utility_func, num_t_samples=10, num_MC=num_MC, rounding_method=rounding_method
    )
    
    # Progressively refine
    for num_samples in [20, 50, 100, max_samples]:
        fine_estimate = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=num_samples, num_MC=num_MC, rounding_method=rounding_method
        )
        
        if abs(fine_estimate - coarse_estimate) < tolerance:
            return fine_estimate
            
        coarse_estimate = fine_estimate
    
    return coarse_estimate


def stratified_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                            utility_func, num_MC):
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
    
    返回:
      目标数据点的估计 Shapley 值（浮点数）
    """
    total = X_train.shape[0]
    # 从训练集中移除目标数据点 i，构建候选池
    indices = [j for j in range(total) if j != i]
    candidate_x = X_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)  # 候选数据数
    
    # 存储每一层(每个大小)的平均边际贡献
    strata_values = []
    
    # 对每个可能的子集大小进行采样
    for j in range(N+1):  # 从0到N（包括空集和全集）
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
            # 对于大小为j的子集，有comb(N,j)种可能的组合，每个组合的权重是
            # factorial(j) * factorial(N-j-1) / factorial(N)
            weight = 1/total
            strata_values.append(weight * avg_contribution)
    
    # 总的Shapley值是所有层的加权和
    shapley_value = sum(strata_values)
    
    return shapley_value


def stratified_shapley_value_with_plot(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                                      utility_func, num_MC, plot=True, save_path=None):
    """
    Compute Shapley value using stratified sampling and visualize marginal contributions by coalition size.
    
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
        - shapley_value: Estimated Shapley value
        - layer_sizes: Coalition sizes (0 to N)
        - layer_contributions: Expected marginal contributions per layer
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    total = X_train.shape[0]
    # Remove target data point i from training set to build candidate pool
    indices = [j for j in range(total) if j != i]
    candidate_x = X_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)  # Number of candidate data points
    
    # Store layer information
    layer_sizes = []
    layer_contributions = []
    strata_values = []
    
    # Sample over each possible coalition size with progress bar
    for j in tqdm(range(N+1), desc=f"Computing stratified Shapley for point {i}"):
        mc_values = []
        # Monte Carlo sampling for coalition size j
        for _ in range(num_MC):
            # Use empty set if j=0
            if j == 0:
                sample_indices = []
            else:
                sample_indices = random.sample(range(N), j)
            
            X_sub = candidate_x[sample_indices] if j > 0 else np.empty((0, X_train.shape[1]))
            y_sub = candidate_y[sample_indices] if j > 0 else np.empty(0)
            
            try:
                # Compute utility v(S)
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except Exception:
                # Return 0 if empty subset or computation error
                util_S = 0.0
                
            # Build S ∪ {i}
            X_sub_i = np.concatenate([X_sub, X_train[i].reshape(1, -1)], axis=0)
            y_sub_i = np.concatenate([y_sub, np.array([y_train[i]])], axis=0)
            
            try:
                # Compute utility v(S ∪ {i})
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except Exception:
                util_S_i = 0.0
                
            # Compute marginal contribution
            mc_values.append(util_S_i - util_S)
        
        # Compute average marginal contribution for this coalition size
        if mc_values:
            avg_contribution = np.mean(mc_values)
            weight = 1/total
            
            layer_sizes.append(j)
            layer_contributions.append(avg_contribution)
            strata_values.append(weight * avg_contribution)
    
    # Total Shapley value is weighted sum of all layers
    shapley_value = sum(strata_values)
    
    # Generate plot
    if plot:
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
        N = len(layer_sizes) - 1  # Max coalition size
        normalized_x = [s/N for s in layer_sizes]  # Scale to [0,1]
        
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
        ax2.text(0.02, max(layer_contributions)*0.8, 
                f'Shapley Value = ∫₀¹ E[Δ(t,i)] dt\n= Area under curve\n= {shapley_value:.6f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    return shapley_value, layer_sizes, layer_contributions


def cc_shapley_nested_trapz(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    clf,
    final_model,
    utility_func,
    *,
    num_t_samples: int = 100,   # t-grid 分辨率 (M)
    num_MC: int = 100,          # 每个 bin 的 MC 次数
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Complementary-Contribution Shapley (Riemann-trapz 近似)。
    在 t ∈ (0,1) 均匀取 M 个中点，每个 bin 做 num_MC 次采样，
    一次采样同时为所有玩家记录正反两侧的互补贡献。
    计算量 O(M · num_MC · 训练开销)。
    """
    if rng is None:
        rng = np.random.default_rng()

    n  = x_train.shape[0]
    M  = num_t_samples
    indices = np.arange(n)

    # -------- 1. t-grid：均匀区间中点 --------
    # t_grid = (np.arange(M) + 0.5) / M            # shape = (M,)
    t_grid = np.linspace(0, 1, num_t_samples, endpoint=True)[1:]  # shape = (M,)

    cc_sum = np.zeros((n, M + 1))
    cc_cnt = np.zeros((n, M + 1 ), dtype=int)

    # -------- 2. 对每个 t-bin 做 MC --------
    for idx_t, t in enumerate(t_grid):
        # j = max(1, min(n, int(round(t * n))))    # 子集大小 ∈ [1, n]
        j = int(np.floor(t * n))  # 子集大小 ∈ [0, n]
        print(f"t={t:.2f}, j={j}", flush=True)
        for _ in range(num_MC):
            if j > 0:
                S_idx   = rng.choice(indices, size=j, replace=False)
            else:
                S_idx   = np.array([], dtype=int)
            comp_idx = np.setdiff1d(indices, S_idx, assume_unique=True)

            # ---- 计算互补贡献 u = U(S) − U(N\S) ----
            clf_s = clone(clf)
            clf_c = clone(clf)

            try:
                u_s = utility_func(x_train[S_idx],  y_train[S_idx],
                                       x_valid, y_valid, clf_s, final_model)
            except:
                u_s = 0.0

            try:
                u_c = utility_func(x_train[comp_idx], y_train[comp_idx],
                                       x_valid, y_valid, clf_c, final_model)
            except:
                u_c = 0.0

            u = u_s - u_c

            # ---- 3. 同时更新两侧玩家 ----
            #  (a) S 内玩家 → bin idx_t
            cc_sum[S_idx,   idx_t] +=  u
            cc_cnt[S_idx,   idx_t] +=  1

            #  (b) 补集玩家 → bin idx_tt 对应 t' = (n-j)/n
            idx_tt  = n - idx_t - 1

            cc_sum[comp_idx, idx_tt] += -u
            cc_cnt[comp_idx, idx_tt] +=  1

    # -------- 4. 各 bin 求均值 → Riemann-trapz 积分 --------
    cc_mean = np.full_like(cc_sum, np.nan, dtype=float)
    mask    = cc_cnt > 0
    cc_mean[mask] = cc_sum[mask] / cc_cnt[mask]

    print(f"cc_mean.shape: {cc_mean.shape}", flush=True)
    print(t_grid.shape, flush=True)
    # --- 梯形积分求 Shapley ---        # M 个中点
    sv = np.nanmean(cc_mean[:, 1:], axis=1)   # (n,)
    return sv


def cc_shapley(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    clf,
    final_model,
    utility_func,
    num_MC: int = 100,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Stratified MC (|S|=j) + 同步记录所有玩家的 CC，符合原文公式.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = x_train.shape[0]
    indices = np.arange(n)

    # 累加器：cc_sum[i,j] 记录第 j 层 CC 总和，cc_cnt[i,j] 记录样本数
    cc_sum  = np.zeros((n, n + 1))   # j 从 1..n，用 j 作列索引
    cc_cnt  = np.zeros((n, n + 1), dtype=int)

    for j in tqdm(range(1, n + 1)):# 层 j
        for _ in range(num_MC):
            S_idx = rng.choice(indices, size=j, replace=False)
            comp_idx = np.setdiff1d(indices, S_idx, assume_unique=True)

            # 计算 u = U(S) - U(N \ S)
            clf_s = clone(clf)
            clf_c = clone(clf)

            try:
                u_s = utility_func(x_train[S_idx],  y_train[S_idx],
                                       x_valid, y_valid, clf_s, final_model)
            except:
                u_s = 0.0
            
            try:
                u_c = utility_func(x_train[comp_idx], y_train[comp_idx],
                                       x_valid, y_valid, clf_c, final_model)
            except:
                u_c = 0.0
            

            u = u_s - u_c                      # CC_N(S)

            # ---- 把这一条样本同时写进两层 --------------------------
            # 对 S 中玩家：层 j 贡献 +u
            cc_sum[S_idx,  j] += u
            cc_cnt[S_idx,  j] += 1

            # 对补集玩家：层 n-j 贡献 -u (因为 CC_N(T)= -u)
            jj = n - j
            if jj > 0:  # when j = n, comp_idx is empty

                cc_sum[comp_idx, jj] += -u
                cc_cnt[comp_idx, jj] += 1
            # ------------------------------------------------------

    cc_mean = np.full_like(cc_sum, np.nan, dtype=float)

    # 对于有样本的地方，计算平均值
    mask = cc_cnt > 0
    cc_mean[mask] = cc_sum[mask] / cc_cnt[mask]

    # 跳过 j=0 列，用 nanmean 自动忽略 cc_cnt==0 的位置
    sv = np.nanmean(cc_mean[:, 1:], axis=1)



    # Shapley_i  = 1/n ∑_{j=1}^n  \overline{SV}_{i,j}
    # sv = np.nanmean(cc_mean, axis=1)    # 跳过 j=0 列
    return sv 


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


def compute_integral_shapley_value(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                 utility_func, method='trapezoid', rounding_method='probabilistic', **kwargs):
    """
    Main interface for computing integral Shapley values.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: Integration method ('trapezoid', 'simpson', 'gaussian', 'adaptive', 'monte_carlo', 'stratified')
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        **kwargs: Method-specific parameters
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    if method == 'trapezoid':
        return compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func, 
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'gaussian':
        return compute_integral_shapley_gaussian(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'simpson':
        return compute_integral_shapley_simpson(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'adaptive':
        return compute_integral_shapley_adaptive(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'monte_carlo':
        return monte_carlo_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'stratified':
        return stratified_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_shapley_values(x_train, y_train, x_valid, y_valid, clf, final_model,
                              utility_func, method='cc', **kwargs):
    """
    Compute Shapley values for all data points using methods that calculate all values simultaneously.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: Method ('cc', 'cc_trapz')
        **kwargs: Method-specific parameters
        
    Returns:
        shapley_values: Array of Shapley values for all data points
    """
    if method == 'cc':
        return cc_shapley(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'cc_trapz':
        return cc_shapley_nested_trapz(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    else:
        raise ValueError(f"Unknown all-points method: {method}")


def compute_shapley_for_params(args):
    """Wrapper function for parallel computation of Shapley values."""
    index, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, method, rounding_method, kwargs = args
    try:
        value = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, index, clf, final_model, 
            utility_func, method=method, rounding_method=rounding_method, **kwargs
        )
        return index, value
    except Exception as e:
        print(f"Error computing Shapley value for index {index}: {str(e)}")
        return index, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Compute Shapley values using integral formulation")
    parser.add_argument("--dataset", type=str, choices=["iris", "wine", "cancer", "synthetic"], 
                       default="iris", help="Dataset to use")
    parser.add_argument("--utility", type=str, choices=["rkhs", "kl", "acc", "cosine"], 
                       default="acc", help="Utility function")
    parser.add_argument("--method", type=str, choices=["trapezoid", "simpson", "gaussian", "adaptive", "monte_carlo", "exact", "stratified", "cc", "cc_trapz"],
                       default="trapezoid", help="Integration method")
    parser.add_argument("--clf", choices=["svm", "lr"], default="svm", help="Base classifier")
    parser.add_argument("--num_t_samples", type=int, default=50, help="Number of t samples for integration")
    parser.add_argument("--num_MC", type=int, default=100, help="Monte Carlo samples per t value")
    parser.add_argument("--num_nodes", type=int, default=32, help="Gaussian quadrature nodes")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Convergence tolerance for adaptive method")
    parser.add_argument("--processes", type=int, default=mp.cpu_count(), help="Number of processes")
    parser.add_argument("--single_point", type=int, default=None, help="Compute for single data point (default: all points)")
    parser.add_argument("--rounding_method", type=str, choices=["probabilistic", "round", "floor", "ceil"],
                       default="probabilistic", help="Method for rounding coalition sizes")
    
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    elif args.dataset == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    elif args.dataset == 'cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

    # Train final model
    if args.clf == "svm":
        final_model = return_model('LinearSVC')
        final_model.fit(x_train, y_train)
        clf = return_model('LinearSVC')
    elif args.clf == "lr":
        final_model = return_model('logistic')
        final_model.fit(x_train, y_train)
        clf = return_model('logistic')

    # Select utility function
    utility_funcs = {
        "rkhs": utility_RKHS,
        "acc": utility_acc,
        "kl": utility_KL,
        "cosine": utility_cosine
    }
    utility_func = utility_funcs[args.utility]

    # Set method parameters
    method_kwargs = {}
    if args.method == 'trapezoid':
        method_kwargs = {'num_t_samples': args.num_t_samples, 'num_MC': args.num_MC}
    elif args.method == 'gaussian':
        method_kwargs = {'num_nodes': args.num_nodes, 'num_MC': args.num_MC}
    elif args.method == 'adaptive':
        method_kwargs = {'tolerance': args.tolerance, 'num_MC': args.num_MC}
    elif args.method == 'monte_carlo':
        method_kwargs = {'num_samples': args.num_MC * args.num_t_samples}
    elif args.method == 'stratified':
        method_kwargs = {'num_MC': args.num_MC}
    elif args.method in ['cc', 'cc_trapz']:
        method_kwargs = {'num_MC': args.num_MC}
        if args.method == 'cc_trapz':
            method_kwargs['num_t_samples'] = args.num_t_samples

    # Compute Shapley values
    if args.single_point is not None:
        target_indices = [args.single_point]
    else:
        target_indices = list(range(len(x_train)))  # Default: compute all points

    if args.method == "exact":
        # Exact computation (sequential)
        print(f"Computing exact Shapley values for {len(target_indices)} data points...")
        results = {}
        for idx in target_indices:
            value = exact_shapley_value(idx, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func)
            results[idx] = value
        results = {"Exact": np.array([results[i] for i in sorted(target_indices)])}
    elif args.method in ['cc', 'cc_trapz']:
        # CC methods compute all points simultaneously
        print(f"Computing {args.method} Shapley values for all {len(x_train)} data points...")
        all_values = compute_all_shapley_values(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, 
            method=args.method, **method_kwargs
        )
        
        if args.single_point is not None:
            selected_values = all_values[target_indices]
        else:
            selected_values = all_values  # All points
            
        results = {args.method.upper(): selected_values}
    else:
        # Other integral methods (parallel)
        process_args = []
        for idx in target_indices:
            process_args.append((idx, x_train, y_train, x_valid, y_valid, clf, final_model, 
                               utility_func, args.method, args.rounding_method, method_kwargs))
        
        print(f"Using {args.processes} processes to compute {len(target_indices)} Shapley values with {args.method} method...")
        
        raw_results = {}
        with mp.Pool(processes=args.processes) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_shapley_for_params, process_args),
                                   total=len(process_args), desc=f"Computing {args.method} Shapley values"):
                raw_results[index] = value
        
        results = {args.method.title(): np.array([raw_results[i] for i in sorted(target_indices)])}

    # Print results
    print(f"\nDataset: {args.dataset}")
    print(f"Utility function: {args.utility}")
    print(f"Method: {args.method}")
    
    for method_name, values in results.items():
        valid_values = values[~np.isnan(values.astype(float)) if values.dtype != object else np.ones(len(values), dtype=bool)]
        if len(valid_values) > 0:
            print(f"\n{method_name} Results:")
            print(f"  Mean: {np.mean(valid_values.astype(float)):.6f}")
            print(f"  Max:  {np.max(valid_values.astype(float)):.6f}")
            print(f"  Min:  {np.min(valid_values.astype(float)):.6f}")

    # Save results
    # Different filename formats for different methods
    if args.method in ['trapezoid', 'gaussian', 'adaptive']:
        # Methods that use t_samples
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}_t{args.num_t_samples}_mc{args.num_MC}.pkl"
    elif args.method in ['stratified', 'monte_carlo']:
        # Methods that only use MC samples
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}_mc{args.num_MC}.pkl"
    else:
        # Fallback for other methods
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}.pkl"
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {pkl_filename}")


if __name__ == "__main__":
    main()