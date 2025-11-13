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
from scipy.interpolate import PchipInterpolator, UnivariateSpline
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



def compute_marginal_contribution_at_t(t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                     utility_func, num_MC=50, rounding_method='probabilistic', rng=None):
    """
    Compute the marginal contribution E[Δ(t,i)] at a specific t value.
    
    Args:
        t: Coalition proportion in [0,1]
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_MC: Monte Carlo samples for expectation
        rounding_method: How to round coalition sizes
        rng: Random number generator
        
    Returns:
        marginal_contribution: E[Δ(t,i)] at this t value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    total = x_train.shape[0]
    N = total
    mc_values = []
    
    for _ in range(num_MC):
        # Compute coalition size using specified rounding method
        m = compute_coalition_size(t, N, method=rounding_method, rng=rng)
        
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
    
    return np.mean(mc_values)


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
    rng = np.random.default_rng()  # Random number generator for probabilistic rounding

    # Sample t values uniformly in [0,1]
    t_values = np.linspace(0, 1, num_t_samples, endpoint=True)
    
    integrand = []
    for t in t_values:
        marginal_contrib = compute_marginal_contribution_at_t(
            t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC, rounding_method, rng
        )
        integrand.append(marginal_contrib)
    
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
    rng = np.random.default_rng()

    def integrand(t):
        """Define integrand function for Gaussian quadrature"""
        t_arr = np.atleast_1d(t)
        out = []
        
        for ti in t_arr:
            marginal_contrib = compute_marginal_contribution_at_t(
                ti, x_train, y_train, x_valid, y_valid, i, clf, final_model,
                utility_func, num_MC, rounding_method, rng
            )
            out.append(marginal_contrib)
        
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
    
    rng = np.random.default_rng()
    
    # Generate t values using complete interval [0,1] for theoretical accuracy
    t_values = np.linspace(0, 1, num_t_samples, endpoint=True)
    integrand_values = []
    
    print(f"Computing Simpson integral with {num_t_samples} t-samples...")
    
    for t in tqdm(t_values, desc="Simpson integration"):
        marginal_contrib = compute_marginal_contribution_at_t(
            t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC, rounding_method, rng
        )
        integrand_values.append(marginal_contrib)
    
    # Apply Simpson's rule
    from scipy.integrate import simpson
    shapley_value = simpson(integrand_values, t_values)
    
    return shapley_value


def choose_optimal_t_samples(data_size, method='simpson', precision='balanced'):
    """
    Choose optimal number of t sampling points based on data size and precision requirements.
    
    Args:
        data_size: Size of training dataset
        method: 'trapezoid', 'simpson', or 'smart_adaptive'
        precision: 'fast', 'balanced', or 'high'
    
    Returns:
        optimal_t_samples: Recommended number of t sampling points or dict of parameters
    """
    if method == 'smart_adaptive':
        # For smart adaptive, return parameters as tolerance instead of fixed samples
        if precision == 'fast':
            return {'tolerance': 1e-3, 'max_depth': 3}
        elif precision == 'balanced':
            return {'tolerance': 1e-4, 'max_depth': 4}
        else:  # high precision
            return {'tolerance': 1e-5, 'max_depth': 5}
    
    elif method == 'simpson':
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


def compute_integral_shapley_smart_adaptive(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                           utility_func, tolerance=1e-6, max_depth=10, num_MC=100,
                                           rounding_method='probabilistic', min_samples_per_interval=3, 
                                           return_sampling_info=False):
    """
    Compute Shapley value using intelligent local adaptive sampling.
    
    This method:
    1. Explores function smoothness across the [0,1] interval
    2. Recursively subdivides regions with high variation
    3. Adaptively allocates sampling density based on local behavior
    4. Uses Richardson extrapolation for error estimation
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        tolerance: Local smoothness tolerance for subdivision
        max_depth: Maximum recursion depth for interval subdivision
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes
        min_samples_per_interval: Minimum sampling points per interval
        return_sampling_info: Whether to return detailed sampling information
        
    Returns:
        If return_sampling_info is False:
            tuple: (shapley_value, actual_budget)
        If return_sampling_info is True:
            tuple: (shapley_value, actual_budget, sampling_info)
    """
    rng = np.random.default_rng()
    
    def compute_interval_integrand(t_values, num_mc_local=None):
        """Compute integrand values at given t points"""
        if num_mc_local is None:
            num_mc_local = num_MC
            
        integrand_values = []
        for t in t_values:
            value = compute_marginal_contribution_at_t(
                t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
                utility_func, num_MC=num_mc_local, rounding_method=rounding_method, rng=rng
            )
            integrand_values.append(value)
        return np.array(integrand_values)
    
    def estimate_local_variation(a, b, num_probe=7):
        """Estimate function variation in interval [a,b] with improved sensitivity"""
        if b - a < 1e-6:  # Too small interval
            return 0.0
            
        t_probe = np.linspace(a, b, num_probe)
        integrand_probe = compute_interval_integrand(t_probe, num_mc_local=max(20, num_MC//2))
        
        # Compute total variation and second differences
        if len(integrand_probe) >= 3:
            # Calculate function range (max - min)
            func_range = np.max(integrand_probe) - np.min(integrand_probe)
            
            # Second difference as smoothness indicator
            second_diffs = []
            dt = (b - a) / (num_probe - 1)
            for j in range(len(integrand_probe) - 2):
                second_diff = abs(integrand_probe[j+2] - 2*integrand_probe[j+1] + integrand_probe[j]) / (dt**2)
                second_diffs.append(second_diff)
            max_second_diff = max(second_diffs) if second_diffs else 0.0
            
            # First difference for variation
            first_diffs = np.abs(np.diff(integrand_probe))
            total_variation = np.sum(first_diffs) / (b - a)
            
            # Standard deviation as another indicator
            std_dev = np.std(integrand_probe)
            
            # Combined metric: emphasize range and variation
            return func_range * 10 + max_second_diff + total_variation + std_dev * 5
        return 0.0
    
    def adaptive_subdivide(a, b, depth=0):
        """Recursively subdivide interval based on local variation"""
        if depth >= max_depth or b - a < 0.01:  # Stop criteria
            return [(a, b)]
        
        # Estimate variation in current interval
        variation = estimate_local_variation(a, b)
        
        if variation > tolerance:
            # High variation - subdivide
            mid = (a + b) / 2
            left_intervals = adaptive_subdivide(a, mid, depth + 1)
            right_intervals = adaptive_subdivide(mid, b, depth + 1)
            return left_intervals + right_intervals
        else:
            # Low variation - keep as single interval
            return [(a, b)]
    
    print(f"Starting simple smart adaptive sampling for data point {i}...")
    
    # Simple approach: Fixed uniform intervals + adaptive sampling
    num_intervals = 20  # Fixed number of intervals
    intervals = [(i/num_intervals, (i+1)/num_intervals) for i in range(num_intervals)]
    
    print(f"Using {num_intervals} fixed uniform intervals")
    
    # Store sampling information for visualization
    sampling_info = {
        'intervals': intervals,
        'interval_info': [],
        'all_t_values': [],
        'all_integrand_values': [],
        'interval_contributions': []
    }
    
    # Single pass: estimate variation and allocate samples directly
    print("  Computing variation and allocating samples...")
    interval_variations = []
    for a, b in intervals:
        local_variation = estimate_local_variation(a, b, num_probe=5)
        interval_variations.append(local_variation)
    
    # Calculate variation statistics for adaptive thresholds
    sorted_variations = sorted(interval_variations, reverse=True)
    max_variation = max(interval_variations) if interval_variations else 1.0
    mean_variation = np.mean(interval_variations) if interval_variations else 0.0
    
    # Use absolute thresholds based on function behavior
    high_threshold = max(0.01, mean_variation * 2)     # Clearly significant variation
    medium_threshold = max(0.005, mean_variation * 0.5) # Moderate variation
    low_threshold = max(0.001, mean_variation * 0.1)    # Minor variation
    # Below low_threshold: essentially flat, use minimal sampling
    
    print(f"  Variation thresholds: high={high_threshold:.2e}, medium={medium_threshold:.2e}, low={low_threshold:.2e}")
    
    # Adaptive sampling based on variation only
    total_integral = 0.0
    total_samples_used = 0
    
    for idx, (a, b) in enumerate(intervals):
        interval_length = b - a
        local_variation = interval_variations[idx]
        
        # Aggressive direct mapping: variation level -> sample count
        if local_variation >= high_threshold:         # Clearly significant variation
            base_samples = 15  # High sampling
        elif local_variation >= medium_threshold:     # Moderate variation
            base_samples = 7   # Medium sampling
        elif local_variation >= low_threshold:        # Minor variation
            base_samples = 3   # Low sampling
        else:                                         # Essentially flat
            base_samples = 2   # Minimal sampling (just endpoints)
        
        # Ensure odd number for better Simpson integration
        if base_samples % 2 == 0:
            base_samples += 1
        
        # Generate sampling points in current interval
        t_values = np.linspace(a, b, base_samples)
        integrand_values = compute_interval_integrand(t_values)
        
        # Compute integral for this interval using Simpson's rule
        from scipy.integrate import simpson
        interval_integral = simpson(integrand_values, t_values)
        total_integral += interval_integral
        total_samples_used += base_samples
        
        # Store sampling information
        if return_sampling_info:
            # Determine which category this interval falls into
            if local_variation >= high_threshold:
                category = "High"
            elif local_variation >= medium_threshold:
                category = "Medium"
            elif local_variation >= low_threshold:
                category = "Low"
            else:
                category = "Minimal"
                
            sampling_info['interval_info'].append({
                'interval': (a, b),
                'length': interval_length,
                'samples': base_samples,
                'integral': interval_integral,
                'variation': local_variation,
                'category': category,
                't_values': t_values,
                'integrand_values': integrand_values
            })
            sampling_info['all_t_values'].extend(t_values)
            sampling_info['all_integrand_values'].extend(integrand_values)
            sampling_info['interval_contributions'].append(interval_integral)
        
        if idx < 5:  # Print details for first few intervals
            category = "High" if local_variation >= high_threshold else "Medium" if local_variation >= medium_threshold else "Low" if local_variation >= low_threshold else "Minimal"
            print(f"  Interval [{a:.3f}, {b:.3f}]: variation={local_variation:.2e} ({category}), samples={base_samples}, integral={interval_integral:.6f}")
    
    print(f"Smart adaptive sampling completed: {total_samples_used} total samples across {len(intervals)} intervals")
    # Return both the shapley value and the actual budget used
    actual_budget = total_samples_used * num_MC
    
    if return_sampling_info:
        return total_integral, actual_budget, sampling_info
    else:
        return total_integral, actual_budget


def visualize_smart_adaptive_sampling(sampling_info, data_point_index, save_path=None):
    """
    Visualize the adaptive sampling pattern used by Smart Adaptive method.
    
    Args:
        sampling_info: Dictionary containing sampling information from smart_adaptive method
        data_point_index: Index of the data point being analyzed
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Simple Smart Adaptive Sampling Pattern
    ax1.set_title(f'Simple Smart Adaptive Sampling for Data Point {data_point_index}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t (Coalition Proportion)')
    ax1.set_ylabel('Sampling Points')
    
    # Use categories to determine colors - clearer visualization
    category_colors = {
        'High': '#FF6B6B',      # Red - High variation
        'Medium': '#4ECDC4',    # Teal - Medium variation  
        'Low': '#45B7D1',       # Blue - Low variation
        'Minimal': '#96CEB4'    # Green - Minimal variation
    }
    
    for idx, interval_info in enumerate(sampling_info['interval_info']):
        a, b = interval_info['interval']
        samples = interval_info['samples']
        length = interval_info['length']
        variation = interval_info.get('variation', 0)
        category = interval_info.get('category', 'Minimal')
        
        # Height represents number of sampling points directly
        height = samples
        
        # Draw rectangle representing interval
        rect = patches.Rectangle((a, 0), length, height, 
                               linewidth=1, edgecolor='black', 
                               facecolor=category_colors.get(category, '#96CEB4'), alpha=0.8)
        ax1.add_patch(rect)
        
        # Add text showing number of samples and category
        ax1.text(a + length/2, height/2, f'{samples}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(a + length/2, height*0.8, f'{category}', 
                ha='center', va='center', fontsize=8, style='italic')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max([info['samples'] for info in sampling_info['interval_info']]) * 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for categories
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=category) 
                      for category, color in category_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Plot 2: Sampling points and integrand values
    ax2.set_title('Integrand Function and Sampling Points', fontsize=12)
    ax2.set_xlabel('t (Coalition Proportion)')
    ax2.set_ylabel('Integrand Value E[Δ(t,i)]')
    
    # Plot integrand values for each interval
    for idx, interval_info in enumerate(sampling_info['interval_info']):
        t_vals = interval_info['t_values']
        integrand_vals = interval_info['integrand_values']
        category = interval_info.get('category', 'Minimal')
        color = category_colors.get(category, '#96CEB4')
        
        # Plot line for this interval
        ax2.plot(t_vals, integrand_vals, 'o-', color=color, 
                linewidth=2, markersize=4, alpha=0.8)
        
        # Highlight sampling points
        ax2.scatter(t_vals, integrand_vals, color=color, 
                   s=30, zorder=5, alpha=0.9)
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Interval contributions to total integral
    ax3.set_title('Interval Contributions to Total Shapley Value', fontsize=12)
    ax3.set_xlabel('Interval Index')
    ax3.set_ylabel('Interval Contribution')
    
    interval_contributions = sampling_info['interval_contributions']
    interval_indices = range(len(interval_contributions))
    
    # Color bars by category
    bar_colors = [category_colors.get(info.get('category', 'Minimal'), '#96CEB4') 
                  for info in sampling_info['interval_info']]
    bars = ax3.bar(interval_indices, interval_contributions, 
                  color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, contrib) in enumerate(zip(bars, interval_contributions)):
        if abs(contrib) > max(abs(min(interval_contributions)), max(interval_contributions)) * 0.1:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                    (0.02 if contrib >= 0 else -0.02) * max(interval_contributions),
                    f'{contrib:.4f}', ha='center', va='bottom' if contrib >= 0 else 'top',
                    fontsize=8)
    
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Add summary statistics
    total_intervals = len(sampling_info['intervals'])
    total_samples = sum([info['samples'] for info in sampling_info['interval_info']])
    total_shapley = sum(interval_contributions)
    
    fig.suptitle(f'Smart Adaptive Sampling Visualization\n'
                f'Total Intervals: {total_intervals}, Total Sampling Points: {total_samples}, '
                f'Shapley Value: {total_shapley:.6f}', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Smart Adaptive sampling visualization saved to: {save_path}")
    
    plt.show()




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
                f'Shapley Value = ∫₀¹ E[Δ(t,i)] dt\\n= Area under curve\\n= {shapley_value:.6f}',
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
    return sv 


def _cc_layer_worker(args):
    """
    Worker function for parallel CC layer processing.
    Each process handles one coalition size (layer) with all its MC samples.
    
    Args:
        args: Tuple containing (j, num_MC, seed, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func)
    
    Returns:
        Tuple: (j, list_of_contributions)
        where list_of_contributions contains (S_idx, comp_idx, cc_value) tuples
    """
    j, num_MC, seed, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func = args
    
    # Create independent random state for this layer
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    indices = np.arange(n)
    
    layer_contributions = []
    
    # Process all MC samples for this coalition size j
    for _ in range(num_MC):
        try:
            # Sample coalition S of size j
            S_idx = rng.choice(indices, size=j, replace=False)
            comp_idx = np.setdiff1d(indices, S_idx, assume_unique=True)
            
            # Calculate CC_N(S) = U(S) - U(N\S)
            clf_s = clone(clf)
            clf_c = clone(clf)
            
            try:
                u_s = utility_func(x_train[S_idx], y_train[S_idx], 
                                  x_valid, y_valid, clf_s, final_model)
            except:
                u_s = 0.0
                
            try:
                u_c = utility_func(x_train[comp_idx], y_train[comp_idx], 
                                  x_valid, y_valid, clf_c, final_model)
            except:
                u_c = 0.0
                
            cc_value = u_s - u_c
            layer_contributions.append((S_idx, comp_idx, cc_value))
            
        except Exception:
            # Add zero contribution if sampling fails
            layer_contributions.append((np.array([]), np.array([]), 0.0))
    
    return j, layer_contributions


def cc_shapley_parallel(x_train: np.ndarray, y_train: np.ndarray, 
                       x_valid: np.ndarray, y_valid: np.ndarray,
                       clf, final_model, utility_func,
                       num_MC: int = 100,
                       num_processes: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Parallel CC Shapley using sampling-level parallelization.
    
    This method:
    1. For each layer j (coalition size), performs num_MC sampling in parallel
    2. Uses complementary contribution: CC_N(S) = U(S) - U(N\\S)
    3. Implements the dual assignment: S players get +cc_value, complement gets -cc_value
    4. Averages across all layers to get final Shapley values
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_MC: Monte Carlo samples per coalition size
        num_processes: Number of parallel processes (default: all CPU cores)
        rng: Random number generator (for reproducibility)
        
    Returns:
        shapley_values: Array of Shapley values for all data points
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
        
    if rng is None:
        rng = np.random.default_rng()
    
    n = x_train.shape[0]
    
    print(f"CC Layer-wise: {n} layers, {num_MC} MC/layer, {num_processes} processes")
    print(f"Total tasks: {n} layers (one per process)")
    
    # Generate layer tasks - one task per coalition size
    layer_tasks = []
    for j in range(1, n + 1):  # Coalition sizes from 1 to n
        # Create unique seed for each layer
        layer_seed = rng.integers(0, 2**31) ^ hash(j) & 0x7FFFFFFF
        task_args = (j, num_MC, layer_seed, x_train, y_train, x_valid, y_valid, 
                    clf, final_model, utility_func)
        layer_tasks.append(task_args)
    
    # Execute layer tasks in parallel
    print(f"Starting parallel CC layer processing...")
    with mp.Pool(processes=num_processes) as pool:
        layer_results = list(tqdm(
            pool.imap_unordered(_cc_layer_worker, layer_tasks),
            total=len(layer_tasks), 
            desc="CC layer processing"
        ))
    
    # Organize results by coalition size (layer j)
    print("Processing results and computing averages...")
    layer_contributions = {}
    for j, contributions in layer_results:
        layer_contributions[j] = contributions
    
    # Build the CC sum and count matrices
    cc_sum = np.zeros((n, n + 1))   # [player, layer]
    cc_cnt = np.zeros((n, n + 1), dtype=int)
    
    for j in range(1, n + 1):
        if j not in layer_contributions:
            continue
            
        for S_idx, comp_idx, cc_value in layer_contributions[j]:
            # Players in S: layer j gets +cc_value
            if len(S_idx) > 0:
                cc_sum[S_idx, j] += cc_value
                cc_cnt[S_idx, j] += 1
            
            # Players in complement: layer (n-j) gets -cc_value
            jj = n - j
            if jj > 0 and len(comp_idx) > 0:
                cc_sum[comp_idx, jj] += -cc_value
                cc_cnt[comp_idx, jj] += 1
    
    # Calculate means and final Shapley values
    cc_mean = np.full_like(cc_sum, np.nan, dtype=float)
    mask = cc_cnt > 0
    cc_mean[mask] = cc_sum[mask] / cc_cnt[mask]
    
    # Skip j=0 column, use nanmean to ignore positions with cc_cnt==0
    sv = np.nanmean(cc_mean[:, 1:], axis=1)
    
    print(f"CC Parallel completed. Shapley values computed for {n} data points.")
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
        method: Integration method ('trapezoid', 'simpson', 'gaussian', 'adaptive', 'smart_adaptive', 'monte_carlo', 'stratified', 'importance_sampling')
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
        # Redirect 'adaptive' to 'smart_adaptive' for consistency
        print("Note: 'adaptive' method redirects to 'smart_adaptive'")
        result = compute_integral_shapley_smart_adaptive(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
        return result[0] if isinstance(result, tuple) else result
    elif method == 'smart_adaptive':
        # For backward compatibility, only return shapley value
        result = compute_integral_shapley_smart_adaptive(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
        return result[0] if isinstance(result, tuple) else result
    elif method == 'monte_carlo':
        return monte_carlo_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'stratified':
        return stratified_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    
    elif method == 'sparse_residual':
        return compute_integral_shapley_sparse_residual(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_integral_shapley_value_with_budget(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                             utility_func, method='trapezoid', rounding_method='probabilistic', 
                                             return_sampling_info=False, **kwargs):
    """
    Main interface for computing integral Shapley values with actual budget information.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: Integration method ('trapezoid', 'simpson', 'gaussian', 'adaptive', 'smart_adaptive', 'monte_carlo', 'stratified')
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        return_sampling_info: Whether to return sampling information (only for smart_adaptive)
        **kwargs: Method-specific parameters
        
    Returns:
        tuple: (shapley_value, actual_budget) for smart_adaptive method, (shapley_value, estimated_budget) for others
        If return_sampling_info=True and method='smart_adaptive': (shapley_value, actual_budget, sampling_info)
    """
    if method == 'smart_adaptive':
        # Smart adaptive returns both shapley value and actual budget
        result = compute_integral_shapley_smart_adaptive(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, return_sampling_info=return_sampling_info, **kwargs
        )
        return result
    else:
        # For other methods, compute estimated budget
        shapley_value = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            method=method, rounding_method=rounding_method, **kwargs
        )
        
        # Estimate budget based on method parameters
        if method == 'simpson':
            num_t_samples = kwargs.get('num_t_samples', 21)
            num_MC = kwargs.get('num_MC', 100)
            estimated_budget = num_t_samples * num_MC
        elif method == 'trapezoid':
            num_t_samples = kwargs.get('num_t_samples', 50)
            num_MC = kwargs.get('num_MC', 100)
            estimated_budget = num_t_samples * num_MC
        elif method == 'monte_carlo':
            estimated_budget = kwargs.get('num_samples', 10000)
        elif method == 'stratified':
            n_points = x_train.shape[0]
            num_MC = kwargs.get('num_MC', 100)
            estimated_budget = n_points * num_MC
        else:
            estimated_budget = kwargs.get('num_MC', 100) * kwargs.get('num_t_samples', 50)
        
        if return_sampling_info:
            return shapley_value, estimated_budget, None  # No sampling info for other methods
        else:
            return shapley_value, estimated_budget


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
        method: Method ('cc', 'cc_parallel', 'cc_trapz', 'cc_integral_parallel')
        **kwargs: Method-specific parameters
        
    Returns:
        shapley_values: Array of Shapley values for all data points
    """
    if method == 'cc':
        return cc_shapley(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'cc_parallel':
        return cc_shapley_parallel(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'cc_trapz':
        return cc_shapley_nested_trapz(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    elif method == 'cc_integral_parallel':
        return cc_shapley_integral_parallel(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    else:
        raise ValueError(f"Unknown all-points method: {method}")


def compute_shapley_for_params(args):
    """Wrapper function for parallel computation of Shapley values."""
    index, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, method, rounding_method, kwargs = args
    try:
        result = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, index, clf, final_model, 
            utility_func, method=method, rounding_method=rounding_method, **kwargs
        )
        
        # 处理稀疏残差方法的特殊返回格式
        if method == 'sparse_residual' and isinstance(result, tuple):
            value, info_dict = result
            return index, value, info_dict
        else:
            return index, result
    except Exception as e:
        print(f"Error computing Shapley value for index {index}: {str(e)}")
        return index, f"Error: {str(e)}"


def compute_shapley_for_params_with_budget(args):
    """Wrapper function for parallel computation of Shapley values with budget information."""
    index, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, method, rounding_method, kwargs = args
    try:
        value, budget = compute_integral_shapley_value_with_budget(
            x_train, y_train, x_valid, y_valid, index, clf, final_model, 
            utility_func, method=method, rounding_method=rounding_method, **kwargs
        )
        return index, value, budget
    except Exception as e:
        print(f"Error computing Shapley value for index {index}: {str(e)}")
        return index, f"Error: {str(e)}", 0


def main():
    parser = argparse.ArgumentParser(description="Compute Shapley values using integral formulation")
    parser.add_argument("--dataset", type=str, choices=["iris", "wine", "cancer", "synthetic"], 
                       default="iris", help="Dataset to use")
    parser.add_argument("--utility", type=str, choices=["rkhs", "kl", "acc", "cosine"], 
                       default="acc", help="Utility function")
    parser.add_argument("--method", type=str, choices=["trapezoid", "simpson", "gaussian", "adaptive", "smart_adaptive", "monte_carlo", "exact", "stratified", "sparse_residual", "cc", "cc_parallel", "cc_trapz", "cc_integral_parallel"],
                       default="trapezoid", help="Integration method (note: 'adaptive' redirects to 'smart_adaptive')")
    parser.add_argument("--clf", choices=["svm", "lr"], default="svm", help="Base classifier")
    parser.add_argument("--num_t_samples", type=int, default=50, help="Number of t samples for integration")
    parser.add_argument("--num_MC", type=int, default=100, help="Monte Carlo samples per t value")
    parser.add_argument("--num_nodes", type=int, default=32, help="Gaussian quadrature nodes")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Convergence tolerance for adaptive method")
    
    # 稀疏残差方法专用参数
    parser.add_argument("--sparse_nodes", type=int, default=8, help="Number of Chebyshev nodes for sparse residual method")
    parser.add_argument("--residual_samples", type=int, default=100, help="Number of residual correction samples")
    parser.add_argument("--fitting_method", type=str, choices=["pchip", "spline"], default="pchip", 
                       help="Fitting method for sparse residual (pchip or spline)")
    
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
    elif args.method == 'adaptive' or args.method == 'smart_adaptive':
        method_kwargs = {'tolerance': args.tolerance, 'num_MC': args.num_MC}
    elif args.method == 'monte_carlo':
        method_kwargs = {'num_samples': args.num_MC}
    elif args.method == 'stratified':
        method_kwargs = {'num_MC': args.num_MC}
    elif args.method == 'sparse_residual':
        method_kwargs = {
            'num_nodes': args.sparse_nodes,
            'num_MC_per_node': args.num_MC,
            'residual_samples': args.residual_samples,
            'fitting_method': args.fitting_method,
            'return_detailed_info': True  # 总是返回详细信息以便分析
        }
    elif args.method in ['cc', 'cc_parallel', 'cc_trapz', 'cc_integral_parallel']:
        method_kwargs = {'num_MC': args.num_MC}
        if args.method == 'cc_parallel':
            method_kwargs['num_processes'] = args.processes
        elif args.method == 'cc_trapz':
            method_kwargs['num_t_samples'] = args.num_t_samples
        elif args.method == 'cc_integral_parallel':
            method_kwargs['num_t_samples'] = args.num_t_samples
            method_kwargs['num_processes'] = args.processes

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
    elif args.method in ['cc', 'cc_parallel', 'cc_trapz', 'cc_integral_parallel']:
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
        detailed_info = {}  # 存储稀疏残差方法的详细信息
        
        with mp.Pool(processes=args.processes) as pool:
            for result in tqdm(pool.imap_unordered(compute_shapley_for_params, process_args),
                             total=len(process_args), desc=f"Computing {args.method} Shapley values"):
                if len(result) == 3:  # 稀疏残差方法返回 (index, value, info_dict)
                    index, value, info_dict = result
                    raw_results[index] = value
                    detailed_info[index] = info_dict
                else:  # 其他方法返回 (index, value)
                    index, value = result
                    raw_results[index] = value
        
        values_array = np.array([raw_results[i] for i in sorted(target_indices)])
        results = {args.method.title(): values_array}
        
        # 如果是稀疏残差方法，打印详细信息
        if args.method == 'sparse_residual' and detailed_info:
            print(f"\n稀疏残差方法详细信息:")
            for idx in sorted(target_indices):
                if idx in detailed_info:
                    info = detailed_info[idx]
                    print(f"  数据点 {idx}:")
                    print(f"    主积分: {info['main_integral']:.6f}")
                    print(f"    残差校正: {info['residual_correction']:.6f} ± {info['residual_std']:.6f}")
                    print(f"    总评估次数: {info['total_function_evaluations']}")
                    print(f"    拟合方法: {info['fitting_method']}")
                    print(f"    节点数: {info['num_nodes']}")
                    print()

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
    if args.method in ['trapezoid', 'gaussian', 'adaptive', 'smart_adaptive']:
        # Methods that use t_samples or tolerance
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}_t{args.num_t_samples}_mc{args.num_MC}.pkl"
    elif args.method in ['stratified', 'monte_carlo', 'cc', 'cc_parallel']:
        # Methods that only use MC samples
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}_mc{args.num_MC}.pkl"
    else:
        # Fallback for other methods
        pkl_filename = f"results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}.pkl"
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {pkl_filename}")



# ============================================================================
# 稀疏积分 + 无偏残差方法 (Sparse Integration + Unbiased Residual Correction)
# ============================================================================

def generate_chebyshev_nodes_with_endpoints(m_inner, interval=(0, 1)):
    """
    生成包含端点的 Chebyshev 节点
    
    Args:
        m_inner: 内部 Chebyshev 节点数量
        interval: 目标区间，默认 (0, 1)
        
    Returns:
        nodes: 包含端点的 Chebyshev 节点数组，总数 = m_inner + 2
    """
    if m_inner < 0:
        raise ValueError("m_inner must be non-negative")
    
    if m_inner == 0:
        # 只返回端点
        return np.array([interval[0], interval[1]])
    
    # 生成内部 Chebyshev 节点
    inner_nodes = generate_chebyshev_nodes(m_inner, interval)
    
    # 添加端点
    a, b = interval
    all_nodes = np.concatenate(([a], inner_nodes, [b]))
    
    # 去重并排序
    nodes = np.unique(np.clip(all_nodes, a, b))
    return np.sort(nodes)


def generate_chebyshev_nodes(m, interval=(0, 1)):
    """
    生成 [0,1] 区间上的 Chebyshev 节点
    
    Args:
        m: 节点数量
        interval: 目标区间，默认 (0, 1)
        
    Returns:
        nodes: Chebyshev 节点数组
    """
    # 生成标准 [-1, 1] 区间的 Chebyshev 节点
    nodes_std = np.cos(np.pi * (2 * np.arange(m) + 1) / (2 * m))
    
    # 映射到目标区间
    a, b = interval
    nodes = (nodes_std + 1) / 2 * (b - a) + a
    
    return np.sort(nodes)  # 从小到大排序


def fit_smooth_approximation(t_nodes, E_values, method='pchip'):
    """
    在节点上拟合光滑近似函数 h(t)
    
    Args:
        t_nodes: t 节点数组
        E_values: 对应的 E[Δ(t,i)] 估计值
        method: 拟合方法 ('pchip', 'spline')
        
    Returns:
        h_func: 拟合的函数对象
    """
    if method == 'pchip':
        # PCHIP 单调三次样条，适合单调函数
        h_func = PchipInterpolator(t_nodes, E_values, extrapolate=True)
    elif method == 'spline':
        # 三次样条
        h_func = UnivariateSpline(t_nodes, E_values, s=0, ext='extrapolate')
    else:
        raise ValueError(f"Unknown fitting method: {method}")
    
    return h_func


def compute_integral_and_segment_means(h_func, n, interval=(0, 1)):
    """
    一次性计算拟合函数的积分和各段平均值
    
    Args:
        h_func: 拟合的函数对象
        n: 数据集大小
        interval: 积分区间
        
    Returns:
        integral_value: 总积分值
        a_s: 每个分段的平均值数组
    """
    a, b = interval
    
    # 优先使用对象自带的高效积分方法
    if hasattr(h_func, "integrate"):  # PchipInterpolator
        integral_value = h_func.integrate(a, b)
        a_s = np.array([n * h_func.integrate(s/n, (s+1)/n) for s in range(n)])
    elif hasattr(h_func, "integral"):  # UnivariateSpline
        integral_value = h_func.integral(a, b)
        a_s = np.array([n * h_func.integral(s/n, (s+1)/n) for s in range(n)])
    else:  # 兜底使用数值积分
        from scipy.integrate import quad
        integral_value, _ = quad(h_func, a, b)
        a_s = np.array([n * quad(h_func, s/n, (s+1)/n)[0] for s in range(n)])
    
    return integral_value, a_s


def generate_sparse_warmup_sizes(n, num_warmup=32):
    """
    生成稀疏预热的尺寸集合
    
    Args:
        n: 数据集大小
        num_warmup: 预热尺寸数量
        
    Returns:
        warmup_sizes: 预热尺寸数组
    """
    # 端点 + Chebyshev 尺寸 + 等距尺寸
    if n <= num_warmup:
        return np.arange(n)
    
    # 端点
    sizes = [0, n-1]
    
    # Chebyshev 尺寸（映射到 [0, n-1]）
    cheb_t = generate_chebyshev_nodes(min(num_warmup-4, n-2), interval=(0, 1))
    cheb_sizes = np.round(cheb_t * (n-1)).astype(int)
    sizes.extend(cheb_sizes)
    
    # 等距尺寸
    num_uniform = max(num_warmup - len(sizes), 0)
    if num_uniform > 0:
        uniform_sizes = np.linspace(0, n-1, num_uniform, dtype=int)
        sizes.extend(uniform_sizes)
    
    # 去重并排序
    sizes = np.unique(np.array(sizes))
    return sizes[sizes < n]  # 确保所有尺寸都在有效范围内


def estimate_residual_sampling_distribution_sparse(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                                  utility_func, a_s, num_warmup=32, warmup_samples=5, 
                                                  lambda_eps=1e-8, rng=None):
    """
    使用稀疏预热估计残差重要性采样分布
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        a_s: 拟合函数的段平均值
        num_warmup: 预热尺寸数量
        warmup_samples: 每个尺寸的预热样本数
        lambda_eps: 最小概率下界
        rng: 随机数生成器
        
    Returns:
        p_s: 采样概率分布
        warmup_evaluations: 预热阶段的函数评估次数
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    candidate_indices = np.arange(n)[np.arange(n) != i]
    
    # 生成稀疏预热尺寸
    warmup_sizes = generate_sparse_warmup_sizes(n, num_warmup)
    
    # 在预热尺寸上估计方差和偏差
    sigma_warmup = {}
    bias_warmup = {}
    evaluation_count = 0
    
    for s in warmup_sizes:
        delta_values = []
        
        if s == 0:
            # s=0 时，S 为空集
            try:
                u_empty = utility_func(np.array([]).reshape(0, -1), np.array([]), 
                                     x_valid, y_valid, final_model, final_model)
                u_with_i = utility_func(x_train[i:i+1], y_train[i:i+1], 
                                      x_valid, y_valid, final_model, final_model)
                delta_values = [u_with_i - u_empty]
                evaluation_count += 2  # 两次效用函数调用
            except:
                continue  # 跳过失败的计算，不污染分布
                
        elif s == n-1:
            # s=n-1 时，S = N\{i}（唯一子集）
            try:
                X_without_i = x_train[candidate_indices]
                y_without_i = y_train[candidate_indices]
                model_without_i = clone(clf)
                model_without_i.fit(X_without_i, y_without_i)
                u_without_i = utility_func(X_without_i, y_without_i, x_valid, y_valid, 
                                         model_without_i, final_model)
                u_full = utility_func(x_train, y_train, x_valid, y_valid, final_model, final_model)
                delta_values = [u_full - u_without_i]
                evaluation_count += 2  # 两次效用函数调用
            except:
                continue
                
        else:
            # 一般情况：随机采样若干个大小为 s 的子集
            for _ in range(warmup_samples):
                try:
                    subset_indices = rng.choice(candidate_indices, size=s, replace=False)
                    
                    # 训练不包含 i 的子集模型
                    X_subset = x_train[subset_indices]
                    y_subset = y_train[subset_indices]
                    model_subset = clone(clf)
                    model_subset.fit(X_subset, y_subset)
                    u_subset = utility_func(X_subset, y_subset, x_valid, y_valid, 
                                          model_subset, final_model)
                    
                    # 训练包含 i 的子集模型
                    subset_with_i = np.concatenate([subset_indices, [i]])
                    X_subset_with_i = x_train[subset_with_i]
                    y_subset_with_i = y_train[subset_with_i]
                    model_subset_with_i = clone(clf)
                    model_subset_with_i.fit(X_subset_with_i, y_subset_with_i)
                    u_subset_with_i = utility_func(X_subset_with_i, y_subset_with_i, 
                                                 x_valid, y_valid, model_subset_with_i, final_model)
                    
                    delta = u_subset_with_i - u_subset
                    delta_values.append(delta)
                    evaluation_count += 2  # 两次效用函数调用
                except:
                    continue  # 跳过失败的计算
        
        # 计算统计量（只有成功的样本）
        if len(delta_values) > 0:
            E_s_est = np.mean(delta_values)
            sigma_warmup[s] = np.std(delta_values, ddof=1) if len(delta_values) > 1 else 0
            bias_warmup[s] = abs(E_s_est - a_s[s])
        else:
            # 如果没有成功的样本，使用保守估计
            sigma_warmup[s] = 1.0  # 保守的标准差
            bias_warmup[s] = abs(a_s[s])
    
    # 插值到所有尺寸
    sigma_s = np.zeros(n)
    bias_s = np.zeros(n)
    
    warmup_s_list = np.array(list(sigma_warmup.keys()))
    sigma_values = np.array([sigma_warmup[s] for s in warmup_s_list])
    bias_values = np.array([bias_warmup[s] for s in warmup_s_list])
    
    if len(warmup_s_list) > 1:
        # 使用 PCHIP 单调样条插值，更稳定不振荡
        try:
            sigma_interp = PchipInterpolator(warmup_s_list, sigma_values, extrapolate=True)
            bias_interp = PchipInterpolator(warmup_s_list, bias_values, extrapolate=True)
            sigma_s = sigma_interp(np.arange(n))
            bias_s = bias_interp(np.arange(n))
            # 确保插值结果为非负
            sigma_s = np.maximum(sigma_s, 0)
            bias_s = np.maximum(bias_s, 0)
        except Exception:
            # 如果PCHIP失败，使用线性插值作为备份
            sigma_s = np.interp(np.arange(n), warmup_s_list, sigma_values)
            bias_s = np.interp(np.arange(n), warmup_s_list, bias_values)
    elif len(warmup_s_list) == 1:
        # 只有一个点，使用常数
        sigma_s.fill(sigma_values[0])
        bias_s.fill(bias_values[0])
    else:
        # 没有成功的预热，使用均匀分布
        sigma_s.fill(1.0)
        bias_s = np.abs(a_s)
    
    # 计算重要性采样权重
    weights = np.sqrt(sigma_s**2 + bias_s**2) + lambda_eps
    p_s = weights / np.sum(weights)
    
    return p_s, evaluation_count


def compute_residual_correction(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                              utility_func, a_s, p_s, K=100, rng=None):
    """
    计算残差校正项：E[∫₀¹ (f*(t) - h(t)) dt] 的重要性采样估计
    
    优化：预计算端点delta值，避免重复训练
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        a_s: 拟合函数的段平均值数组
        p_s: 采样分布
        K: 残差采样次数
        rng: 随机数生成器
        
    Returns:
        residual_estimate: 残差估计值
        residual_std: 残差估计的标准误差
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    idx_wo_i = np.arange(n) != i
    candidate_indices = np.arange(n)[idx_wo_i]
    
    # ===== 端点预计算 =====
    # s=0 的情况
    try:
        u_empty = utility_func(np.empty((0, x_train.shape[1])), np.empty((0,)), 
                               x_valid, y_valid, final_model, final_model)
        u_with_i = utility_func(x_train[i:i+1], y_train[i:i+1],
                                x_valid, y_valid, final_model, final_model)
        delta_s0 = u_with_i - u_empty
    except Exception:
        delta_s0 = 0.0
    
    # s=n-1 的情况
    try:
        X_wo = x_train[candidate_indices]
        y_wo = y_train[candidate_indices]
        m_wo = clone(clf)
        m_wo.fit(X_wo, y_wo)
        u_wo = utility_func(X_wo, y_wo, x_valid, y_valid, m_wo, final_model)
        u_full = utility_func(x_train, y_train, x_valid, y_valid, final_model, final_model)
        delta_sn1 = u_full - u_wo
    except Exception:
        delta_sn1 = 0.0
    
    # ===== 重要性采样 =====
    residual_samples = []
    successes = 0
    attempts = 0
    max_attempts = K * 3  # 防止无限循环，最多尝试3倍
    
    while successes < K and attempts < max_attempts:
        attempts += 1
        # 按分布 p_s 采样尺寸 s
        s = rng.choice(n, p=p_s)
        
        try:
            if s == 0:
                delta = delta_s0  # 使用预计算值
            elif s == n-1:
                delta = delta_sn1  # 使用预计算值
            else:
                # 一般情况：随机采样大小为 s 的子集
                subset_indices = rng.choice(candidate_indices, size=s, replace=False)
                
                # 训练不包含 i 的子集模型
                X_subset = x_train[subset_indices]
                y_subset = y_train[subset_indices]
                model_subset = clone(clf)
                model_subset.fit(X_subset, y_subset)
                u_subset = utility_func(X_subset, y_subset, x_valid, y_valid, 
                                      model_subset, final_model)
                
                # 训练包含 i 的子集模型
                subset_with_i = np.concatenate([subset_indices, [i]])
                X_subset_with_i = x_train[subset_with_i]
                y_subset_with_i = y_train[subset_with_i]
                model_subset_with_i = clone(clf)
                model_subset_with_i.fit(X_subset_with_i, y_subset_with_i)
                u_subset_with_i = utility_func(X_subset_with_i, y_subset_with_i, 
                                             x_valid, y_valid, model_subset_with_i, final_model)
                
                delta = u_subset_with_i - u_subset
            
            # 计算重要性采样权重
            Y = (delta - a_s[s]) / (n * p_s[s])
            residual_samples.append(Y)
            successes += 1
            
        except Exception:
            # 失败时直接重试，不添加0，保持无偏性
            continue
    
    # 如果成功样本不足，记录警告但仍计算
    if successes < K:
        print(f"警告: 残差采样只成功 {successes}/{K} 次")
    
    if successes == 0:
        # 极端情况：所有采样都失败
        residual_estimate = 0.0
        residual_std = 0.0
    else:
        residual_samples = np.asarray(residual_samples)
        residual_estimate = float(residual_samples.mean())
        residual_std = (float(residual_samples.std(ddof=1) / np.sqrt(successes)) if successes > 1 else 0.0)
    
    return residual_estimate, residual_std


def compute_integral_shapley_sparse_residual(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                           utility_func, num_nodes=8, num_MC_per_node=50, 
                                           residual_samples=100, fitting_method='pchip',
                                           rounding_method='probabilistic', warmup_per_node=5,
                                           lambda_eps=1e-8, return_detailed_info=False, rng=None):
    """
    稀疏积分 + 无偏残差方法计算 Shapley 值（优化版本）
    
    核心思想：
    SV_i = ∫₀¹ h(t) dt + ∫₀¹ [f*(t) - h(t)] dt
           ^主积分(解析)    ^残差(重要性采样校正)
    
    优化：
    - 使用包含端点的 Chebyshev 节点
    - 一次性计算积分和段平均值
    - 稀疏预热估计采样分布，复杂度从 O(n) 降到 O(num_warmup)
    - 修正 off-by-one 错误
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        num_nodes: Chebyshev 节点数量 (m << n)
        num_MC_per_node: 每个节点的 MC 采样次数
        residual_samples: 残差校正的采样次数 K
        fitting_method: 拟合方法 ('pchip', 'spline')
        rounding_method: 联盟大小舍入方法
        warmup_per_node: 预热阶段每个节点的采样数
        lambda_eps: 采样概率下界
        return_detailed_info: 是否返回详细信息
        rng: 随机数生成器
        
    Returns:
        如果 return_detailed_info=False: shapley_value
        如果 return_detailed_info=True: (shapley_value, info_dict)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    
    # ===== 第一阶段：稀疏节点拟合 =====
    
    # 1) 生成包含端点的 Chebyshev 节点
    t_nodes = generate_chebyshev_nodes_with_endpoints(max(num_nodes-2, 0), interval=(0, 1))
    
    # 2) 在每个节点高精度估计 E[Δ(t,i)]
    print(f"稀疏积分: 在 {len(t_nodes)} 个 Chebyshev 节点（含端点）上估计积分函数...")
    E_values = []
    for t in t_nodes:
        marginal_contrib = compute_marginal_contribution_at_t(
            t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC_per_node, rounding_method, rng
        )
        E_values.append(marginal_contrib)
    
    E_values = np.array(E_values)
    
    # 3) 拟合光滑函数 h(t)
    print(f"拟合光滑近似函数 (方法: {fitting_method})...")
    h_func = fit_smooth_approximation(t_nodes, E_values, method=fitting_method)
    
    # 4) 一次性计算主积分和段平均值
    print("计算主积分和段平均值...")
    main_integral, a_s = compute_integral_and_segment_means(h_func, n, interval=(0, 1))
    
    # ===== 第二阶段：残差校正 =====
    
    # 5) 稀疏预热估计最优重要性采样分布
    print("稀疏预热估计残差重要性采样分布...")
    p_s, warmup_evaluations = estimate_residual_sampling_distribution_sparse(
        x_train, y_train, x_valid, y_valid, i, clf, final_model,
        utility_func, a_s, num_warmup=min(32, n), warmup_samples=warmup_per_node, 
        lambda_eps=lambda_eps, rng=rng
    )
    
    # 6) 计算残差校正
    print(f"计算残差校正 (采样 {residual_samples} 次)...")
    residual_correction, residual_std = compute_residual_correction(
        x_train, y_train, x_valid, y_valid, i, clf, final_model,
        utility_func, a_s, p_s, K=residual_samples, rng=rng
    )
    
    # ===== 最终结果 =====
    shapley_value = main_integral + residual_correction
    
    if return_detailed_info:
        info_dict = {
            'main_integral': main_integral,
            'residual_correction': residual_correction,
            'residual_std': residual_std,
            'num_nodes': len(t_nodes),
            'num_MC_per_node': num_MC_per_node,
            'residual_samples': residual_samples,
            't_nodes': t_nodes,
            'E_values': E_values,
            'fitting_method': fitting_method,
            'sampling_distribution': p_s,
            'warmup_evaluations': warmup_evaluations,
            'total_function_evaluations': len(t_nodes) * num_MC_per_node + warmup_evaluations + residual_samples
        }
        return shapley_value, info_dict
    else:
        return shapley_value


if __name__ == "__main__":
    main()