#!/usr/bin/env python
"""
Base functions and utilities for integral Shapley value computation.

This module contains core utility functions used across different methods.
"""

import numpy as np
import random
from sklearn.base import clone
from typing import Optional

# Import math utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import compute_coalition_size

def compute_marginal_contribution_at_t(
    t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
    utility_func, num_MC=50, rounding_method='round', rng=None
):
    """
    方案B：直接在 t 上取层（按 n 分区），再做 MC 估计 E[Δ(t,i)].
    - n = |train|, others = n-1
    - floor : m = floor(n * t)
      round : m = round(n * t)
      ceil  : m = ceil(n * t)
    - probabilistic : 令 x = n*t, s=floor(x), delta=x-s；
        以 P(s)=1-delta, P(s+1)=delta 采样一次 m（若 s=n-1 则 m=n-1）
    这样底层“取层”与你上层的 n-等分映射完全对齐，可直接在 t 上计算。
    """
    import numpy as np
    from sklearn.base import clone

    if rng is None:
        rng = np.random.default_rng()

    # 安全：t ∈ [0,1]
    t = float(t)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0

    n = x_train.shape[0]
    others = n - 1
    if others < 0:
        return 0.0

    # ---- t -> m（按 n 分区）----
    def t_to_m(tval: float) -> int:
        x = n * tval
        if rounding_method == 'floor':
            m = int(np.floor(x))
        elif rounding_method == 'ceil':
            m = int(np.ceil(x))
        elif rounding_method == 'round':
            m = int(np.floor(x + 0.5))
        elif rounding_method == 'probabilistic':
            s = int(np.floor(x))
            if s >= n:  # t=1 的情况
                return others
            delta = x - s  # ∈ [0,1)
            if s >= others:  # s==n-1
                return others
            # 伯努利一次，成本不翻倍
            return s if (rng.random() < (1.0 - delta)) else (s + 1)
        else:
            raise ValueError(f"Unknown rounding_method: {rounding_method}")

        # 夹取到 [0, others]
        if m < 0: m = 0
        if m > others: m = others
        return m

    cand = np.delete(np.arange(n), i)  # 其余样本索引
    d = x_train.shape[1] if x_train.ndim == 2 else None

    mc_values = []
    for _ in range(num_MC):
        m = t_to_m(t)

        if m == 0:
            X_sub = np.empty((0, d)) if d is not None else np.empty((0, 0))
            y_sub = np.empty((0,), dtype=y_train.dtype)
        else:
            idx = rng.choice(cand, size=m, replace=False)
            X_sub = x_train[idx]
            y_sub = y_train[idx]

        # u(S)
        uS = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)

        # u(S ∪ {i})
        if m == 0:
            X_Si = x_train[i][None, :]
            y_Si = y_train[i][None]
        else:
            X_Si = np.vstack([X_sub, x_train[i]])
            y_Si = np.concatenate([y_sub, y_train[i][None]])

        uSi = utility_func(X_Si, y_Si, x_valid, y_valid, clone(clf), final_model)
        mc_values.append(uSi - uS)

    return float(np.mean(mc_values))


# def compute_marginal_contribution_at_t(
#     t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_MC=50, rounding_method='probabilistic', rng=None
# ):
#     """
#     估计在给定 t 的 E[Δ(t,i)]
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     n = x_train.shape[0]
#     N_others = n - 1
#     cand = np.delete(np.arange(n), i)  # 其他样本索引

#     mc_values = []
#     for _ in range(num_MC):
#         m = compute_coalition_size(t, N_others, method=rounding_method, rng=rng)

#         if m == 0:
#             X_sub = np.empty((0, x_train.shape[1]))
#             y_sub = np.empty((0,), dtype=y_train.dtype)
#         else:
#             idx = rng.choice(cand, size=m, replace=False)
#             X_sub = x_train[idx]
#             y_sub = y_train[idx]

#         # u(S)
#         uS = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)

#         # u(S ∪ {i})
#         if m == 0:
#             X_Si = x_train[i][None, :]
#             y_Si = y_train[i][None]
#         else:
#             X_Si = np.vstack([X_sub, x_train[i]])
#             y_Si = np.concatenate([y_sub, y_train[i][None]])

#         uSi = utility_func(X_Si, y_Si, x_valid, y_valid, clone(clf), final_model)

#         mc_values.append(uSi - uS)

#     return float(np.mean(mc_values))


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


def compute_mare(predicted, ground_truth, epsilon=1e-8):
    """
    计算平均相对误差 (MARE - Mean Absolute Relative Error)
    
    Args:
        predicted: 预测的Shapley值
        ground_truth: 真实的Shapley值
        epsilon: 避免除零的最小阈值
        
    Returns:
        mare: 平均相对误差 (0到1之间的值)
    """
    # 处理接近零的ground truth值
    mask = np.abs(ground_truth) >= epsilon
    
    if np.sum(mask) == 0:
        print(f"Warning: All ground truth values are near zero (< {epsilon})")
        return np.nan
    
    # 只计算非零值的相对误差
    relative_errors = np.abs((predicted[mask] - ground_truth[mask]) / ground_truth[mask])
    mare = np.mean(relative_errors)
    
    # 统计信息
    n_total = len(ground_truth)
    n_valid = np.sum(mask)
    if n_valid < n_total:
        print(f"  Note: Used {n_valid}/{n_total} points (excluded {n_total-n_valid} near-zero values)")
    
    return mare