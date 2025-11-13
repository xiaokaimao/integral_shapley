"""
Mathematical utility functions for integral Shapley value computation.

This module contains general-purpose mathematical functions that are used
across the integral Shapley value implementation, particularly for handling
coalition size computation and probabilistic rounding.
"""

import numpy as np
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# def probabilistic_round(x, rng=None):
#     """
#     Probabilistic rounding to eliminate systematic bias.
    
#     For a float x, randomly choose between floor(x) and ceil(x) with 
#     probabilities proportional to their distances from x.
    
#     Args:
#         x: Float value to round
#         rng: Random number generator (optional)
    
#     Returns:
#         Integer result of probabilistic rounding
        
#     Example:
#         For x = 5.3:
#         - Returns 5 with probability 0.7 (6 - 5.3)
#         - Returns 6 with probability 0.3 (5.3 - 5)
#     """
#     if rng is None:
#         rng = np.random.default_rng()
    
#     # Handle negative numbers
#     if x < 0:
#         return -probabilistic_round(-x, rng)
    
#     # Get floor and ceiling
#     x_floor = int(np.floor(x))
#     x_ceil = int(np.ceil(x))
    
#     # If x is already an integer, return it
#     if x_floor == x_ceil:
#         return x_floor
    
#     # Calculate probability of choosing floor
#     # P(floor) = (ceil - x) / (ceil - floor) = ceil - x (since ceil - floor = 1)
#     prob_floor = x_ceil - x
    
#     # Randomly choose between floor and ceil
#     if rng.random() < prob_floor:
#         return x_floor
#     else:
#         return x_ceil


# def deterministic_round(x, method='round'):
#     """
#     Deterministic rounding methods for comparison.
    
#     Args:
#         x: Float value to round
#         method: 'round', 'floor', or 'ceil'
    
#     Returns:
#         Integer result of deterministic rounding
#     """
#     if method == 'round':
#         return int(np.round(x))
#     elif method == 'floor':
#         return int(np.floor(x))
#     elif method == 'ceil':
#         return int(np.ceil(x))
#     else:
#         raise ValueError(f"Unknown rounding method: {method}")


def compute_coalition_size(t, N_others, method='probabilistic', rng=None):
    """
    返回在 t 下的联盟规模 m，基于“其他样本数” N_others（= n-1）。
    method: 'probabilistic' | 'round' | 'floor' | 'ceil'
    """
    if rng is None:
        rng = np.random.default_rng()
    t = float(np.clip(t, 0.0, 1.0))
    x = t * N_others
    if method == 'probabilistic':
        s = int(np.floor(x))
        delta = x - s
        if rng.random() < delta:
            s += 1
    elif method == 'round':
        s = int(np.round(x))
    elif method == 'floor':
        s = int(np.floor(x))
    elif method == 'ceil':
        s = int(np.ceil(x))
    else:
        raise ValueError(f"Unknown rounding_method: {method}")
    return int(np.clip(s, 0, N_others))


def estimate_rounding_bias(N_values, num_trials=10000, methods=['round', 'floor', 'ceil', 'probabilistic']):
    """
    Estimate the bias introduced by different rounding methods.
    
    Args:
        N_values: List of dataset sizes to test
        num_trials: Number of random t values to test
        methods: List of rounding methods to compare
    
    Returns:
        DataFrame with bias analysis results
    """
    results = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    for N in N_values:
        for method in methods:
            # Generate random t values
            t_values = rng.uniform(0, 1, num_trials)
            
            # Compute coalition sizes
            coalition_sizes = []
            for t in t_values:
                s = compute_coalition_size(t, N, method=method, rng=rng)
                coalition_sizes.append(s)
            
            # Compute expected and actual values
            expected_sizes = t_values * (N - 1)
            actual_sizes = np.array(coalition_sizes)
            
            # Compute bias metrics
            bias = np.mean(actual_sizes - expected_sizes)
            rmse = np.sqrt(np.mean((actual_sizes - expected_sizes)**2))
            
            results.append({
                'N': N,
                'method': method,
                'bias': bias,
                'rmse': rmse,
                'expected_mean': np.mean(expected_sizes),
                'actual_mean': np.mean(actual_sizes)
            })
    
    return pd.DataFrame(results) if HAS_PANDAS else results