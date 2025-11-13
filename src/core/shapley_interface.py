#!/usr/bin/env python
"""
Main interface for integral Shapley value computation.

This module provides a unified interface to access all Shapley value computation methods:
- Basic integration methods (trapezoid, Gaussian, Simpson)
- Adaptive sampling methods
- CC methods (basic, parallel, integral)
- Advanced sampling methods (importance sampling, sparse residual)
- Traditional methods (Monte Carlo, exact, stratified)
"""

# Import all method modules
from .basic_integration import (
    compute_integral_shapley_trapezoid,
    compute_integral_shapley_simpson,
    compute_integral_shapley_auto
)

from .adaptive_methods import (
    compute_integral_shapley_smart_adaptive,
    visualize_smart_adaptive_sampling
)

from .cc_methods import (
    cc_shapley,
    cc_shapley_parallel,
    # cc_shapley_integral_layer_parallel,
    # cc_shapley_nested_trapz,
    # cc_shapley_sparse_trapezoid
    integral_cc_sparse_all_players_parallel
)

from .advanced_sampling import (
    compute_integral_shapley_importance_sampling,
    compute_integral_shapley_sparse_residual
)

from .traditional_methods import (
    monte_carlo_shapley_value,
    exact_shapley_value,
    stratified_shapley_value,
    stratified_shapley_value_with_plot
)


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
        method: Integration method ('trapezoid', 'simpson', 'adaptive', 'smart_adaptive', 'monte_carlo', 'stratified', 'importance_sampling', 'sparse_residual')
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
    elif method == 'importance_sampling':
        result = compute_integral_shapley_importance_sampling(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
        return result[0] if isinstance(result, tuple) else result
    elif method == 'sparse_residual':
        result = compute_integral_shapley_sparse_residual(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
        return result[0] if isinstance(result, tuple) else result
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
        method: Integration method ('trapezoid', 'simpson',  'adaptive', 'smart_adaptive', 'monte_carlo', 'stratified')
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
        method: Method ('cc', 'cc_parallel', 'cc_trapz', 'cc_integral_parallel', 'cc_sparse_trapz')
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
    elif method == 'cc_integral_parallel':
        sv, info = integral_cc_sparse_all_players_parallel(
            x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
        return sv
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


# Re-export key functions for backward compatibility
__all__ = [
    # Main interfaces
    'compute_integral_shapley_value',
    'compute_integral_shapley_value_with_budget', 
    'compute_all_shapley_values',
    'compute_shapley_for_params',
    'compute_shapley_for_params_with_budget',
    
    # Basic integration methods
    'compute_integral_shapley_trapezoid',
    'compute_integral_shapley_simpson', 
    'compute_integral_shapley_auto',
    
    # Adaptive methods
    'compute_integral_shapley_smart_adaptive',
    'visualize_smart_adaptive_sampling',
    
    # # CC methods
    'cc_shapley',
    'cc_shapley_parallel',
    # 'cc_shapley_integral_layer_parallel',
    # 'cc_shapley_nested_trapz',
    # 'cc_shapley_sparse_trapezoid',
    
    # Advanced sampling
    'compute_integral_shapley_importance_sampling',
    'compute_integral_shapley_sparse_residual',
    
    # Traditional methods
    'monte_carlo_shapley_value',
    'exact_shapley_value', 
    'stratified_shapley_value',
    'stratified_shapley_value_with_plot'
]