"""
Core computation module for Integral Shapley Values.

This module contains the main algorithms for computing Shapley values
using integral formulation and various numerical integration methods.
"""

from .integral_shapley import (
    compute_integral_shapley_value,
    compute_integral_shapley_trapezoid,
    compute_integral_shapley_gaussian,
    compute_integral_shapley_smart_adaptive,
    monte_carlo_shapley_value,
    exact_shapley_value,
    compute_marginal_contribution_at_t
)

__all__ = [
    'compute_integral_shapley_value',
    'compute_integral_shapley_trapezoid', 
    'compute_integral_shapley_gaussian',
    'compute_integral_shapley_smart_adaptive',
    'monte_carlo_shapley_value',
    'exact_shapley_value',
    'compute_marginal_contribution_at_t'
]