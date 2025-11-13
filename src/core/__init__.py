"""
Core module for integral Shapley value computation.

This package provides modular implementations of various Shapley value computation methods:
- Base utilities and common functions
- Basic integration methods (trapezoid, Gaussian, Simpson)  
- Adaptive sampling methods
- Complementary Contribution (CC) methods
- Advanced sampling methods (importance sampling, sparse residual)
- Traditional methods (Monte Carlo, exact, stratified)
- Unified interface for all methods
"""

# Import main interface
from .shapley_interface import *

# For backward compatibility, also import key functions directly
from .base import compute_marginal_contribution_at_t, compute_mare
from .basic_integration import (
    compute_integral_shapley_trapezoid,
    compute_integral_shapley_simpson
)
from .adaptive_methods import compute_integral_shapley_smart_adaptive
from .traditional_methods import monte_carlo_shapley_value, exact_shapley_value

# Version information
__version__ = "1.0.0"
__author__ = "Integral Shapley Research Team"