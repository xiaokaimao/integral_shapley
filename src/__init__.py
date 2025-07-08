"""
Integral Shapley Values (ISV) - A research toolkit for efficient Shapley value computation

This package implements Shapley value computation using integral formulation:
SV_i = ∫_0^1 E[Δ(t,i)] dt

Key modules:
- core.integral_shapley: Main computation engine
- utils.utilities: Utility functions for different similarity measures
- utils.model_utils: Model factory and helpers
- experiments: Research scripts for analysis and comparison
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Import main functions for easy access
from src.core.integral_shapley import compute_integral_shapley_value
from src.utils.utilities import utility_acc, utility_RKHS, utility_KL, utility_cosine
from src.utils.model_utils import return_model

__all__ = [
    'compute_integral_shapley_value',
    'utility_acc', 
    'utility_RKHS',
    'utility_KL', 
    'utility_cosine',
    'return_model'
]