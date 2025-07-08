"""
Utility functions for Integral Shapley Values computation.

This module provides:
- Utility functions for measuring model similarity (RKHS, KL divergence, accuracy, cosine)
- Model factory functions for creating different types of classifiers
- Helper functions for data processing and evaluation
"""

from .utilities import (
    utility_acc,
    utility_RKHS, 
    utility_KL,
    utility_cosine,
    rbf_kernel,
    rkhs_inner_product_multiclass,
    rkhs_norm_multiclass,
    rkhs_cosine_similarity_multiclass
)

from .model_utils import return_model

__all__ = [
    'utility_acc',
    'utility_RKHS',
    'utility_KL', 
    'utility_cosine',
    'rbf_kernel',
    'rkhs_inner_product_multiclass',
    'rkhs_norm_multiclass', 
    'rkhs_cosine_similarity_multiclass',
    'return_model'
]