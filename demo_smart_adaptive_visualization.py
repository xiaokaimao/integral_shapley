#!/usr/bin/env python
"""
Demo script to visualize Smart Adaptive sampling pattern.

This script demonstrates how to use the Smart Adaptive method with visualization
to see exactly which points are sampled and how the algorithm adapts to function complexity.
"""

import numpy as np
import sys
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.integral_shapley import (
    compute_integral_shapley_value_with_budget,
    visualize_smart_adaptive_sampling
)
from src.utils.utilities import utility_acc
from src.utils.model_utils import return_model


def demo_smart_adaptive_visualization():
    """Demonstrate Smart Adaptive sampling visualization."""
    
    print("="*60)
    print("Smart Adaptive Sampling Visualization Demo")
    print("="*60)
    
    # Load and prepare data
    print("\n1. Loading and preparing Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # Train models
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    n_points = len(x_train)
    print(f"Dataset: Cancer, {n_points} training points")
    
    # Test different tolerance levels
    test_cases = [
        {"name": "High Tolerance (Fast)", "tolerance": 1e-2, "max_depth": 3},
        {"name": "Medium Tolerance (Balanced)", "tolerance": 1e-3, "max_depth": 4},
        {"name": "Low Tolerance (Precise)", "tolerance": 1e-4, "max_depth": 5}
    ]
    
    # Choose a representative data point
    target_point = 0
    
    for case_idx, case in enumerate(test_cases):
        print(f"\n{case_idx + 2}. Computing Smart Adaptive Shapley for point {target_point}")
        print(f"   Case: {case['name']}")
        print(f"   Tolerance: {case['tolerance']}, Max Depth: {case['max_depth']}")
        
        # Compute Shapley value with sampling information
        method_kwargs = {
            'tolerance': case['tolerance'],
            'max_depth': case['max_depth'],
            'num_MC': 50,  # Reduced for faster demo
            'min_samples_per_interval': 3
        }
        
        result = compute_integral_shapley_value_with_budget(
            x_train, y_train, x_valid, y_valid, target_point, 
            clf, final_model, utility_acc, 
            method='smart_adaptive', rounding_method='probabilistic',
            return_sampling_info=True, **method_kwargs
        )
        
        shapley_value, actual_budget, sampling_info = result
        
        print(f"   Shapley Value: {shapley_value:.6f}")
        print(f"   Actual Budget: {actual_budget}")
        print(f"   Number of Intervals: {len(sampling_info['intervals'])}")
        print(f"   Total Sampling Points: {sum([info['samples'] for info in sampling_info['interval_info']])}")
        
        # Create visualization
        save_path = f"results/plots/smart_adaptive_demo_{case['name'].lower().replace(' ', '_')}_point_{target_point}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"   Creating visualization...")
        visualize_smart_adaptive_sampling(
            sampling_info, 
            target_point, 
            save_path=save_path
        )
        
        print(f"   ✓ Visualization saved to: {save_path}")
    
    print(f"\n{len(test_cases) + 2}. Summary")
    print("="*60)
    print("Demo completed! Check the generated plots to see:")
    print("1. How Smart Adaptive subdivides the [0,1] interval")
    print("2. Where sampling points are concentrated")
    print("3. The integrand function values at sampling points")
    print("4. Contribution of each interval to the final Shapley value")
    print("\nKey insights:")
    print("- Lower tolerance = more intervals = higher sampling density")
    print("- Intervals with high variation get more sampling points")
    print("- Different data points may have different complexity patterns")


if __name__ == "__main__":
    try:
        demo_smart_adaptive_visualization()
        print("\n✅ Smart Adaptive visualization demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()