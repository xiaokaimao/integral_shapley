#!/usr/bin/env python
"""
Example Usage of Integral Shapley Values

This script demonstrates the basic usage of the integral Shapley values toolkit,
showcasing different integration methods and their advantages.
"""

import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the integral Shapley toolkit
from src.core.integral_shapley import compute_integral_shapley_value, monte_carlo_shapley_value
from src.utils.utilities import utility_acc, utility_RKHS, utility_KL
from src.utils.model_utils import return_model


def basic_example():
    """Basic example of computing integral Shapley values."""
    
    print("=" * 60)
    print("BASIC INTEGRAL SHAPLEY VALUES EXAMPLE")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    print(f"   Training set size: {len(x_train)}")
    print(f"   Validation set size: {len(x_valid)}")
    
    # Train final model
    print("\n2. Training reference model...")
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    accuracy = final_model.score(x_valid, y_valid)
    print(f"   Model accuracy: {accuracy:.3f}")
    
    # Compute Shapley values for first data point using different methods
    target_point = 0
    print(f"\n3. Computing Shapley value for data point {target_point}...")
    
    methods = {
        'trapezoid': {'num_t_samples': 30, 'num_MC': 50},
        'gaussian': {'num_nodes': 16, 'num_MC': 50},
        'adaptive': {'tolerance': 1e-3, 'num_MC': 50}
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"\n   Method: {method_name}")
        start_time = time.time()
        
        shapley_value = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, target_point,
            clf, final_model, utility_acc, method=method_name, 
            rounding_method='probabilistic', **params
        )
        
        exec_time = time.time() - start_time
        results[method_name] = (shapley_value, exec_time)
        
        print(f"     Shapley value: {shapley_value:.6f}")
        print(f"     Execution time: {exec_time:.3f}s")
        print(f"     (using probabilistic rounding)")
    
    # Compare with traditional Monte Carlo
    print(f"\n   Method: traditional_monte_carlo")
    start_time = time.time()
    
    mc_value = monte_carlo_shapley_value(
        target_point, x_train, y_train, x_valid, y_valid,
        clf, final_model, utility_acc, num_samples=2000
    )
    
    mc_time = time.time() - start_time
    results['monte_carlo'] = (mc_value, mc_time)
    
    print(f"     Shapley value: {mc_value:.6f}")
    print(f"     Execution time: {mc_time:.3f}s")
    
    # Summary
    print(f"\n4. Results Summary:")
    print(f"   {'Method':<20} {'Shapley Value':<15} {'Time (s)':<10} {'Speedup':<10}")
    print(f"   {'-'*20} {'-'*15} {'-'*10} {'-'*10}")
    
    for method, (value, exec_time) in results.items():
        speedup = mc_time / exec_time if method != 'monte_carlo' else 1.0
        print(f"   {method:<20} {value:<15.6f} {exec_time:<10.3f} {speedup:<10.1f}x")


def utility_comparison_example():
    """Example comparing different utility functions."""
    
    print("\n\n" + "=" * 60)
    print("UTILITY FUNCTION COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Prepare data (same as before)
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    # Test different utility functions
    utilities = {
        'accuracy': utility_acc,
        'rkhs_similarity': utility_RKHS,
        'kl_divergence': utility_KL
    }
    
    target_point = 0
    method = 'trapezoid'
    params = {'num_t_samples': 30, 'num_MC': 50}
    
    print(f"\nComputing Shapley values with different utility functions...")
    print(f"Data point: {target_point}, Method: {method}")
    
    utility_results = {}
    
    for utility_name, utility_func in utilities.items():
        print(f"\n  Utility: {utility_name}")
        
        try:
            start_time = time.time()
            shapley_value = compute_integral_shapley_value(
                x_train, y_train, x_valid, y_valid, target_point,
                clf, final_model, utility_func, method=method, **params
            )
            exec_time = time.time() - start_time
            
            utility_results[utility_name] = (shapley_value, exec_time)
            print(f"    Shapley value: {shapley_value:.6f}")
            print(f"    Execution time: {exec_time:.3f}s")
            
        except Exception as e:
            print(f"    Error: {e}")
            utility_results[utility_name] = (None, None)
    
    # Summary
    print(f"\nUtility Function Results Summary:")
    print(f"{'Utility':<20} {'Shapley Value':<15} {'Time (s)':<10}")
    print(f"{'-'*20} {'-'*15} {'-'*10}")
    
    for utility, (value, exec_time) in utility_results.items():
        if value is not None:
            print(f"{utility:<20} {value:<15.6f} {exec_time:<10.3f}")
        else:
            print(f"{utility:<20} {'Failed':<15} {'N/A':<10}")


def multiple_points_example():
    """Example computing Shapley values for multiple data points."""
    
    print("\n\n" + "=" * 60)
    print("MULTIPLE DATA POINTS EXAMPLE")
    print("=" * 60)
    
    # Prepare data
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    # Compute for first 5 data points
    target_points = list(range(min(5, len(x_train))))
    method = 'trapezoid'
    params = {'num_t_samples': 20, 'num_MC': 30}
    
    print(f"\nComputing Shapley values for {len(target_points)} data points...")
    print(f"Method: {method}, Parameters: {params}")
    
    all_values = []
    total_time = 0
    
    for i in target_points:
        start_time = time.time()
        shapley_value = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, i,
            clf, final_model, utility_acc, method=method, **params
        )
        exec_time = time.time() - start_time
        total_time += exec_time
        
        all_values.append(shapley_value)
        print(f"  Point {i}: {shapley_value:.6f} (time: {exec_time:.3f}s)")
    
    print(f"\nSummary:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per point: {total_time/len(target_points):.3f}s")
    print(f"  Shapley values range: [{min(all_values):.6f}, {max(all_values):.6f}]")
    print(f"  Mean Shapley value: {np.mean(all_values):.6f}")
    print(f"  Std Shapley value: {np.std(all_values):.6f}")


def rounding_method_example():
    """Example demonstrating different rounding methods for coalition sizes."""
    
    print("\n\n" + "=" * 60)
    print("ROUNDING METHODS COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Prepare data
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    # Test different rounding methods
    rounding_methods = ['probabilistic', 'round', 'floor', 'ceil']
    target_point = 0
    method = 'trapezoid'
    params = {'num_t_samples': 25, 'num_MC': 40}
    
    print(f"\nComparing rounding methods for coalition size calculation...")
    print(f"Data point: {target_point}, Method: {method}, Dataset size: {len(x_train)}")
    print(f"Method parameters: {params}")
    
    rounding_results = {}
    
    for rounding_method in rounding_methods:
        print(f"\n  Rounding method: {rounding_method}")
        
        # Run multiple times to see variance
        values = []
        times = []
        
        for rep in range(3):
            start_time = time.time()
            shapley_value = compute_integral_shapley_value(
                x_train, y_train, x_valid, y_valid, target_point,
                clf, final_model, utility_acc, method=method,
                rounding_method=rounding_method, **params
            )
            exec_time = time.time() - start_time
            
            values.append(shapley_value)
            times.append(exec_time)
        
        mean_value = np.mean(values)
        std_value = np.std(values)
        mean_time = np.mean(times)
        
        rounding_results[rounding_method] = (mean_value, std_value, mean_time)
        
        print(f"    Shapley value: {mean_value:.6f} ± {std_value:.6f}")
        print(f"    Time: {mean_time:.3f}s")
        
        if rounding_method == 'probabilistic':
            print(f"    (Theoretically unbiased - recommended for research)")
        elif rounding_method == 'round':
            print(f"    (Standard rounding - good practical choice)")
        elif rounding_method == 'floor':
            print(f"    (Systematic bias toward smaller coalitions)")
        elif rounding_method == 'ceil':
            print(f"    (Systematic bias toward larger coalitions)")
    
    # Summary
    print(f"\nRounding Methods Summary:")
    print(f"{'Method':<15} {'Mean Value':<12} {'Std Dev':<10} {'Time (s)':<8}")
    print(f"{'-'*15} {'-'*12} {'-'*10} {'-'*8}")
    
    for method, (mean_val, std_val, time_val) in rounding_results.items():
        print(f"{method:<15} {mean_val:<12.6f} {std_val:<10.6f} {time_val:<8.3f}")
    
    print(f"\nRecommendation:")
    print(f"- For research and theoretical accuracy: use 'probabilistic' rounding")
    print(f"- For practical applications: 'round' (standard rounding) is sufficient")
    print(f"- Avoid 'floor' and 'ceil' as they introduce systematic bias")


def main():
    """Run all examples."""
    print("INTEGRAL SHAPLEY VALUES - USAGE EXAMPLES")
    print("This demonstrates the key features and advantages of integral-based Shapley computation")
    
    # Run examples
    basic_example()
    utility_comparison_example()
    multiple_points_example()
    rounding_method_example()
    
    print("\n\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Integral methods are typically faster than traditional Monte Carlo")
    print("2. Different utility functions capture different aspects of data value")
    print("3. The toolkit handles multiple data points efficiently")
    print("4. Adaptive methods can automatically optimize sampling density")
    print("5. Probabilistic rounding eliminates systematic bias (recommended for research)")
    print("\nFor more advanced usage, see the experiments/ directory!")
    print("Run 'python src/experiments/rounding_method_study.py' for detailed rounding analysis!")


if __name__ == "__main__":
    main()