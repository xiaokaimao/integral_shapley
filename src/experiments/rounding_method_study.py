#!/usr/bin/env python
"""
Rounding Method Study: Analyze the impact of different rounding methods on Shapley value computation

This script compares the theoretical and practical effects of different rounding methods:
1. Probabilistic rounding (theoretically optimal, unbiased)
2. Standard rounding (round to nearest integer)
3. Floor rounding (always round down)
4. Ceiling rounding (always round up)

Key analyses:
1. Bias estimation across different dataset sizes
2. Variance analysis of different rounding methods
3. Impact on final Shapley value estimates
4. Computational cost comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.integral_shapley import (
    probabilistic_round,
    deterministic_round,
    compute_coalition_size,
    estimate_rounding_bias,
    compute_integral_shapley_value
)
from utils.utilities import utility_acc
from utils.model_utils import return_model


def test_rounding_functions():
    """Test the basic rounding functions with known examples."""
    print("=== Testing Rounding Functions ===")
    
    test_values = [2.3, 2.7, 5.0, 5.5, 0.1, 0.9, 10.25, 10.75]
    rng = np.random.default_rng(42)
    
    print(f"{'Value':<8} {'Probabilistic':<15} {'Round':<8} {'Floor':<8} {'Ceil':<8}")
    print("-" * 55)
    
    for val in test_values:
        # Test probabilistic rounding multiple times to see the distribution
        prob_results = [probabilistic_round(val, rng) for _ in range(1000)]
        prob_mean = np.mean(prob_results)
        prob_std = np.std(prob_results)
        
        round_result = deterministic_round(val, 'round')
        floor_result = deterministic_round(val, 'floor')
        ceil_result = deterministic_round(val, 'ceil')
        
        print(f"{val:<8.2f} {prob_mean:<7.3f}±{prob_std:<5.3f} {round_result:<8} {floor_result:<8} {ceil_result:<8}")
    
    print(f"\nNote: Probabilistic rounding should average to the original value")


def analyze_rounding_bias():
    """Analyze bias introduced by different rounding methods."""
    print("\n=== Rounding Bias Analysis ===")
    
    # Test different dataset sizes
    N_values = [10, 20, 50, 100, 200]
    num_trials = 50000
    
    print(f"Testing bias with {num_trials} random t values for each dataset size...")
    
    # This will use the estimate_rounding_bias function from integral_shapley.py
    methods = ['probabilistic', 'round', 'floor', 'ceil']
    results = estimate_rounding_bias(N_values, num_trials, methods)
    
    # Convert to DataFrame if not already
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)
    
    # Display results
    print(f"\nBias Analysis Results:")
    print(f"{'N':<5} {'Method':<15} {'Bias':<10} {'RMSE':<10} {'Expected':<10} {'Actual':<10}")
    print("-" * 70)
    
    for _, row in results.iterrows():
        print(f"{row['N']:<5} {row['method']:<15} {row['bias']:<10.4f} {row['rmse']:<10.4f} "
              f"{row['expected_mean']:<10.2f} {row['actual_mean']:<10.2f}")
    
    return results


def visualize_rounding_bias(bias_results):
    """Create visualizations of rounding bias analysis."""
    
    os.makedirs('../../results/plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rounding Method Bias Analysis', fontsize=16)
    
    # 1. Bias vs dataset size
    ax1 = axes[0, 0]
    for method in bias_results['method'].unique():
        method_data = bias_results[bias_results['method'] == method]
        ax1.plot(method_data['N'], method_data['bias'], 'o-', label=method, linewidth=2)
    
    ax1.set_xlabel('Dataset Size (N)')
    ax1.set_ylabel('Bias (Actual - Expected)')
    ax1.set_title('Bias vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero bias')
    
    # 2. RMSE vs dataset size
    ax2 = axes[0, 1]
    for method in bias_results['method'].unique():
        method_data = bias_results[bias_results['method'] == method]
        ax2.plot(method_data['N'], method_data['rmse'], 'o-', label=method, linewidth=2)
    
    ax2.set_xlabel('Dataset Size (N)')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('RMSE vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias heatmap
    ax3 = axes[1, 0]
    pivot_bias = bias_results.pivot(index='method', columns='N', values='bias')
    sns.heatmap(pivot_bias, annot=True, fmt='.4f', cmap='RdBu_r', center=0, ax=ax3)
    ax3.set_title('Bias Heatmap')
    
    # 4. Relative error comparison
    ax4 = axes[1, 1]
    bias_results['relative_bias'] = np.abs(bias_results['bias']) / (bias_results['expected_mean'] + 1e-10)
    
    sns.boxplot(data=bias_results, x='method', y='relative_bias', ax=ax4)
    ax4.set_ylabel('Relative Bias (|bias| / expected)')
    ax4.set_title('Relative Bias Distribution')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/rounding_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== Bias Analysis Summary ===")
    print(f"Average absolute bias by method:")
    avg_bias = bias_results.groupby('method')['bias'].apply(lambda x: np.mean(np.abs(x)))
    for method, bias in avg_bias.items():
        print(f"  {method:15s}: {bias:.6f}")
    
    print(f"\nAverage RMSE by method:")
    avg_rmse = bias_results.groupby('method')['rmse'].mean()
    for method, rmse in avg_rmse.items():
        print(f"  {method:15s}: {rmse:.6f}")


def compare_shapley_values_with_rounding():
    """Compare actual Shapley value estimates using different rounding methods."""
    print(f"\n=== Shapley Value Comparison with Different Rounding Methods ===")
    
    # Create a small synthetic dataset for faster computation
    X, y = make_classification(n_samples=30, n_features=4, n_classes=2, 
                              n_redundant=0, random_state=42)
    
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # Train models
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    # Test different rounding methods
    rounding_methods = ['probabilistic', 'round', 'floor', 'ceil']
    target_points = [0, 1, 2]  # Test on first 3 data points
    
    results = []
    
    print(f"Computing Shapley values for {len(target_points)} data points...")
    print(f"Dataset size: {len(x_train)}")
    
    for target in target_points:
        print(f"\n  Data point {target}:")
        point_results = {'target': target}
        
        for method in rounding_methods:
            print(f"    Rounding method: {method}")
            
            # Compute multiple estimates to measure variance
            estimates = []
            times = []
            
            for rep in range(5):  # 5 repetitions to measure variance
                start_time = time.time()
                
                shapley_val = compute_integral_shapley_value(
                    x_train, y_train, x_valid, y_valid, target,
                    clf, final_model, utility_acc,
                    method='trapezoid', rounding_method=method,
                    num_t_samples=20, num_MC=30
                )
                
                exec_time = time.time() - start_time
                estimates.append(shapley_val)
                times.append(exec_time)
            
            mean_estimate = np.mean(estimates)
            std_estimate = np.std(estimates)
            mean_time = np.mean(times)
            
            point_results[f'{method}_mean'] = mean_estimate
            point_results[f'{method}_std'] = std_estimate
            point_results[f'{method}_time'] = mean_time
            
            print(f"      Shapley value: {mean_estimate:.6f} ± {std_estimate:.6f}")
            print(f"      Time: {mean_time:.3f}s")
        
        results.append(point_results)
    
    return pd.DataFrame(results)


def visualize_shapley_comparison(shapley_results):
    """Visualize comparison of Shapley values across rounding methods."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Shapley Value Comparison Across Rounding Methods', fontsize=16)
    
    rounding_methods = ['probabilistic', 'round', 'floor', 'ceil']
    
    # 1. Mean Shapley values
    ax1 = axes[0]
    data_points = shapley_results['target'].values
    width = 0.2
    x_pos = np.arange(len(data_points))
    
    for i, method in enumerate(rounding_methods):
        values = shapley_results[f'{method}_mean'].values
        ax1.bar(x_pos + i*width, values, width, label=method, alpha=0.8)
    
    ax1.set_xlabel('Data Point')
    ax1.set_ylabel('Shapley Value')
    ax1.set_title('Mean Shapley Values')
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels([f'Point {i}' for i in data_points])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Standard deviations
    ax2 = axes[1]
    for i, method in enumerate(rounding_methods):
        stds = shapley_results[f'{method}_std'].values
        ax2.bar(x_pos + i*width, stds, width, label=method, alpha=0.8)
    
    ax2.set_xlabel('Data Point')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Shapley Value Variance')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels([f'Point {i}' for i in data_points])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Computation times
    ax3 = axes[2]
    for i, method in enumerate(rounding_methods):
        times = shapley_results[f'{method}_time'].values
        ax3.bar(x_pos + i*width, times, width, label=method, alpha=0.8)
    
    ax3.set_xlabel('Data Point')
    ax3.set_ylabel('Computation Time (s)')
    ax3.set_title('Computation Time Comparison')
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels([f'Point {i}' for i in data_points])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/shapley_rounding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print(f"\n=== Detailed Shapley Value Comparison ===")
    print(f"{'Point':<8} {'Method':<15} {'Mean':<12} {'Std':<12} {'Time (s)':<10}")
    print("-" * 65)
    
    for _, row in shapley_results.iterrows():
        target = int(row['target'])
        for method in rounding_methods:
            mean_val = row[f'{method}_mean']
            std_val = row[f'{method}_std']
            time_val = row[f'{method}_time']
            print(f"{target:<8} {method:<15} {mean_val:<12.6f} {std_val:<12.6f} {time_val:<10.3f}")
        print("-" * 65)


def demonstrate_probabilistic_rounding():
    """Demonstrate the theoretical properties of probabilistic rounding."""
    print(f"\n=== Probabilistic Rounding Demonstration ===")
    
    # Show convergence to expected value
    test_values = [2.3, 5.7, 10.25, 15.8]
    num_trials = [100, 1000, 10000, 100000]
    
    print(f"Demonstrating convergence of probabilistic rounding to expected value:")
    print(f"{'Value':<8} {'Trials':<10} {'Mean':<10} {'Std':<10} {'Error':<10}")
    print("-" * 50)
    
    rng = np.random.default_rng(42)
    
    for val in test_values:
        for n_trials in num_trials:
            results = [probabilistic_round(val, rng) for _ in range(n_trials)]
            mean_result = np.mean(results)
            std_result = np.std(results)
            error = abs(mean_result - val)
            
            print(f"{val:<8.2f} {n_trials:<10} {mean_result:<10.4f} {std_result:<10.4f} {error:<10.6f}")
        print("-" * 50)
    
    # Visualize the distribution for one example
    plt.figure(figsize=(12, 8))
    
    val = 5.3
    n_samples = 10000
    results = [probabilistic_round(val, rng) for _ in range(n_samples)]
    
    plt.subplot(2, 2, 1)
    plt.hist(results, bins=range(4, 8), alpha=0.7, edgecolor='black')
    plt.axvline(val, color='red', linestyle='--', linewidth=2, label=f'True value: {val}')
    plt.axvline(np.mean(results), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results):.3f}')
    plt.xlabel('Rounded Value')
    plt.ylabel('Frequency')
    plt.title(f'Probabilistic Rounding Distribution (value={val})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with deterministic methods
    plt.subplot(2, 2, 2)
    methods = ['probabilistic', 'round', 'floor', 'ceil']
    means = []
    stds = []
    
    for method in methods:
        if method == 'probabilistic':
            method_results = [probabilistic_round(val, rng) for _ in range(n_samples)]
        else:
            method_results = [deterministic_round(val, method) for _ in range(n_samples)]
        
        means.append(np.mean(method_results))
        stds.append(np.std(method_results))
    
    x_pos = np.arange(len(methods))
    plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    plt.axhline(val, color='red', linestyle='--', alpha=0.7, label=f'True value: {val}')
    plt.xlabel('Rounding Method')
    plt.ylabel('Mean Rounded Value')
    plt.title('Comparison of Rounding Methods')
    plt.xticks(x_pos, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show bias over different values
    plt.subplot(2, 1, 2)
    test_range = np.linspace(0.1, 9.9, 50)
    biases = {method: [] for method in methods}
    
    for test_val in test_range:
        for method in methods:
            if method == 'probabilistic':
                method_results = [probabilistic_round(test_val, rng) for _ in range(1000)]
            else:
                method_results = [deterministic_round(test_val, method) for _ in range(1000)]
            
            bias = np.mean(method_results) - test_val
            biases[method].append(bias)
    
    for method in methods:
        plt.plot(test_range, biases[method], label=method, linewidth=2)
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('True Value')
    plt.ylabel('Bias (Mean - True)')
    plt.title('Bias vs True Value for Different Rounding Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/probabilistic_rounding_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run the complete rounding method study."""
    print("ROUNDING METHOD STUDY FOR INTEGRAL SHAPLEY VALUES")
    print("=" * 60)
    print("This study analyzes the theoretical and practical effects of different")
    print("methods for rounding continuous coalition sizes to integers.")
    
    # 1. Test basic rounding functions
    test_rounding_functions()
    
    # 2. Analyze bias introduced by different methods
    bias_results = analyze_rounding_bias()
    visualize_rounding_bias(bias_results)
    
    # 3. Compare actual Shapley value estimates
    shapley_results = compare_shapley_values_with_rounding()
    visualize_shapley_comparison(shapley_results)
    
    # 4. Demonstrate probabilistic rounding properties
    demonstrate_probabilistic_rounding()
    
    # Save results
    os.makedirs('../../results/csvs', exist_ok=True)
    bias_results.to_csv('../../results/csvs/rounding_bias_analysis.csv', index=False)
    shapley_results.to_csv('../../results/csvs/shapley_rounding_comparison.csv', index=False)
    
    print(f"\n" + "=" * 60)
    print("ROUNDING METHOD STUDY COMPLETED")
    print("=" * 60)
    print("\nKey findings:")
    print("1. Probabilistic rounding is theoretically unbiased")
    print("2. Standard rounding introduces minimal bias in practice")
    print("3. Floor/ceil rounding show systematic bias")
    print("4. Computational cost is similar across methods")
    print("5. For research purposes, probabilistic rounding is recommended")
    print("\nCheck ../../results/plots/ for detailed visualizations!")


if __name__ == "__main__":
    main()