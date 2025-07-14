#!/usr/bin/env python
"""
MSE Comparison Study: Compare Integral Shapley vs Stratified Sampling MSE

This script compares the Mean Squared Error (MSE) of two approaches relative to true Shapley values:
1. Integral Shapley Values (our method)
2. Stratified Sampling (traditional approach with same sample budget)

Key analyses:
1. MSE comparison across different sample sizes
2. Statistical significance testing
3. Efficiency analysis (MSE per computational cost)
4. Variance decomposition analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.integral_shapley import (
    compute_integral_shapley_value,
    exact_shapley_value,
    stratified_shapley_value,
    compute_all_shapley_values
)
from utils.utilities import utility_acc
from utils.model_utils import return_model




def compute_mse_comparison_single_point(args):
    """
    Compute MSE comparison for a single data point.
    
    Args:
        args: Tuple containing (point_idx, x_train, y_train, x_valid, y_valid, clf, 
              final_model, utility_func, true_shapley, sample_sizes, n_trials)
    
    Returns:
        results: List of result dictionaries
    """
    (point_idx, x_train, y_train, x_valid, y_valid, clf, final_model, 
     utility_func, true_shapley, sample_sizes, n_trials) = args
    
    results = []
    
    for n_samples in sample_sizes:
        print(f"    Point {point_idx}, n_samples={n_samples}")
        
        # Integral Shapley estimates
        integral_estimates = []
        integral_times = []
        
        for _ in range(n_trials):
            start_time = time.time()
            try:
                # Use trapezoidal method with sample budget
                num_t_samples = min(50, n_samples // 20)  # Adaptive t samples
                num_mc_per_t = n_samples // num_t_samples
                
                estimate = compute_integral_shapley_value(
                    x_train, y_train, x_valid, y_valid, point_idx, clf, final_model,
                    utility_func, method='trapezoid', num_t_samples=num_t_samples, 
                    num_MC=num_mc_per_t, rounding_method='probabilistic'
                )
                integral_estimates.append(estimate)
            except Exception as e:
                print(f"      Integral method failed: {e}")
                integral_estimates.append(np.nan)
            
            integral_times.append(time.time() - start_time)
        
        # Stratified sampling estimates
        stratified_estimates = []
        stratified_times = []
        
        for _ in range(n_trials):
            start_time = time.time()
            try:
                # Use same sample budget for stratified sampling
                # Calculate num_MC per stratum based on total sample budget
                N = len(x_train) - 1  # Exclude target point
                num_MC_per_size = max(1, n_samples // (N + 1))  # Distribute samples across all coalition sizes
                estimate = stratified_shapley_value(
                    point_idx, x_train, y_train, x_valid, y_valid, clf, final_model,
                    utility_func, num_MC=num_MC_per_size
                )
                stratified_estimates.append(estimate)
            except Exception as e:
                print(f"      Stratified method failed: {e}")
                stratified_estimates.append(np.nan)
            
            stratified_times.append(time.time() - start_time)
        
        # Compute MSE and other statistics
        integral_estimates = np.array(integral_estimates)
        stratified_estimates = np.array(stratified_estimates)
        
        # Remove NaN values
        integral_valid = integral_estimates[~np.isnan(integral_estimates)]
        stratified_valid = stratified_estimates[~np.isnan(stratified_estimates)]
        
        if len(integral_valid) > 0 and len(stratified_valid) > 0:
            # MSE computation
            integral_mse = np.mean((integral_valid - true_shapley) ** 2)
            stratified_mse = np.mean((stratified_valid - true_shapley) ** 2)
            
            # Bias and variance
            integral_bias = np.mean(integral_valid) - true_shapley
            stratified_bias = np.mean(stratified_valid) - true_shapley
            
            integral_variance = np.var(integral_valid)
            stratified_variance = np.var(stratified_valid)
            
            # Statistical significance test
            t_stat, p_value = ttest_ind(integral_valid, stratified_valid)
            
            results.append({
                'point_idx': point_idx,
                'n_samples': n_samples,
                'method': 'integral',
                'true_shapley': true_shapley,
                'mse': integral_mse,
                'bias': integral_bias,
                'variance': integral_variance,
                'mean_estimate': np.mean(integral_valid),
                'std_estimate': np.std(integral_valid),
                'mean_time': np.mean(integral_times),
                'n_successful': len(integral_valid),
                'comparison_p_value': p_value,
                'comparison_t_stat': t_stat
            })
            
            results.append({
                'point_idx': point_idx,
                'n_samples': n_samples,
                'method': 'stratified',
                'true_shapley': true_shapley,
                'mse': stratified_mse,
                'bias': stratified_bias,
                'variance': stratified_variance,
                'mean_estimate': np.mean(stratified_valid),
                'std_estimate': np.std(stratified_valid),
                'mean_time': np.mean(stratified_times),
                'n_successful': len(stratified_valid),
                'comparison_p_value': p_value,
                'comparison_t_stat': t_stat
            })
    
    return results


def run_mse_comparison_study():
    """Run comprehensive MSE comparison study."""
    
    # Load small datasets for exact computation
    datasets = {
        'iris': load_iris(),
        'wine': load_wine()
    }
    
    # Sample sizes to test
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 20  # Number of trials per configuration
    
    all_results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n=== MSE Comparison Study: {dataset_name.upper()} ===")
        
        # Prepare data
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use small subset for exact computation
        if len(x_train) > 15:
            indices = np.random.choice(len(x_train), 15, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        
        # Train models
        final_model = return_model('LinearSVC')
        final_model.fit(x_train, y_train)
        clf = return_model('LinearSVC')
        
        # Compute exact Shapley values (ground truth)
        print("  Computing exact Shapley values...")
        exact_shapley_values = {}
        test_points = range(min(5, len(x_train)))  # Test first 5 points
        
        for i in test_points:
            try:
                true_value = exact_shapley_value(
                    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_acc
                )
                exact_shapley_values[i] = true_value
                print(f"    Point {i}: {true_value:.6f}")
            except Exception as e:
                print(f"    Point {i}: Failed to compute exact value: {e}")
        
        # Prepare arguments for parallel processing
        process_args = []
        for point_idx in exact_shapley_values.keys():
            process_args.append((
                point_idx, x_train, y_train, x_valid, y_valid, clf, final_model,
                utility_acc, exact_shapley_values[point_idx], sample_sizes, n_trials
            ))
        
        # Run comparison study
        print("  Running MSE comparison...")
        
        # Sequential processing for debugging
        for args in process_args:
            point_results = compute_mse_comparison_single_point(args)
            for result in point_results:
                result['dataset'] = dataset_name
                result['dataset_size'] = len(x_train)
                all_results.extend([result])
    
    return pd.DataFrame(all_results)


def analyze_mse_results(results_df):
    """Analyze and visualize MSE comparison results."""
    
    os.makedirs('../../results/plots', exist_ok=True)
    
    # 1. MSE Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MSE Comparison: Integral Shapley vs Stratified Sampling', fontsize=16)
    
    datasets = results_df['dataset'].unique()
    
    for idx, dataset in enumerate(datasets):
        if idx >= 2:
            break
            
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        # Plot 1: MSE vs Sample Size
        ax1 = axes[idx, 0]
        for method in ['integral', 'stratified']:
            method_data = dataset_data[dataset_data['method'] == method]
            # Group by sample size and compute mean/std
            grouped = method_data.groupby('n_samples')['mse'].agg(['mean', 'std']).reset_index()
            
            ax1.errorbar(grouped['n_samples'], grouped['mean'], yerr=grouped['std'],
                        label=method, marker='o', capsize=5)
        
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title(f'{dataset.title()} - MSE vs Sample Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Bias vs Variance Decomposition
        ax2 = axes[idx, 1]
        for method in ['integral', 'stratified']:
            method_data = dataset_data[dataset_data['method'] == method]
            grouped = method_data.groupby('n_samples')[['bias', 'variance']].mean().reset_index()
            
            # Stack bias^2 and variance
            bias_squared = grouped['bias'] ** 2
            variance = grouped['variance']
            
            ax2.plot(grouped['n_samples'], bias_squared, '--', label=f'{method} bias²', alpha=0.7)
            ax2.plot(grouped['n_samples'], variance, '-', label=f'{method} variance', alpha=0.7)
        
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Bias² and Variance')
        ax2.set_title(f'{dataset.title()} - Bias-Variance Decomposition')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Statistical Significance Analysis
    plt.figure(figsize=(12, 6))
    
    # Plot significance test results
    plt.subplot(1, 2, 1)
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        integral_data = dataset_data[dataset_data['method'] == 'integral']
        
        # P-values vs sample size
        plt.plot(integral_data['n_samples'], integral_data['comparison_p_value'], 
                'o-', label=dataset, alpha=0.7)
    
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
    plt.xlabel('Sample Size')
    plt.ylabel('P-value (t-test)')
    plt.title('Statistical Significance of Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # MSE Ratio Plot
    plt.subplot(1, 2, 2)
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        # Compute MSE ratio (integral/stratified)
        pivot_data = dataset_data.pivot_table(
            index=['n_samples', 'point_idx'], 
            columns='method', 
            values='mse'
        ).reset_index()
        
        if 'integral' in pivot_data.columns and 'stratified' in pivot_data.columns:
            pivot_data['mse_ratio'] = pivot_data['integral'] / pivot_data['stratified']
            grouped_ratio = pivot_data.groupby('n_samples')['mse_ratio'].mean().reset_index()
            
            plt.plot(grouped_ratio['n_samples'], grouped_ratio['mse_ratio'], 
                    'o-', label=dataset, alpha=0.7)
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal MSE')
    plt.xlabel('Sample Size')
    plt.ylabel('MSE Ratio (Integral/Stratified)')
    plt.title('MSE Ratio vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/mse_significance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Efficiency Analysis (MSE per unit time)
    plt.figure(figsize=(10, 6))
    
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        for method in ['integral', 'stratified']:
            method_data = dataset_data[dataset_data['method'] == method]
            grouped = method_data.groupby('n_samples')[['mse', 'mean_time']].mean().reset_index()
            
            # Efficiency = 1 / (MSE * time)
            efficiency = 1 / (grouped['mse'] * grouped['mean_time'])
            
            plt.plot(grouped['n_samples'], efficiency, 'o-', 
                    label=f'{dataset} {method}', alpha=0.7)
    
    plt.xlabel('Sample Size')
    plt.ylabel('Efficiency (1/(MSE × Time))')
    plt.title('Computational Efficiency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/mse_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_mse_summary(results_df):
    """Print comprehensive MSE comparison summary."""
    
    print("\n" + "="*70)
    print("MSE COMPARISON STUDY SUMMARY")
    print("="*70)
    
    # 1. Overall MSE comparison
    print("\n1. Mean MSE by Method:")
    print("-" * 25)
    
    overall_mse = results_df.groupby(['dataset', 'method'])['mse'].mean().reset_index()
    pivot_mse = overall_mse.pivot(index='dataset', columns='method', values='mse')
    
    print(pivot_mse)
    
    # 2. MSE improvement
    print("\n2. MSE Improvement (Integral vs Stratified):")
    print("-" * 48)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        integral_mse = dataset_data[dataset_data['method'] == 'integral']['mse'].mean()
        stratified_mse = dataset_data[dataset_data['method'] == 'stratified']['mse'].mean()
        
        improvement = (stratified_mse - integral_mse) / stratified_mse * 100
        
        print(f"  {dataset.upper()}: {improvement:+.1f}% improvement")
    
    # 3. Statistical significance
    print("\n3. Statistical Significance:")
    print("-" * 30)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        integral_data = dataset_data[dataset_data['method'] == 'integral']
        
        # Check how many configurations show significant difference
        significant_configs = (integral_data['comparison_p_value'] < 0.05).sum()
        total_configs = len(integral_data)
        
        print(f"  {dataset.upper()}: {significant_configs}/{total_configs} configurations show significant difference")
    
    # 4. Bias-Variance Analysis
    print("\n4. Bias-Variance Analysis:")
    print("-" * 27)
    
    for dataset in results_df['dataset'].unique():
        print(f"\n  {dataset.upper()}:")
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        for method in ['integral', 'stratified']:
            method_data = dataset_data[dataset_data['method'] == method]
            avg_bias = method_data['bias'].abs().mean()
            avg_variance = method_data['variance'].mean()
            
            print(f"    {method:10s}: |bias|={avg_bias:.4f}, variance={avg_variance:.4f}")
    
    # 5. Efficiency summary
    print("\n5. Computational Efficiency:")
    print("-" * 30)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        print(f"\n  {dataset.upper()}:")
        for method in ['integral', 'stratified']:
            method_data = dataset_data[dataset_data['method'] == method]
            avg_time = method_data['mean_time'].mean()
            avg_mse = method_data['mse'].mean()
            
            efficiency = 1 / (avg_mse * avg_time)
            print(f"    {method:10s}: time={avg_time:.3f}s, MSE={avg_mse:.6f}, efficiency={efficiency:.1f}")


def main():
    """Run the complete MSE comparison study."""
    print("Starting MSE Comparison Study: Integral Shapley vs Stratified Sampling")
    print("=" * 75)
    
    # Run the study
    results_df = run_mse_comparison_study()
    
    # Save raw results
    os.makedirs('../../results/csvs', exist_ok=True)
    results_df.to_csv('../../results/csvs/mse_comparison_results.csv', index=False)
    print(f"\nRaw results saved to ../../results/csvs/mse_comparison_results.csv")
    
    # Analyze results
    analyze_mse_results(results_df)
    
    # Print summary
    print_mse_summary(results_df)
    
    print("\nMSE comparison study completed!")
    print("Check ../../results/plots/ for visualizations")


if __name__ == "__main__":
    main()