#!/usr/bin/env python
"""
Convergence Study: Analyze convergence properties of integral Shapley methods

This script studies how different integration methods converge to the true Shapley value
as we increase the number of sample points. Key analyses include:

1. Convergence rate comparison across methods
2. Sample efficiency analysis  
3. Error estimation and confidence intervals
4. Optimal parameter selection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.integral_shapley import (
    compute_integral_shapley_value,
    monte_carlo_shapley_value,
    exact_shapley_value
)
from utils.utilities import utility_acc, utility_RKHS
from utils.model_utils import return_model


def run_convergence_experiment(x_train, y_train, x_valid, y_valid, target_point, 
                             clf, final_model, utility_func, true_value=None):
    """
    Run convergence experiment for a single data point across different sample sizes.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        target_point: Index of target data point
        clf: Classifier template
        final_model: Reference model
        utility_func: Utility function
        true_value: Ground truth Shapley value (if available)
    
    Returns:
        results: DataFrame with convergence data
    """
    
    # Sample sizes to test
    trapezoid_samples = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    gaussian_nodes = [4, 8, 12, 16, 20, 24, 32, 40, 48]
    mc_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
    
    results = []
    
    # Number of repetitions for statistical significance
    n_repetitions = 5
    
    print(f"    Testing convergence for data point {target_point}")
    
    # 1. Trapezoidal method convergence
    print("      Trapezoidal method...")
    for n_samples in trapezoid_samples:
        values = []
        for rep in range(n_repetitions):
            try:
                value = compute_integral_shapley_value(
                    x_train, y_train, x_valid, y_valid, target_point, 
                    clf, final_model, utility_func,
                    method='trapezoid', num_t_samples=n_samples, num_MC=50
                )
                values.append(value)
            except Exception as e:
                print(f"        Error with trapezoid n={n_samples}: {e}")
                continue
        
        if values:
            results.append({
                'method': 'trapezoid',
                'sample_size': n_samples,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'error_vs_true': abs(np.mean(values) - true_value) if true_value is not None else np.nan,
                'n_repetitions': len(values)
            })
    
    # 2. Gaussian quadrature convergence
    print("      Gaussian quadrature...")
    for n_nodes in gaussian_nodes:
        values = []
        for rep in range(n_repetitions):
            try:
                value = compute_integral_shapley_value(
                    x_train, y_train, x_valid, y_valid, target_point,
                    clf, final_model, utility_func,
                    method='gaussian', num_nodes=n_nodes, num_MC=50
                )
                values.append(value)
            except Exception as e:
                print(f"        Error with gaussian n={n_nodes}: {e}")
                continue
        
        if values:
            results.append({
                'method': 'gaussian',
                'sample_size': n_nodes,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'error_vs_true': abs(np.mean(values) - true_value) if true_value is not None else np.nan,
                'n_repetitions': len(values)
            })
    
    # 3. Monte Carlo convergence (traditional)
    print("      Monte Carlo method...")
    for n_mc in mc_samples:
        values = []
        for rep in range(n_repetitions):
            try:
                value = monte_carlo_shapley_value(
                    target_point, x_train, y_train, x_valid, y_valid, 
                    clf, final_model, utility_func, num_samples=n_mc
                )
                values.append(value)
            except Exception as e:
                print(f"        Error with MC n={n_mc}: {e}")
                continue
        
        if values:
            results.append({
                'method': 'monte_carlo',
                'sample_size': n_mc,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'error_vs_true': abs(np.mean(values) - true_value) if true_value is not None else np.nan,
                'n_repetitions': len(values)
            })
    
    return pd.DataFrame(results)


def run_comprehensive_convergence_study():
    """Run convergence study across datasets and methods."""
    
    # Load datasets  
    datasets = {
        'iris': load_iris(),
        'wine': load_wine()
    }
    
    all_results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n=== Convergence study: {dataset_name.upper()} ===")
        
        # Prepare data
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        
        # Train models
        final_model = return_model('LinearSVC')
        final_model.fit(x_train, y_train)
        clf = return_model('LinearSVC')
        
        # For small datasets, compute exact Shapley value as ground truth
        true_value = None
        test_point = 0
        
        if len(x_train) <= 15:  # Only for very small datasets
            print(f"  Computing exact Shapley value for ground truth...")
            try:
                true_value = exact_shapley_value(
                    test_point, x_train, y_train, x_valid, y_valid, 
                    clf, final_model, utility_acc
                )
                print(f"  Exact value: {true_value:.6f}")
            except Exception as e:
                print(f"  Could not compute exact value: {e}")
        
        # Run convergence experiment
        results = run_convergence_experiment(
            x_train, y_train, x_valid, y_valid, test_point,
            clf, final_model, utility_acc, true_value
        )
        
        # Add metadata
        results['dataset'] = dataset_name
        results['dataset_size'] = len(x_train)
        results['true_value'] = true_value
        
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


def analyze_convergence_results(results_df):
    """Analyze and visualize convergence results."""
    
    os.makedirs('../../results/plots', exist_ok=True)
    
    # 1. Convergence curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Convergence Analysis of Integral Shapley Methods', fontsize=16)
    
    datasets = results_df['dataset'].unique()
    methods = results_df['method'].unique()
    colors = sns.color_palette("husl", len(methods))
    
    for idx, dataset in enumerate(datasets):
        if idx >= 2:  # Only plot first 2 datasets
            break
            
        dataset_data = results_df[results_df['dataset'] == dataset]
        true_val = dataset_data['true_value'].iloc[0] if not pd.isna(dataset_data['true_value'].iloc[0]) else None
        
        # Plot 1: Convergence to true value (if available)
        ax1 = axes[idx, 0]
        for method_idx, method in enumerate(methods):
            method_data = dataset_data[dataset_data['method'] == method].sort_values('sample_size')
            
            if len(method_data) > 0:
                ax1.errorbar(method_data['sample_size'], method_data['mean_value'],
                           yerr=method_data['std_value'], label=method, 
                           color=colors[method_idx], marker='o', capsize=3)
        
        if true_val is not None:
            ax1.axhline(y=true_val, color='red', linestyle='--', alpha=0.7, label='True value')
        
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Shapley Value Estimate')
        ax1.set_title(f'{dataset.title()} - Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Error vs sample size (if true value available)
        ax2 = axes[idx, 1]
        if true_val is not None:
            for method_idx, method in enumerate(methods):
                method_data = dataset_data[dataset_data['method'] == method].sort_values('sample_size')
                valid_data = method_data.dropna(subset=['error_vs_true'])
                
                if len(valid_data) > 0:
                    ax2.loglog(valid_data['sample_size'], valid_data['error_vs_true'],
                             label=method, color=colors[method_idx], marker='o')
        
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title(f'{dataset.title()} - Error vs Sample Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Variance analysis
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    # Standard deviation vs sample size
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        plt.loglog(method_data['sample_size'], method_data['std_value'], 
                  'o-', label=method, alpha=0.7)
    
    plt.xlabel('Sample Size')
    plt.ylabel('Standard Deviation')
    plt.title('Variance vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Coefficient of variation
    results_df['cv'] = results_df['std_value'] / np.abs(results_df['mean_value'])
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        valid_data = method_data.dropna(subset=['cv'])
        valid_data = valid_data[np.isfinite(valid_data['cv'])]
        
        if len(valid_data) > 0:
            plt.semilogx(valid_data['sample_size'], valid_data['cv'], 
                        'o-', label=method, alpha=0.7)
    
    plt.xlabel('Sample Size')
    plt.ylabel('Coefficient of Variation')
    plt.title('Relative Variance vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Sample efficiency comparison
    plt.figure(figsize=(10, 6))
    
    # For each method, find sample size needed to achieve certain error thresholds
    error_thresholds = [0.1, 0.05, 0.01, 0.005]
    efficiency_data = []
    
    for dataset in datasets:
        dataset_data = results_df[results_df['dataset'] == dataset]
        true_val = dataset_data['true_value'].iloc[0]
        
        if pd.isna(true_val):
            continue
            
        for method in methods:
            method_data = dataset_data[dataset_data['method'] == method].sort_values('sample_size')
            valid_data = method_data.dropna(subset=['error_vs_true'])
            
            for threshold in error_thresholds:
                # Find first sample size that achieves this error
                below_threshold = valid_data[valid_data['error_vs_true'] <= threshold]
                if len(below_threshold) > 0:
                    min_samples = below_threshold['sample_size'].min()
                    efficiency_data.append({
                        'dataset': dataset,
                        'method': method,
                        'error_threshold': threshold,
                        'min_samples': min_samples
                    })
    
    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create efficiency plot
        for threshold in error_thresholds:
            threshold_data = efficiency_df[efficiency_df['error_threshold'] == threshold]
            pivot_data = threshold_data.pivot_table(
                index='method', columns='dataset', values='min_samples', aggfunc='mean'
            )
            
            if not pivot_data.empty:
                x_pos = np.arange(len(pivot_data.index))
                width = 0.8 / len(error_thresholds)
                offset = (list(error_thresholds).index(threshold) - len(error_thresholds)/2 + 0.5) * width
                
                for col_idx, dataset in enumerate(pivot_data.columns):
                    values = pivot_data[dataset].values
                    plt.bar(x_pos + offset, values, width, 
                           label=f'{dataset} (error ≤ {threshold})', alpha=0.7)
        
        plt.xlabel('Method')
        plt.ylabel('Minimum Sample Size')
        plt.title('Sample Efficiency Comparison')
        plt.xticks(x_pos, pivot_data.index)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/sample_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_convergence_summary(results_df):
    """Print comprehensive convergence analysis summary."""
    
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # 1. Method comparison at fixed sample sizes
    print("\n1. Method Comparison at Fixed Sample Sizes:")
    print("-" * 44)
    
    # Compare methods at similar computational cost
    reference_samples = {'trapezoid': 50, 'gaussian': 32, 'monte_carlo': 5000}
    
    for dataset in results_df['dataset'].unique():
        print(f"\n  {dataset.upper()} Dataset:")
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        for method, target_samples in reference_samples.items():
            method_data = dataset_data[dataset_data['method'] == method]
            # Find closest sample size
            if len(method_data) > 0:
                closest_idx = np.argmin(np.abs(method_data['sample_size'] - target_samples))
                closest_row = method_data.iloc[closest_idx]
                
                print(f"    {method:12s} (n={closest_row['sample_size']:4.0f}): "
                      f"{closest_row['mean_value']:.4f} ± {closest_row['std_value']:.4f}")
    
    # 2. Convergence rates
    print("\n2. Convergence Rate Analysis:")
    print("-" * 30)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        true_val = dataset_data['true_value'].iloc[0]
        
        if pd.isna(true_val):
            print(f"\n  {dataset.upper()}: No ground truth available")
            continue
            
        print(f"\n  {dataset.upper()} (true value: {true_val:.4f}):")
        
        for method in results_df['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method].sort_values('sample_size')
            valid_data = method_data.dropna(subset=['error_vs_true'])
            
            if len(valid_data) >= 3:
                # Fit convergence rate (error ∝ n^(-α))
                log_samples = np.log(valid_data['sample_size'])
                log_errors = np.log(valid_data['error_vs_true'] + 1e-10)  # Avoid log(0)
                
                # Linear regression in log space
                coeff = np.polyfit(log_samples, log_errors, 1)
                convergence_rate = -coeff[0]  # Negative of slope
                
                print(f"    {method:12s}: convergence rate ≈ n^(-{convergence_rate:.2f})")
    
    # 3. Efficiency recommendations
    print("\n3. Efficiency Recommendations:")
    print("-" * 32)
    
    # Find most efficient method for different error tolerances
    error_tolerances = [0.1, 0.05, 0.01]
    
    for tolerance in error_tolerances:
        print(f"\n  For error tolerance ≤ {tolerance}:")
        
        best_methods = {}
        for dataset in results_df['dataset'].unique():
            dataset_data = results_df[results_df['dataset'] == dataset]
            true_val = dataset_data['true_value'].iloc[0]
            
            if pd.isna(true_val):
                continue
                
            min_samples_by_method = {}
            for method in results_df['method'].unique():
                method_data = dataset_data[dataset_data['method'] == method]
                valid_data = method_data.dropna(subset=['error_vs_true'])
                
                below_tolerance = valid_data[valid_data['error_vs_true'] <= tolerance]
                if len(below_tolerance) > 0:
                    min_samples_by_method[method] = below_tolerance['sample_size'].min()
            
            if min_samples_by_method:
                best_method = min(min_samples_by_method, key=min_samples_by_method.get)
                best_samples = min_samples_by_method[best_method]
                best_methods[dataset] = (best_method, best_samples)
        
        for dataset, (best_method, samples) in best_methods.items():
            print(f"    {dataset:8s}: {best_method} (n={samples})")
    
    # 4. Variance analysis
    print("\n4. Variance Analysis:")
    print("-" * 19)
    
    avg_cv_by_method = results_df.groupby('method')['cv'].mean()
    
    print("  Average coefficient of variation:")
    for method, cv in avg_cv_by_method.items():
        if not pd.isna(cv):
            print(f"    {method:12s}: {cv:.3f}")
    
    # Find most stable method
    most_stable = avg_cv_by_method.idxmin()
    print(f"\n  Most stable method: {most_stable}")


def main():
    """Run the complete convergence study."""
    print("Starting Integral Shapley Values Convergence Study")
    print("=" * 52)
    
    # Run the study
    results_df = run_comprehensive_convergence_study()
    
    # Save raw results
    os.makedirs('../../results/csvs', exist_ok=True)
    results_df.to_csv('../../results/csvs/convergence_study_results.csv', index=False)
    print(f"\nRaw results saved to ../../results/csvs/convergence_study_results.csv")
    
    # Analyze results
    analyze_convergence_results(results_df)
    
    # Print summary
    print_convergence_summary(results_df)
    
    print(f"\nConvergence study completed!")
    print(f"Check ../../results/plots/ for visualizations")


if __name__ == "__main__":
    main()