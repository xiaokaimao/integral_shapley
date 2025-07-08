#!/usr/bin/env python
"""
Efficiency Study: Compare computational efficiency of different Shapley value methods

This script compares:
1. Traditional exact computation (for small datasets)
2. Traditional Monte Carlo sampling
3. Integral-based methods (trapezoid, Gaussian quadrature, adaptive)

The goal is to demonstrate the computational advantages of integral methods,
especially for larger datasets where exact computation becomes infeasible.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
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
from utils.utilities import utility_acc
from utils.model_utils import return_model


def time_method(method_func, *args, **kwargs):
    """Time a method and return both result and execution time."""
    start_time = time.time()
    try:
        result = method_func(*args, **kwargs)
        success = True
    except Exception as e:
        result = str(e)
        success = False
    end_time = time.time()
    return result, end_time - start_time, success


def run_efficiency_study():
    """Run comprehensive efficiency study across different methods and datasets."""
    
    # Load datasets
    datasets = {
        'iris': load_iris(),
        'wine': load_wine(),
        'cancer': load_breast_cancer()
    }
    
    # Methods to compare
    methods = {
        'exact': lambda *args, **kwargs: exact_shapley_value(*args, **kwargs),
        'monte_carlo': lambda *args, **kwargs: monte_carlo_shapley_value(*args, **kwargs),
        'integral_trapezoid': lambda *args, **kwargs: compute_integral_shapley_value(
            *args, method='trapezoid', **kwargs
        ),
        'integral_gaussian': lambda *args, **kwargs: compute_integral_shapley_value(
            *args, method='gaussian', **kwargs
        ),
        'integral_adaptive': lambda *args, **kwargs: compute_integral_shapley_value(
            *args, method='adaptive', **kwargs
        )
    }
    
    # Parameters for different methods
    method_params = {
        'exact': {},
        'monte_carlo': {'num_samples': 1000},
        'integral_trapezoid': {'num_t_samples': 20, 'num_MC': 50},
        'integral_gaussian': {'num_nodes': 16, 'num_MC': 50},
        'integral_adaptive': {'tolerance': 1e-3, 'num_MC': 50}
    }
    
    results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n=== Processing {dataset_name.upper()} dataset ===")
        
        # Prepare data
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        
        # Train final model
        final_model = return_model('LinearSVC')
        final_model.fit(x_train, y_train)
        clf = return_model('LinearSVC')
        
        # Test on first few data points
        test_indices = [0, 1, 2] if len(x_train) > 2 else [0]
        
        for i in test_indices:
            print(f"  Testing data point {i}")
            
            for method_name, method_func in methods.items():
                print(f"    Method: {method_name}")
                
                # Skip exact method for large datasets
                if method_name == 'exact' and len(x_train) > 20:
                    print(f"      Skipping exact method for large dataset (n={len(x_train)})")
                    continue
                
                # Prepare arguments based on method type
                if method_name in ['monte_carlo', 'exact']:
                    args = (i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_acc)
                else:
                    args = (x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_acc)
                
                kwargs = method_params[method_name]
                
                # Time the method
                result, exec_time, success = time_method(method_func, *args, **kwargs)
                
                results.append({
                    'dataset': dataset_name,
                    'dataset_size': len(x_train),
                    'data_point': i,
                    'method': method_name,
                    'execution_time': exec_time,
                    'shapley_value': result if success else np.nan,
                    'success': success
                })
                
                print(f"      Time: {exec_time:.3f}s, Value: {result if success else 'Failed'}")
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze and visualize efficiency study results."""
    
    # Create output directory
    os.makedirs('../../results/plots', exist_ok=True)
    
    # 1. Execution time comparison
    plt.figure(figsize=(12, 8))
    
    # Filter successful runs only
    df_success = df[df['success'] == True].copy()
    
    # Box plot of execution times by method
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_success, x='method', y='execution_time')
    plt.xticks(rotation=45)
    plt.title('Execution Time by Method')
    plt.ylabel('Time (seconds)')
    
    # Execution time by dataset size
    plt.subplot(2, 2, 2)
    for method in df_success['method'].unique():
        method_data = df_success[df_success['method'] == method]
        plt.scatter(method_data['dataset_size'], method_data['execution_time'], 
                   label=method, alpha=0.7)
    plt.xlabel('Dataset Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Dataset Size')
    plt.legend()
    plt.yscale('log')
    
    # Average execution time by method and dataset
    plt.subplot(2, 2, 3)
    avg_times = df_success.groupby(['dataset', 'method'])['execution_time'].mean().reset_index()
    pivot_data = avg_times.pivot(index='dataset', columns='method', values='execution_time')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Execution Time (seconds)')
    
    # Shapley value consistency across methods
    plt.subplot(2, 2, 4)
    if len(df_success) > 0:
        # Check correlation between methods for same data points
        pivot_values = df_success.pivot_table(
            index=['dataset', 'data_point'], 
            columns='method', 
            values='shapley_value'
        ).reset_index()
        
        # Compare integral methods with monte carlo as baseline
        if 'monte_carlo' in pivot_values.columns and 'integral_trapezoid' in pivot_values.columns:
            valid_comparisons = pivot_values.dropna(subset=['monte_carlo', 'integral_trapezoid'])
            if len(valid_comparisons) > 0:
                plt.scatter(valid_comparisons['monte_carlo'], valid_comparisons['integral_trapezoid'])
                plt.xlabel('Monte Carlo Shapley Value')
                plt.ylabel('Integral Trapezoid Shapley Value')
                plt.title('Method Consistency')
                # Add diagonal line
                min_val = min(plt.xlim()[0], plt.ylim()[0])
                max_val = max(plt.xlim()[1], plt.ylim()[1])
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/efficiency_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary statistics
    print("\n=== EFFICIENCY STUDY SUMMARY ===")
    print("\nSuccess rates by method:")
    success_rates = df.groupby('method')['success'].mean()
    for method, rate in success_rates.items():
        print(f"  {method}: {rate:.1%}")
    
    print("\nAverage execution times (successful runs only):")
    avg_times = df_success.groupby('method')['execution_time'].agg(['mean', 'std'])
    for method, stats in avg_times.iterrows():
        print(f"  {method}: {stats['mean']:.3f} ± {stats['std']:.3f} seconds")
    
    print("\nSpeedup compared to Monte Carlo:")
    mc_time = avg_times.loc['monte_carlo', 'mean'] if 'monte_carlo' in avg_times.index else None
    if mc_time is not None:
        for method, stats in avg_times.iterrows():
            if method != 'monte_carlo':
                speedup = mc_time / stats['mean']
                print(f"  {method}: {speedup:.1f}x faster")
    
    # 3. Scalability analysis
    print("\nScalability analysis:")
    for dataset in df_success['dataset'].unique():
        dataset_data = df_success[df_success['dataset'] == dataset]
        dataset_size = dataset_data['dataset_size'].iloc[0]
        print(f"\n  {dataset.upper()} (n={dataset_size}):")
        
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]
            avg_time = method_data['execution_time'].mean()
            print(f"    {method}: {avg_time:.3f}s average")


def main():
    """Run the complete efficiency study."""
    print("Starting Integral Shapley Values Efficiency Study")
    print("=" * 50)
    
    # Run the study
    results_df = run_efficiency_study()
    
    # Save raw results
    os.makedirs('../../results/csvs', exist_ok=True)
    results_df.to_csv('../../results/csvs/efficiency_study_results.csv', index=False)
    print(f"\nRaw results saved to ../../results/csvs/efficiency_study_results.csv")
    
    # Analyze results
    analyze_results(results_df)
    
    print("\nEfficiency study completed!")
    print("Check ../../results/plots/efficiency_study.png for visualizations")


if __name__ == "__main__":
    main()