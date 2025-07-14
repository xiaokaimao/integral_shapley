#!/usr/bin/env python
"""
Simple MSE Comparison: Fixed Sample Budget k

Direct comparison using existing parallel functions from integral_shapley.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing as mp
from tqdm import tqdm
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.integral_shapley import (
    monte_carlo_shapley_value,
    stratified_shapley_value,
    compute_shapley_for_params
)
from utils.utilities import utility_acc
from utils.model_utils import return_model


def compute_mc_single(args):
    """Compute Monte Carlo Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_samples = args
    return i, monte_carlo_shapley_value(
        i, x_train, y_train, x_valid, y_valid, clf, final_model, 
        utility_func, num_samples=num_samples
    )


def compute_stratified_single(args):
    """Compute Stratified Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_mc = args
    return i, stratified_shapley_value(
        i, x_train, y_train, x_valid, y_valid, clf, final_model, 
        utility_func, num_MC=num_mc
    )


def main():
    """Simple MSE comparison with fixed sample budgets."""
    
    # Sample budgets to test
    k_values = [200, 500, 1000, 2000, 5000, 10000]
    k_ground_truth = 100000
    
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # Train models
    final_model = return_model('SVC')
    final_model.fit(x_train, y_train)
    clf = return_model('SVC')
    
    n_points = len(x_train)
    print(f"Dataset: iris, {n_points} points")
    
    # 1. Compute ground truth with high-budget Monte Carlo (parallel)
    print(f"\n1. Computing ground truth (k={k_ground_truth} Monte Carlo, parallel)...")
    
    # Prepare arguments for parallel processing
    process_args = []
    for i in range(n_points):
        process_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                           utility_acc, k_ground_truth))
    
    # Parallel computation
    ground_truth_dict = {}
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for index, value in tqdm(pool.imap_unordered(compute_mc_single, process_args),
                               total=len(process_args), desc="Computing ground truth"):
            ground_truth_dict[index] = value
    
    ground_truth = np.array([ground_truth_dict[i] for i in range(n_points)])
    
    # 2. Test different methods with different k values
    results = []
    
    for k in k_values:
        print(f"\n2. Testing k={k}")
        
        # Monte Carlo: k samples (parallel)
        print("   Monte Carlo (parallel)...")
        mc_args = []
        for i in range(n_points):
            mc_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                           utility_acc, k))
        
        mc_dict = {}
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_mc_single, mc_args),
                                   total=len(mc_args), desc="Monte Carlo"):
                mc_dict[index] = value
        
        mc_values = np.array([mc_dict[i] for i in range(n_points)])
        mc_mse = np.mean((mc_values - ground_truth) ** 2)
        results.append({'k': k, 'method': 'monte_carlo', 'mse': mc_mse})
        print(f"      MSE: {mc_mse:.6f}")
        
        # Stratified: mc = k/N (parallel)
        print("   Stratified (parallel)...")
        mc_per_coalition = max(1, k // n_points)
        
        stratified_args = []
        for i in range(n_points):
            stratified_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                                  utility_acc, mc_per_coalition))
        
        stratified_dict = {}
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_stratified_single, stratified_args),
                                   total=len(stratified_args), desc="Stratified"):
                stratified_dict[index] = value
        
        stratified_values = np.array([stratified_dict[i] for i in range(n_points)])
        stratified_mse = np.mean((stratified_values - ground_truth) ** 2)
        results.append({'k': k, 'method': 'stratified', 'mse': stratified_mse})
        print(f"      MSE: {stratified_mse:.6f}")
        
        # Simpson: t*mc = k (parallel, use existing function)
        print("   Simpson (parallel)...")
        # t_samples = max(5, min(50, int(np.sqrt(k))))
        t_samples = 21
        # Ensure odd number for Simpson's rule
        if t_samples % 2 == 0:
            t_samples += 1
        mc_samples = k // t_samples
        print(f"      Using t={t_samples}, mc={mc_samples}")
        
        # Use existing parallel function from integral_shapley.py
        simpson_args = []
        for i in range(n_points):
            method_kwargs = {'num_t_samples': t_samples, 'num_MC': mc_samples}
            simpson_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                               utility_acc, 'simpson', 'probabilistic', method_kwargs))
        
        simpson_dict = {}
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_shapley_for_params, simpson_args),
                                   total=len(simpson_args), desc="Simpson"):
                simpson_dict[index] = value
        
        simpson_values = np.array([simpson_dict[i] for i in range(n_points)])
        simpson_mse = np.mean((simpson_values - ground_truth) ** 2)
        results.append({'k': k, 'method': 'simpson', 'mse': simpson_mse})
        print(f"      MSE: {simpson_mse:.6f}")
    
    # 3. Create results DataFrame and plot
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    methods = results_df['method'].unique()
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        plt.plot(method_data['k'], method_data['mse'], 'o-', label=method, linewidth=2, markersize=6)
    
    plt.xlabel('Sample Budget k')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Sample Budget (Fair Comparison)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Save results
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/csvs', exist_ok=True)
    
    plt.savefig('results/plots/simple_mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results_df.to_csv('results/csvs/simple_mse_comparison.csv', index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("SIMPLE MSE COMPARISON SUMMARY")
    print("="*50)
    
    for k in k_values:
        print(f"\nk = {k}:")
        k_data = results_df[results_df['k'] == k].sort_values('mse')
        for _, row in k_data.iterrows():
            print(f"  {row['method']:12s}: {row['mse']:.6f}")
    
    print(f"\nGround truth computed with {k_ground_truth} Monte Carlo samples")
    print("Results saved to results")


if __name__ == "__main__":
    main()