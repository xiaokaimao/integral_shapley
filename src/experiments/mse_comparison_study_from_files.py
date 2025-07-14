#!/usr/bin/env python
"""
MSE Comparison Study From Files: Load and Compare Precomputed Shapley Values

This script loads precomputed Shapley values from pickle files and compares:
1. Integral Shapley Values (our method) with different sampling configurations
2. Exact Shapley Values (ground truth) 
3. Stratified Sampling results (if available)

Key analyses:
1. MSE comparison across different sampling configurations
2. Bias-Variance decomposition
3. Efficiency analysis based on sampling budget
4. Statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
import re
from scipy.stats import ttest_ind
from collections import defaultdict

def load_shapley_files(results_dir="../../results/pickles/"):
    """
    Load all Shapley value files from the results directory.
    
    Args:
        results_dir: Directory containing pickle files
        
    Returns:
        dict: Dictionary with file info and loaded data
    """
    files_data = {}
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    for file_path in pickle_files:
        filename = os.path.basename(file_path)
        
        # Parse filename to extract parameters
        # Format: {clf}_shapley_{dataset}_{utility}_{method}_t{num_t}_mc{num_mc}.pkl
        # or older format: {clf}_shapley_{dataset}_{utility}_{method}.pkl
        
        pattern = r'(\w+)_shapley_(\w+)_(\w+)_(\w+)(?:_t(\d+)_mc(\d+))?\.pkl'
        match = re.match(pattern, filename)
        
        if match:
            clf, dataset, utility, method, num_t, num_mc = match.groups()
            
            # Load data
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                file_info = {
                    'filename': filename,
                    'path': file_path,
                    'clf': clf,
                    'dataset': dataset,
                    'utility': utility,
                    'method': method,
                    'num_t_samples': int(num_t) if num_t else None,
                    'num_mc_samples': int(num_mc) if num_mc else None,
                    'data': data
                }
                
                files_data[filename] = file_info
                print(f"Loaded: {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return files_data

def extract_shapley_values(files_data):
    """
    Extract Shapley values from loaded data for comparison.
    
    Args:
        files_data: Dictionary of loaded file data
        
    Returns:
        dict: Organized Shapley values by dataset and method
    """
    shapley_data = defaultdict(lambda: defaultdict(list))
    
    for filename, file_info in files_data.items():
        data = file_info['data']
        dataset = file_info['dataset']
        method = file_info['method']
        
        # Extract Shapley values based on data structure
        if isinstance(data, dict):
            if 'shapley_values' in data:
                values = data['shapley_values']
            elif 'results' in data:
                values = data['results']
            else:
                # Try to find array-like values
                values = next((v for v in data.values() if isinstance(v, (list, np.ndarray))), None)
        elif isinstance(data, (list, np.ndarray)):
            values = data
        else:
            print(f"Unknown data structure in {filename}")
            continue
        
        if values is not None:
            shapley_data[dataset][method].append({
                'values': np.array(values),
                'file_info': file_info
            })
    
    return shapley_data

def compute_mse_from_files(shapley_data):
    """
    Compute MSE comparison from loaded Shapley values.
    
    Args:
        shapley_data: Dictionary of Shapley values by dataset and method
        
    Returns:
        pd.DataFrame: MSE comparison results
    """
    results = []
    
    for dataset, methods in shapley_data.items():
        print(f"\nProcessing dataset: {dataset}")
        
        # Find exact/ground truth values
        exact_values = None
        if 'exact' in methods:
            exact_values = methods['exact'][0]['values']
            print(f"  Found exact values: {len(exact_values)} points")
        else:
            print(f"  No exact values found for {dataset}")
            continue
        
        # Compare other methods against exact values
        for method_name, method_data in methods.items():
            if method_name == 'exact':
                continue
                
            print(f"  Comparing method: {method_name}")
            
            for data_entry in method_data:
                estimated_values = data_entry['values']
                file_info = data_entry['file_info']
                
                # Ensure same length
                min_len = min(len(exact_values), len(estimated_values))
                exact_subset = exact_values[:min_len]
                estimated_subset = estimated_values[:min_len]
                
                # Compute MSE
                mse = np.mean((estimated_subset - exact_subset) ** 2)
                
                # Extract sampling info
                num_t = file_info.get('num_t_samples', 'unknown')
                num_mc = file_info.get('num_mc_samples', 'unknown')
                total_samples = num_t * num_mc if (num_t != 'unknown' and num_mc != 'unknown') else 'unknown'
                
                results.append({
                    'dataset': dataset,
                    'method': method_name,
                    'filename': file_info['filename'],
                    'num_t_samples': num_t,
                    'num_mc_samples': num_mc,
                    'total_samples': total_samples,
                    'mse': mse,
                    'mean_estimate': np.mean(estimated_subset),
                    'mean_exact': np.mean(exact_subset),
                    'n_points': min_len
                })
    
    return pd.DataFrame(results)

def analyze_mse_from_files(results_df):
    """Analyze and visualize MSE results from files."""
    
    os.makedirs('../../results/plots', exist_ok=True)
    
    # Filter out unknown sampling configurations
    valid_results = results_df[results_df['total_samples'] != 'unknown'].copy()
    
    if len(valid_results) == 0:
        print("No valid sampling configurations found for analysis")
        return
    
    # Convert to numeric
    valid_results['total_samples'] = pd.to_numeric(valid_results['total_samples'])
    
    # 1. MSE vs Total Samples
    plt.figure(figsize=(12, 8))
    
    datasets = valid_results['dataset'].unique()
    methods = valid_results['method'].unique()
    
    for i, dataset in enumerate(datasets):
        plt.subplot(2, 2, i+1)
        
        dataset_data = valid_results[valid_results['dataset'] == dataset]
        
        for method in methods:
            method_data = dataset_data[dataset_data['method'] == method]
            if len(method_data) > 0:
                plt.scatter(method_data['total_samples'], method_data['mse'], 
                          label=method, alpha=0.7, s=60)
        
        plt.xlabel('Total Samples (t_samples × mc_samples)')
        plt.ylabel('Mean Squared Error')
        plt.title(f'{dataset.upper()} - MSE vs Sample Budget')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/mse_from_files.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Sampling Configuration Analysis
    plt.figure(figsize=(12, 8))
    
    for i, dataset in enumerate(datasets):
        plt.subplot(2, 2, i+1)
        
        dataset_data = valid_results[valid_results['dataset'] == dataset]
        
        # Create scatter plot with t_samples and mc_samples as dimensions
        scatter = plt.scatter(dataset_data['num_t_samples'], dataset_data['num_mc_samples'], 
                            c=dataset_data['mse'], cmap='viridis', s=100, alpha=0.7)
        
        plt.xlabel('Number of t samples')
        plt.ylabel('Number of MC samples per t')
        plt.title(f'{dataset.upper()} - Sampling Configuration vs MSE')
        plt.colorbar(scatter, label='MSE')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/sampling_config_from_files.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_mse_summary_from_files(results_df):
    """Print summary of MSE comparison from files."""
    
    print("\n" + "="*70)
    print("MSE COMPARISON FROM FILES SUMMARY")
    print("="*70)
    
    # 1. Available files summary
    print("\n1. Available Files:")
    print("-" * 20)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        print(f"\n  {dataset.upper()}:")
        
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]
            print(f"    {method}: {len(method_data)} configurations")
    
    # 2. MSE statistics
    print("\n2. MSE Statistics:")
    print("-" * 20)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        print(f"\n  {dataset.upper()}:")
        
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]
            if len(method_data) > 0:
                print(f"    {method:15s}: MSE={method_data['mse'].mean():.6f} ± {method_data['mse'].std():.6f}")
    
    # 3. Best configurations
    print("\n3. Best Configurations (Lowest MSE):")
    print("-" * 42)
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        best_config = dataset_data.loc[dataset_data['mse'].idxmin()]
        
        print(f"\n  {dataset.upper()}:")
        print(f"    Method: {best_config['method']}")
        print(f"    MSE: {best_config['mse']:.6f}")
        print(f"    Samples: t={best_config['num_t_samples']}, mc={best_config['num_mc_samples']}")
        print(f"    File: {best_config['filename']}")

def main():
    """Run MSE comparison study from precomputed files."""
    
    print("MSE Comparison Study: Loading from Precomputed Files")
    print("=" * 55)
    
    # Load all Shapley value files
    print("\nLoading Shapley value files...")
    files_data = load_shapley_files()
    
    if not files_data:
        print("No Shapley value files found!")
        return
    
    print(f"Loaded {len(files_data)} files")
    
    # Extract Shapley values
    print("\nExtracting Shapley values...")
    shapley_data = extract_shapley_values(files_data)
    
    # Compute MSE comparisons
    print("\nComputing MSE comparisons...")
    results_df = compute_mse_from_files(shapley_data)
    
    if len(results_df) == 0:
        print("No valid comparisons could be made!")
        return
    
    # Save results
    os.makedirs('../../results/csvs', exist_ok=True)
    results_df.to_csv('../../results/csvs/mse_comparison_from_files.csv', index=False)
    print(f"\nResults saved to ../../results/csvs/mse_comparison_from_files.csv")
    
    # Analyze results
    analyze_mse_from_files(results_df)
    
    # Print summary
    print_mse_summary_from_files(results_df)
    
    print("\nMSE comparison from files completed!")
    print("Check ../../results/plots/ for visualizations")

if __name__ == "__main__":
    main()