#!/usr/bin/env python
"""
Smoothness Analysis: Study the smoothness properties of the integrand E[Δ(t,i)]

This script analyzes the smoothness characteristics of the Shapley value integrand
across different datasets, utility functions, and data points. Understanding
smoothness helps optimize sampling strategies and integration methods.

Key analyses:
1. Integrand curve visualization
2. Smoothness metrics (second derivatives, variation)
3. Optimal sampling density estimation
4. Comparison across different utility functions
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
from core.integral_shapley import estimate_smoothness
from utils.utilities import utility_acc, utility_RKHS, utility_KL, utility_cosine
from utils.model_utils import return_model


def compute_integrand_curve(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                          utility_func, num_t_samples=100, num_MC=50):
    """
    Compute the integrand E[Δ(t,i)] across a range of t values.
    
    Returns:
        t_values: Array of t values
        integrand_values: Corresponding integrand values
        std_values: Standard deviations (uncertainty estimates)
    """
    total = x_train.shape[0]
    indices = [j for j in range(total) if j != i]
    candidate_x = x_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)
    
    # Sample t values uniformly
    t_values = np.linspace(0.01, 0.99, num_t_samples)  # Avoid endpoints
    integrand_values = []
    std_values = []
    
    for t in t_values:
        m = max(int(np.floor(t * N)), 1)
        mc_values = []
        
        for _ in range(num_MC):
            sample_indices = np.random.choice(N, size=m, replace=False)
            X_sub = candidate_x[sample_indices]
            y_sub = candidate_y[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clf, final_model)
            except:
                util_S = 0.0
                
            X_sub_i = np.vstack([X_sub, x_train[i]])
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clf, final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand_values.append(np.mean(mc_values))
        std_values.append(np.std(mc_values))
    
    return t_values, np.array(integrand_values), np.array(std_values)


def analyze_smoothness_metrics(t_values, integrand_values):
    """
    Compute various smoothness metrics for the integrand.
    
    Returns:
        metrics: Dictionary of smoothness metrics
    """
    # First and second derivatives (finite differences)
    dt = t_values[1] - t_values[0]
    first_deriv = np.gradient(integrand_values, dt)
    second_deriv = np.gradient(first_deriv, dt)
    
    # Smoothness metrics
    metrics = {
        'total_variation': np.sum(np.abs(np.diff(integrand_values))),
        'max_first_deriv': np.max(np.abs(first_deriv)),
        'max_second_deriv': np.max(np.abs(second_deriv)),
        'mean_second_deriv': np.mean(np.abs(second_deriv)),
        'std_second_deriv': np.std(second_deriv),
        'smoothness_score': 1.0 / (1.0 + np.mean(np.abs(second_deriv))),  # Higher = smoother
        'monotonicity': np.sum(np.diff(integrand_values) > 0) / len(np.diff(integrand_values))
    }
    
    return metrics, first_deriv, second_deriv


def run_smoothness_study():
    """Run comprehensive smoothness analysis across datasets and utility functions."""
    
    # Load datasets
    datasets = {
        'iris': load_iris(),
        'wine': load_wine()
    }
    
    # Utility functions to test
    utility_functions = {
        'accuracy': utility_acc,
        'rkhs': utility_RKHS,
        'kl_divergence': utility_KL,
        'cosine': utility_cosine
    }
    
    results = []
    curve_data = []
    
    for dataset_name, data in datasets.items():
        print(f"\n=== Analyzing {dataset_name.upper()} dataset ===")
        
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
        
        # Test on a few representative data points
        test_indices = [0, len(x_train)//4, len(x_train)//2] if len(x_train) > 2 else [0]
        
        for i in test_indices[:2]:  # Limit to 2 points for efficiency
            print(f"  Analyzing data point {i}")
            
            for utility_name, utility_func in utility_functions.items():
                print(f"    Utility: {utility_name}")
                
                try:
                    # Compute integrand curve
                    t_vals, integrand_vals, std_vals = compute_integrand_curve(
                        x_train, y_train, x_valid, y_valid, i, clf, final_model,
                        utility_func, num_t_samples=50, num_MC=30
                    )
                    
                    # Analyze smoothness
                    metrics, first_deriv, second_deriv = analyze_smoothness_metrics(t_vals, integrand_vals)
                    
                    # Store results
                    result = {
                        'dataset': dataset_name,
                        'dataset_size': len(x_train),
                        'data_point': i,
                        'utility_function': utility_name,
                        **metrics
                    }
                    results.append(result)
                    
                    # Store curve data for visualization
                    for j, (t, integrand, std) in enumerate(zip(t_vals, integrand_vals, std_vals)):
                        curve_data.append({
                            'dataset': dataset_name,
                            'data_point': i,
                            'utility_function': utility_name,
                            't': t,
                            'integrand': integrand,
                            'std': std,
                            'first_deriv': first_deriv[j] if j < len(first_deriv) else np.nan,
                            'second_deriv': second_deriv[j] if j < len(second_deriv) else np.nan
                        })
                        
                    print(f"      Smoothness score: {metrics['smoothness_score']:.3f}")
                    
                except Exception as e:
                    print(f"      Error: {e}")
    
    return pd.DataFrame(results), pd.DataFrame(curve_data)


def visualize_smoothness_analysis(results_df, curve_df):
    """Create comprehensive visualizations of smoothness analysis."""
    
    os.makedirs('../../results/plots', exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Integrand curves for different utility functions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Integrand Curves E[Δ(t,i)] by Utility Function', fontsize=16)
    
    utility_funcs = curve_df['utility_function'].unique()
    datasets = curve_df['dataset'].unique()
    
    for idx, utility in enumerate(utility_funcs[:4]):  # Show first 4 utilities
        ax = axes[idx//2, idx%2]
        
        for dataset in datasets:
            data_subset = curve_df[
                (curve_df['utility_function'] == utility) & 
                (curve_df['dataset'] == dataset) &
                (curve_df['data_point'] == 0)  # Focus on first data point
            ]
            
            if len(data_subset) > 0:
                ax.plot(data_subset['t'], data_subset['integrand'], 
                       label=f'{dataset}', linewidth=2, alpha=0.8)
                # Add uncertainty bands
                ax.fill_between(data_subset['t'], 
                              data_subset['integrand'] - data_subset['std'],
                              data_subset['integrand'] + data_subset['std'],
                              alpha=0.2)
        
        ax.set_xlabel('t (coalition proportion)')
        ax.set_ylabel('E[Δ(t,i)]')
        ax.set_title(f'{utility.replace("_", " ").title()} Utility')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/integrand_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Smoothness metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Smoothness Metrics Comparison', fontsize=16)
    
    metrics_to_plot = ['smoothness_score', 'total_variation', 'max_second_deriv', 
                      'monotonicity', 'mean_second_deriv', 'std_second_deriv']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx//3, idx%3]
        
        # Box plot by utility function
        sns.boxplot(data=results_df, x='utility_function', y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/smoothness_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of smoothness scores
    plt.figure(figsize=(10, 6))
    
    # Create pivot table for heatmap
    smoothness_pivot = results_df.pivot_table(
        index='utility_function', 
        columns='dataset', 
        values='smoothness_score',
        aggfunc='mean'
    )
    
    sns.heatmap(smoothness_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Smoothness Score (higher = smoother)'})
    plt.title('Average Smoothness Score by Utility Function and Dataset')
    plt.tight_layout()
    plt.savefig('../../results/plots/smoothness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Second derivative analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot second derivatives for one representative case
    iris_acc = curve_df[
        (curve_df['dataset'] == 'iris') & 
        (curve_df['utility_function'] == 'accuracy') &
        (curve_df['data_point'] == 0)
    ]
    
    if len(iris_acc) > 0:
        axes[0].plot(iris_acc['t'], iris_acc['integrand'], 'b-', linewidth=2, label='Integrand')
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('E[Δ(t,i)]', color='b')
        axes[0].tick_params(axis='y', labelcolor='b')
        axes[0].grid(True, alpha=0.3)
        
        # Plot second derivative on secondary axis
        ax2 = axes[0].twinx()
        ax2.plot(iris_acc['t'], iris_acc['second_deriv'], 'r--', alpha=0.7, label='Second Derivative')
        ax2.set_ylabel('Second Derivative', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        axes[0].set_title('Integrand and Second Derivative (Iris, Accuracy)')
    
    # Distribution of second derivatives
    axes[1].hist(curve_df['second_deriv'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Second Derivative Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Second Derivatives')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/second_derivative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_statistics(results_df):
    """Print comprehensive summary statistics."""
    
    print("\n" + "="*60)
    print("SMOOTHNESS ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n1. Overall Smoothness by Utility Function:")
    print("-" * 45)
    smoothness_by_utility = results_df.groupby('utility_function')['smoothness_score'].agg(['mean', 'std', 'min', 'max'])
    for utility, stats in smoothness_by_utility.iterrows():
        print(f"  {utility:15s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
    
    print("\n2. Total Variation by Utility Function:")
    print("-" * 42)
    variation_by_utility = results_df.groupby('utility_function')['total_variation'].agg(['mean', 'std'])
    for utility, stats in variation_by_utility.iterrows():
        print(f"  {utility:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\n3. Monotonicity Analysis:")
    print("-" * 25)
    monotonicity_by_utility = results_df.groupby('utility_function')['monotonicity'].mean()
    for utility, monotonicity in monotonicity_by_utility.items():
        print(f"  {utility:15s}: {monotonicity:.1%} monotonic")
    
    print("\n4. Dataset Comparison:")
    print("-" * 20)
    dataset_comparison = results_df.groupby('dataset')[['smoothness_score', 'total_variation']].mean()
    for dataset, stats in dataset_comparison.iterrows():
        print(f"  {dataset:8s}: Smoothness={stats['smoothness_score']:.3f}, "
              f"Variation={stats['total_variation']:.4f}")
    
    # Recommendations
    print("\n5. Recommendations for Integration:")
    print("-" * 35)
    
    # Find smoothest utility function
    smoothest_utility = results_df.groupby('utility_function')['smoothness_score'].mean().idxmax()
    smoothest_score = results_df.groupby('utility_function')['smoothness_score'].mean().max()
    
    print(f"  • Smoothest utility function: {smoothest_utility} (score: {smoothest_score:.3f})")
    print(f"    → Recommended: Use fewer t samples for {smoothest_utility}")
    
    # Find least smooth utility function
    roughest_utility = results_df.groupby('utility_function')['smoothness_score'].mean().idxmin()
    roughest_score = results_df.groupby('utility_function')['smoothness_score'].mean().min()
    
    print(f"  • Least smooth utility function: {roughest_utility} (score: {roughest_score:.3f})")
    print(f"    → Recommended: Use more t samples or adaptive sampling for {roughest_utility}")
    
    # Sampling recommendations
    print(f"\n  • General sampling recommendations:")
    avg_smoothness = results_df['smoothness_score'].mean()
    if avg_smoothness > 0.8:
        print(f"    → High smoothness detected: 20-30 t samples should suffice")
    elif avg_smoothness > 0.6:
        print(f"    → Moderate smoothness: 30-50 t samples recommended")
    else:
        print(f"    → Low smoothness: 50+ t samples or adaptive methods recommended")


def main():
    """Run the complete smoothness analysis."""
    print("Starting Integral Shapley Values Smoothness Analysis")
    print("=" * 55)
    
    # Run the analysis
    results_df, curve_df = run_smoothness_study()
    
    # Save raw results
    os.makedirs('../../results/csvs', exist_ok=True)
    results_df.to_csv('../../results/csvs/smoothness_analysis_results.csv', index=False)
    curve_df.to_csv('../../results/csvs/integrand_curves_data.csv', index=False)
    print(f"\nRaw results saved to ../../results/csvs/")
    
    # Create visualizations
    visualize_smoothness_analysis(results_df, curve_df)
    
    # Print summary
    print_summary_statistics(results_df)
    
    print(f"\nSmoothnesh analysis completed!")
    print(f"Check ../../results/plots/ for visualizations")


if __name__ == "__main__":
    main()