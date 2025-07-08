#!/usr/bin/env python
"""
Integral Shapley Values Computation

This module implements efficient Shapley value computation using integral formulation:
SV_i = ∫_0^1 E[Δ(t,i)] dt

where Δ(t,i) = v(S_t ∪ {i}) - v(S_t) is the marginal contribution of data point i
when added to a random coalition S_t of size determined by t·(N-1) using configurable rounding.

Key advantages:
1. Computational efficiency through smart sampling of t values
2. Exploitation of smoothness properties for better approximation
3. Multiple integration methods (trapezoid, Gaussian quadrature, adaptive)
"""

import argparse
import numpy as np
import random
import pickle
import multiprocessing as mp
from tqdm import tqdm
import itertools
from math import factorial
from typing import Callable, Optional
from scipy.integrate import simpson, fixed_quad
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Import utility functions and model factory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utilities import utility_RKHS, utility_KL, utility_acc, utility_cosine
from utils.model_utils import return_model
from utils.math_utils import compute_coalition_size



def estimate_smoothness(x_train, y_train, x_valid, y_valid, i, clf, final_model, 
                       utility_func, num_probe_points=10, num_MC_probe=20):
    """
    Estimate the smoothness of the integrand E[Δ(t,i)] by computing second differences.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_probe_points: Number of t values to probe for smoothness
        num_MC_probe: Monte Carlo samples per probe point
        
    Returns:
        max_second_diff: Maximum estimated second difference (smoothness indicator)
    """
    total = x_train.shape[0]
    indices = [j for j in range(total) if j != i]
    candidate_x = x_train[indices]
    candidate_y = y_train[indices]
    N = len(candidate_x)
    
    # Sample t values uniformly in (0,1)
    t_values = np.linspace(0.1, 0.9, num_probe_points)
    integrand_values = []
    
    for t in t_values:
        m = max(int(np.floor(t * N)), 1)
        mc_values = []
        
        for _ in range(num_MC_probe):
            sample_indices = random.sample(range(N), m)
            X_sub = candidate_x[sample_indices]
            y_sub = candidate_y[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S = 0.0
                
            X_sub_i = np.vstack([X_sub, x_train[i]])
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand_values.append(np.mean(mc_values))
    
    # Compute second differences to estimate smoothness
    if len(integrand_values) >= 3:
        second_diffs = []
        dt = t_values[1] - t_values[0]
        for j in range(len(integrand_values) - 2):
            second_diff = abs(integrand_values[j+2] - 2*integrand_values[j+1] + integrand_values[j]) / (dt**2)
            second_diffs.append(second_diff)
        return max(second_diffs) if second_diffs else 0.0
    else:
        return 0.0


def compute_integral_shapley_trapezoid(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                     utility_func, num_t_samples=50, num_MC=100, 
                                     rounding_method='probabilistic'):
    """
    Compute Shapley value using trapezoidal integration.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data  
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_t_samples: Number of t values to sample
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    total = x_train.shape[0]
    N = total  # Total number of data points
    rng = np.random.default_rng()  # Random number generator for probabilistic rounding

    # Sample t values uniformly in [0,1]
    t_values = np.linspace(0, 1, num_t_samples, endpoint=True)
    
    integrand = []
    for t in t_values:
        # Use the new coalition size computation with chosen rounding method
        m = compute_coalition_size(t, N, method=rounding_method, rng=rng)
        mc_values = []
        
        for _ in range(num_MC):
            if m == 0:
                # Empty coalition
                X_sub = np.empty((0, x_train.shape[1]))
                y_sub = np.empty(0)
            else:
                # Sample m points from all candidates except target point i
                candidate_indices = [j for j in range(total) if j != i]
                sample_indices = random.sample(candidate_indices, m)
                X_sub = x_train[sample_indices]
                y_sub = y_train[sample_indices]
            
            try:
                util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S = 0.0
                
            X_sub_i = np.vstack([X_sub, x_train[i]]) if m > 0 else x_train[i].reshape(1, -1)
            y_sub_i = np.append(y_sub, y_train[i])
            
            try:
                util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
            except:
                util_S_i = 0.0
                
            mc_values.append(util_S_i - util_S)
        
        integrand.append(np.mean(mc_values))
    
    # Trapezoidal integration
    shapley_value = np.trapezoid(integrand, t_values)
    return shapley_value


def compute_integral_shapley_gaussian(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                    utility_func, num_nodes=32, num_MC=100, 
                                    rounding_method='probabilistic'):
    """
    Compute Shapley value using Gaussian-Legendre quadrature.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index  
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_nodes: Number of Gaussian quadrature nodes
        num_MC: Monte Carlo samples per node
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    total = x_train.shape[0]
    N = total
    rng = np.random.default_rng()

    def integrand(t):
        """Define integrand function for Gaussian quadrature"""
        t_arr = np.atleast_1d(t)
        out = []
        
        for ti in t_arr:
            m = compute_coalition_size(ti, N, method=rounding_method, rng=rng)
            mc_values = []
            
            for _ in range(num_MC):
                if m == 0:
                    X_sub = np.empty((0, x_train.shape[1]))
                    y_sub = np.empty(0)
                else:
                    candidate_indices = [j for j in range(total) if j != i]
                    sample_indices = random.sample(candidate_indices, m)
                    X_sub = x_train[sample_indices]
                    y_sub = y_train[sample_indices]
                
                try:
                    util_S = utility_func(X_sub, y_sub, x_valid, y_valid, clone(clf), final_model)
                except:
                    util_S = 0.0
                    
                X_sub_i = np.vstack([X_sub, x_train[i]]) if m > 0 else x_train[i].reshape(1, -1)
                y_sub_i = np.append(y_sub, y_train[i])
                
                try:
                    util_S_i = utility_func(X_sub_i, y_sub_i, x_valid, y_valid, clone(clf), final_model)
                except:
                    util_S_i = 0.0
                    
                mc_values.append(util_S_i - util_S)
            
            out.append(np.mean(mc_values))
        
        return np.array(out)[0] if np.isscalar(t) else np.array(out)

    # Gaussian-Legendre quadrature
    shapley_value, _ = fixed_quad(integrand, 0.0, 1.0, n=num_nodes)
    return shapley_value


def compute_integral_shapley_adaptive(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                    utility_func, tolerance=1e-4, max_samples=200, num_MC=100,
                                    rounding_method='probabilistic'):
    """
    Compute Shapley value using adaptive sampling based on smoothness detection.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        tolerance: Convergence tolerance
        max_samples: Maximum number of t samples
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    # Start with a coarse estimate
    coarse_estimate = compute_integral_shapley_trapezoid(
        x_train, y_train, x_valid, y_valid, i, clf, final_model, 
        utility_func, num_t_samples=10, num_MC=num_MC, rounding_method=rounding_method
    )
    
    # Progressively refine
    for num_samples in [20, 50, 100, max_samples]:
        fine_estimate = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=num_samples, num_MC=num_MC, rounding_method=rounding_method
        )
        
        if abs(fine_estimate - coarse_estimate) < tolerance:
            return fine_estimate
            
        coarse_estimate = fine_estimate
    
    return coarse_estimate


def monte_carlo_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, 
                            utility_func, num_samples=10000):
    """
    Traditional Monte Carlo estimation of Shapley values for comparison.
    
    Args:
        i: Target data point index
        X_train, y_train: Training data
        x_valid, y_valid: Validation data
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_samples: Number of random permutations
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    n = X_train.shape[0]
    indices = list(range(n))
    contributions = []
    
    for _ in tqdm(range(num_samples), desc=f"MC sampling for data point {i}", leave=False):
        perm = np.random.permutation(indices)
        pos = np.where(perm == i)[0][0]
        S = list(perm[:pos])
        
        X_S = X_train[S]
        y_S = y_train[S]
        
        try:
            v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
        except:
            v_S = 0.0
        
        X_S_i = np.vstack([X_S, X_train[i]]) if len(S) > 0 else X_train[i].reshape(1, -1)
        y_S_i = np.append(y_S, y_train[i])
        
        try:
            v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
        except:
            v_S_i = 0.0
        
        contributions.append(v_S_i - v_S)
    
    return np.mean(contributions)


def exact_shapley_value(i, X_train, y_train, x_valid, y_valid, clf, final_model, utility_func):
    """
    Exact Shapley value computation (for small datasets only).
    
    Args:
        i: Target data point index
        X_train, y_train: Training data
        x_valid, y_valid: Validation data  
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        
    Returns:
        shapley_value: Exact Shapley value for data point i
    """
    n = X_train.shape[0]
    indices = list(range(n))
    indices.remove(i)
    shapley_value = 0.0
    
    for r in tqdm(range(len(indices) + 1), desc=f"Exact Shapley for point {i}"):
        for subset in itertools.combinations(indices, r):
            S = list(subset)
            X_S = X_train[S]
            y_S = y_train[S]
            
            try:
                v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S = 0.0
            
            X_S_i = np.vstack([X_S, X_train[i]]) if len(S) > 0 else X_train[i].reshape(1, -1)
            y_S_i = np.append(y_S, y_train[i])
            
            try:
                v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S_i = 0.0
            
            delta = v_S_i - v_S
            weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            shapley_value += weight * delta
    
    return shapley_value


def compute_integral_shapley_value(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                 utility_func, method='trapezoid', rounding_method='probabilistic', **kwargs):
    """
    Main interface for computing integral Shapley values.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: Integration method ('trapezoid', 'gaussian', 'adaptive', 'monte_carlo')
        rounding_method: How to round coalition sizes ('probabilistic', 'round', 'floor', 'ceil')
        **kwargs: Method-specific parameters
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    if method == 'trapezoid':
        return compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func, 
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'gaussian':
        return compute_integral_shapley_gaussian(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'adaptive':
        return compute_integral_shapley_adaptive(
            x_train, y_train, x_valid, y_valid, i, clf, final_model, utility_func,
            rounding_method=rounding_method, **kwargs
        )
    elif method == 'monte_carlo':
        return monte_carlo_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_shapley_for_params(args):
    """Wrapper function for parallel computation of Shapley values."""
    index, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, method, rounding_method, kwargs = args
    try:
        value = compute_integral_shapley_value(
            x_train, y_train, x_valid, y_valid, index, clf, final_model, 
            utility_func, method=method, rounding_method=rounding_method, **kwargs
        )
        return index, value
    except Exception as e:
        print(f"Error computing Shapley value for index {index}: {str(e)}")
        return index, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Compute Shapley values using integral formulation")
    parser.add_argument("--dataset", type=str, choices=["iris", "wine", "cancer", "synthetic"], 
                       default="iris", help="Dataset to use")
    parser.add_argument("--utility", type=str, choices=["rkhs", "kl", "acc", "cosine"], 
                       default="acc", help="Utility function")
    parser.add_argument("--method", type=str, choices=["trapezoid", "gaussian", "adaptive", "monte_carlo", "exact"],
                       default="trapezoid", help="Integration method")
    parser.add_argument("--clf", choices=["svm", "lr"], default="svm", help="Base classifier")
    parser.add_argument("--num_t_samples", type=int, default=50, help="Number of t samples for integration")
    parser.add_argument("--num_MC", type=int, default=100, help="Monte Carlo samples per t value")
    parser.add_argument("--num_nodes", type=int, default=32, help="Gaussian quadrature nodes")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Convergence tolerance for adaptive method")
    parser.add_argument("--processes", type=int, default=mp.cpu_count(), help="Number of processes")
    parser.add_argument("--all_points", action="store_true", help="Compute for all data points")
    parser.add_argument("--rounding_method", type=str, choices=["probabilistic", "round", "floor", "ceil"],
                       default="probabilistic", help="Method for rounding coalition sizes")
    
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    elif args.dataset == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    elif args.dataset == 'cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

    # Train final model
    if args.clf == "svm":
        final_model = return_model('LinearSVC')
        final_model.fit(x_train, y_train)
        clf = return_model('LinearSVC')
    elif args.clf == "lr":
        final_model = return_model('logistic')
        final_model.fit(x_train, y_train)
        clf = return_model('logistic')

    # Select utility function
    utility_funcs = {
        "rkhs": utility_RKHS,
        "acc": utility_acc,
        "kl": utility_KL,
        "cosine": utility_cosine
    }
    utility_func = utility_funcs[args.utility]

    # Set method parameters
    method_kwargs = {}
    if args.method == 'trapezoid':
        method_kwargs = {'num_t_samples': args.num_t_samples, 'num_MC': args.num_MC}
    elif args.method == 'gaussian':
        method_kwargs = {'num_nodes': args.num_nodes, 'num_MC': args.num_MC}
    elif args.method == 'adaptive':
        method_kwargs = {'tolerance': args.tolerance, 'num_MC': args.num_MC}
    elif args.method == 'monte_carlo':
        method_kwargs = {'num_samples': args.num_MC * args.num_t_samples}

    # Compute Shapley values
    if args.all_points:
        target_indices = list(range(len(x_train)))
    else:
        target_indices = [0]  # Just compute for first data point

    if args.method == "exact":
        # Exact computation (sequential)
        print(f"Computing exact Shapley values for {len(target_indices)} data points...")
        results = {}
        for idx in target_indices:
            value = exact_shapley_value(idx, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func)
            results[idx] = value
        results = {"Exact": np.array([results[i] for i in sorted(target_indices)])}
    else:
        # Integral methods (parallel)
        process_args = []
        for idx in target_indices:
            process_args.append((idx, x_train, y_train, x_valid, y_valid, clf, final_model, 
                               utility_func, args.method, args.rounding_method, method_kwargs))
        
        print(f"Using {args.processes} processes to compute {len(target_indices)} Shapley values with {args.method} method...")
        
        raw_results = {}
        with mp.Pool(processes=args.processes) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_shapley_for_params, process_args),
                                   total=len(process_args), desc=f"Computing {args.method} Shapley values"):
                raw_results[index] = value
        
        results = {args.method.title(): np.array([raw_results[i] for i in sorted(target_indices)])}

    # Print results
    print(f"\nDataset: {args.dataset}")
    print(f"Utility function: {args.utility}")
    print(f"Method: {args.method}")
    
    for method_name, values in results.items():
        valid_values = values[~np.isnan(values.astype(float)) if values.dtype != object else np.ones(len(values), dtype=bool)]
        if len(valid_values) > 0:
            print(f"\n{method_name} Results:")
            print(f"  Mean: {np.mean(valid_values.astype(float)):.6f}")
            print(f"  Max:  {np.max(valid_values.astype(float)):.6f}")
            print(f"  Min:  {np.min(valid_values.astype(float)):.6f}")

    # Save results
    pkl_filename = f"../../results/pickles/{args.clf}_shapley_{args.dataset}_{args.utility}_{args.method}.pkl"
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {pkl_filename}")


if __name__ == "__main__":
    main()