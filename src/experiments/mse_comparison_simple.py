#!/usr/bin/env python
"""
Simple MARE Comparison: Fixed Sample Budget k

Direct comparison using existing parallel functions from integral_shapley.py
MARE = Mean Absolute Relative Error = (1/n) * sum(|predicted - ground_truth| / |ground_truth|)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle
import multiprocessing as mp
from tqdm import tqdm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sklearn.dummy import DummyClassifier

from src.core.shapley_interface import (
    monte_carlo_shapley_value,
    stratified_shapley_value,
    compute_integral_shapley_trapezoid,
    compute_integral_shapley_simpson,
    compute_shapley_for_params,
    compute_shapley_for_params_with_budget,
    compute_integral_shapley_value_with_budget,
    visualize_smart_adaptive_sampling,
    integral_cc_sparse_all_players_parallel,
    cc_shapley_parallel,
    # cc_shapley_integral_layer_parallel,
    # compute_integral_shapley_importance_sampling,
    # compute_integral_shapley_sparse_residual
)
from src.core.adaptive_methods import compute_integral_cc_smart_adaptive
from src.utils.utilities import utility_acc, utility_voting_game, utility_airport_game
from src.utils.model_utils import return_model
from src.utils.data_utils import load_dataset
from src.utils.voting_utils import compute_exact_voting_shapley, get_voting_weights
from src.utils.airport_utils import compute_exact_airport_shapley, get_airport_costs

def compute_mc_single(args):
    """Compute Monte Carlo Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_samples = args
    try:
        return i, monte_carlo_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, 
            utility_func, num_samples=num_samples
        )
    except Exception as e:
        print(f"Error computing MC for point {i}: {e}")
        return i, np.nan


def compute_stratified_single(args):
    """Compute Stratified Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_mc = args
    try:
        return i, stratified_shapley_value(
            i, x_train, y_train, x_valid, y_valid, clf, final_model, 
            utility_func, num_MC=num_mc
        )
    except Exception as e:
        print(f"Error computing Stratified for point {i}: {e}")
        return i, np.nan


def compute_trapezoid_single(args):
    """Compute Trapezoid Integration Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_t_samples, num_MC = args
    try:
        return i, compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=num_t_samples, num_MC=num_MC, 
            rounding_method='probabilistic'
        )
    except Exception as e:
        print(f"Error computing Trapezoid for point {i}: {e}")
        return i, np.nan


def compute_simpson_single(args):
    """Compute Simpson Integration Shapley value for a single point."""
    i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_t_samples, num_MC = args
    try:
        return i, compute_integral_shapley_simpson(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=num_t_samples, num_MC=num_MC, 
            rounding_method='probabilistic'
        )
    except Exception as e:
        print(f"Error computing Simpson for point {i}: {e}")
        return i, np.nan



# def compute_sparse_residual_single(args):
#     """Compute Sparse Residual Shapley value for a single point."""
#     i, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_nodes, num_MC_per_node, residual_samples = args
#     try:
#         result = compute_integral_shapley_sparse_residual(
#             x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_nodes=num_nodes, num_MC_per_node=num_MC_per_node, 
#             residual_samples=residual_samples, return_detailed_info=False
#         )
#         return i, result
#     except Exception as e:
#         print(f"Error computing Sparse Residual for point {i}: {e}")
#         return i, np.nan


# def compute_cc_integral_values(x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, K):
#     """Compute CC Integral Shapley values for all points using parallel method."""
#     try:
#         cc_integral_values = cc_shapley_integral_layer_parallel(
#             x_train, y_train, x_valid, y_valid, clf, final_model, 
#             utility_func, K=K
#         )
#         return cc_integral_values
#     except Exception as e:
#         print(f"Error computing CC Integral values: {e}")
#         return np.full(len(x_train), np.nan)


def compute_cc_values(x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, num_mc):
    """Compute CC Shapley values for all points using parallel method."""
    try:
        cc_values = cc_shapley_parallel(
            x_train, y_train, x_valid, y_valid, clf, final_model, 
            utility_func, num_MC=num_mc, num_processes=args.processes
        )
        return cc_values
    except Exception as e:
        print(f"Error computing CC values: {e}")
        return np.full(len(x_train), np.nan)


def compute_mare(predicted, ground_truth, epsilon=1e-8):
    """
    计算平均相对误差 (MARE - Mean Absolute Relative Error)
    
    Args:
        predicted: 预测的Shapley值
        ground_truth: 真实的Shapley值
        epsilon: 避免除零的最小阈值
        
    Returns:
        mare: 平均相对误差 (0到1之间的值)
    """
    # 处理接近零的ground truth值
    mask = np.abs(ground_truth) >= epsilon
    
    if np.sum(mask) == 0:
        print(f"Warning: All ground truth values are near zero (< {epsilon})")
        return np.nan
    
    # 只计算非零值的相对误差
    relative_errors = np.abs((predicted[mask] - ground_truth[mask]) / ground_truth[mask])
    mare = np.mean(relative_errors)
    
    # 统计信息
    n_total = len(ground_truth)
    n_valid = np.sum(mask)
    if n_valid < n_total:
        print(f"  Note: Used {n_valid}/{n_total} points (excluded {n_total-n_valid} near-zero values)")
    
    return mare


def load_ground_truth_from_file(clf, dataset_name, game_data=None):
    """Load ground truth from high-precision stratified sampling file"""
    dataset_name = dataset_name.lower()
    if dataset_name == "voting":
        weights = game_data if game_data is not None else get_voting_weights()
        print("Computing exact Shapley values for voting game...")
        return compute_exact_voting_shapley(weights)
    if dataset_name == "airport":
        costs = game_data if game_data is not None else get_airport_costs()
        print("Computing exact Shapley values for airport game...")
        return compute_exact_airport_shapley(costs)
    # pickle_file = f"results/pickles/svm_shapley_{dataset_name}_acc_monte_carlo_mc1000000.pkl"
    
    # try:
    #     with open(pickle_file, 'rb') as f:
    #         data = pickle.load(f)
        
    #     # Extract Shapley values - handle different possible data structures
    #     if isinstance(data, dict):
    #         if 'Stratified' in data:
    #             ground_truth = data['Stratified']
    #         elif 'stratified' in data:
    #             ground_truth = data['stratified']
    #         else:
    #             # Take the first available array
    #             ground_truth = list(data.values())[0]
    #     else:
    #         # Assume it's directly the array
    #         ground_truth = data
            
    #     print(f"✓ Loaded ground truth from {pickle_file}")
    #     print(f"  Shape: {np.array(ground_truth).shape}")
    #     print(f"  Sample values: {np.array(ground_truth)[:5]}")
    #     return np.array(ground_truth)
        
    # except FileNotFoundError:
    #     print(f"✗ Ground truth file not found: {pickle_file}")
    #     print("Please run the stratified method first to generate ground truth")
    #     return None
    # except Exception as e:
    #     print(f"✗ Error loading ground truth: {e}")
    
    file_name = f"/home/maoxiaokai/python_project/integral_shapley/results/pickles/{clf}_{dataset_name}_shapley_values_acc_groundtruth.npy"
    #load file
    groundtruth_sv = np.load(file_name)
    return groundtruth_sv


def main(visualize_sampling=False, target_point_for_viz=405):
    """Simple MARE comparison with target sample budgets, using actual budgets for Smart Adaptive."""
    
    # Target sample budgets to test
    k_values = [500, 1000, 2000, 5000, 10000]
    



    x_train, x_valid, y_train, y_valid = load_dataset(args.dataset)
    dataset_name = args.dataset.lower()
    utility_func = utility_acc
    voting_weights = None
    airport_costs = None
        
    print("Training set size:", x_train.shape)
    print("Training set type:", type(x_train), type(y_train))
    
    if dataset_name == "voting":
        voting_weights = x_train.flatten()
        total_weight = float(np.sum(voting_weights))
        final_model = {'threshold': total_weight / 2.0}
        clf = DummyClassifier(strategy='most_frequent')
        utility_func = utility_voting_game
        model = 'voting'
    elif dataset_name == "airport":
        airport_costs = x_train.flatten()
        final_model = {}
        clf = DummyClassifier(strategy='most_frequent')
        utility_func = utility_airport_game
        model = 'airport'
    else:
        if args.clf == 'svm':
            final_model = return_model('LinearSVC')
            final_model.fit(x_train, y_train)
            clf = return_model('LinearSVC')
        elif args.clf == 'lr':
            final_model = return_model('logistic')
            final_model.fit(x_train, y_train)
            clf = return_model('logistic')
        else:
            raise ValueError(f"Unsupported classifier: {args.clf}")
        model = args.clf
    
    n_points = len(x_train)
    print(f"Dataset: {dataset_name}, {n_points} points")
    
    
    # 1. Load ground truth from file
    print(f"\n1. Loading ground truth from file...")
    game_data = None
    if dataset_name == "voting":
        game_data = voting_weights
    elif dataset_name == "airport":
        game_data = airport_costs
    ground_truth = load_ground_truth_from_file(model, dataset_name, game_data)
    
    if ground_truth is None:
        print("Cannot proceed without ground truth. Exiting.")
        return
    
    # Ensure ground truth matches training set size
    if len(ground_truth) != n_points:
        print(f"Warning: Ground truth size ({len(ground_truth)}) != training set size ({n_points})")
        if len(ground_truth) > n_points:
            ground_truth = ground_truth[:n_points]
            print(f"Truncated ground truth to {n_points} points")
        else:
            print("Ground truth too small. Exiting.")
            return
    
    # 2. Test different methods with different k values
    results = []
    
    for k in k_values:
        print(f"\n2. Testing k={k}")
        
        # 1. Monte Carlo: k samples (parallel)
        print("   Monte Carlo (parallel)...")
        mc_args = []
        for i in range(n_points):
            mc_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                           utility_func, k))
        
        mc_dict = {}
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_mc_single, mc_args),
                                   total=len(mc_args), desc="Monte Carlo"):
                mc_dict[index] = value
        
        mc_values = np.array([mc_dict[i] for i in range(n_points)])
        mc_mare = compute_mare(mc_values, ground_truth)
        results.append({'k': k, 'method': 'monte_carlo', 'mare': mc_mare})
        print(f"      MARE: {mc_mare:.4f} ({mc_mare*100:.2f}%)")
        
        # # 2. Stratified: mc = k/N (parallel)
        # print("   Stratified (parallel)...")
        # # Stratified sampling has N layers (coalition sizes 0 to N-1, where N = n_points-1)
        # mc_per_coalition = max(1, k // n_points)
        
        # stratified_args = []
        # for i in range(n_points):
        #     stratified_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
        #                           utility_func, mc_per_coalition))
        
        # stratified_dict = {}
        # with mp.Pool(processes=args.processes) as pool:
        #     for index, value in tqdm(pool.imap_unordered(compute_stratified_single, stratified_args),
        #                            total=len(stratified_args), desc="Stratified"):
        #         stratified_dict[index] = value
        
        # stratified_values = np.array([stratified_dict[i] for i in range(n_points)])
        # stratified_mare = compute_mare(stratified_values, ground_truth)
        # # Stratified uses N * mc_per_coalition total budget (N = n_points for each data point)
        # stratified_actual_budget = n_points * mc_per_coalition
        # results.append({'k': stratified_actual_budget, 'method': 'stratified', 'mare': stratified_mare})
        # print(f"      MARE: {stratified_mare:.4f} ({stratified_mare*100:.2f}%)")
        # print(f"      Actual budget: {stratified_actual_budget} (target: {k})")
        
        # 3. Trapezoid Integration: t_samples * mc_samples = k (parallel)
        print("   Trapezoid Integration (parallel)...")
        # 梯形积分：选择 t 样本数和 MC 样本数
        # t_samples = max(10, min(50, int(np.sqrt(k))))
        t_samples = min(n_points, int(20 * (k ** (1/3))))  # 经验公式，确保不超过 n_points
        print(f"      Chosen t_samples={t_samples} based on k={k} and n_points={n_points}")
        # mc_samples = max(10, k // t_samples)
        mc_samples = k // t_samples
        
        print(f"      Using t_samples={t_samples}, mc_samples={mc_samples}")
        
        trapezoid_args = []
        for i in range(n_points):
            trapezoid_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                                 utility_func, t_samples, mc_samples))
        
        trapezoid_dict = {}
        with mp.Pool(processes=args.processes) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_trapezoid_single, trapezoid_args),
                                   total=len(trapezoid_args), desc="Trapezoid"):
                trapezoid_dict[index] = value
        
        trapezoid_values = np.array([trapezoid_dict[i] for i in range(n_points)])
        trapezoid_mare = compute_mare(trapezoid_values, ground_truth)
        # Trapezoid uses t_samples * mc_samples total budget
        trapezoid_actual_budget = t_samples * mc_samples
        results.append({'k': trapezoid_actual_budget, 'method': 'trapezoid', 'mare': trapezoid_mare})
        print(f"      MARE: {trapezoid_mare:.4f} ({trapezoid_mare*100:.2f}%)")
        print(f"      Actual budget: {trapezoid_actual_budget} (target: {k})")
        
        # 4. Simpson Integration: t_samples * mc_samples = k (parallel)
        print("   Simpson Integration (parallel)...")
        # Simpson 积分：使用奇数个 t 样本数
        # t_samples_simpson = max(101, 2*int(np.sqrt(k)) + 1)  # 确保奇数
        t_samples_simpson = min(n_points, int(20 * (k ** (1/3))))
        if t_samples_simpson % 2 == 0:
            t_samples_simpson += 1
        mc_samples_simpson = k // t_samples_simpson
        
        print(f"      Using t_samples={t_samples_simpson}, mc_samples={mc_samples_simpson}")
        
        simpson_args = []
        for i in range(n_points):
            simpson_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                               utility_func, t_samples_simpson, mc_samples_simpson))
        
        simpson_dict = {}
        with mp.Pool(processes=args.processes) as pool:
            for index, value in tqdm(pool.imap_unordered(compute_simpson_single, simpson_args),
                                   total=len(simpson_args), desc="Simpson"):
                simpson_dict[index] = value
        
        simpson_values = np.array([simpson_dict[i] for i in range(n_points)])
        simpson_mare = compute_mare(simpson_values, ground_truth)
        # # Simpson uses t_samples * mc_samples total budget
        # simpson_actual_budget = t_samples_simpson * mc_samples_simpson
        # results.append({'k': simpson_actual_budget, 'method': 'simpson', 'mare': simpson_mare})
        # print(f"      MARE: {simpson_mare:.4f} ({simpson_mare*100:.2f}%)")
        # print(f"      Actual budget: {simpson_actual_budget} (target: {k})")
        
        
        # CC (Complementary Contribution): mc = k/N (parallel)
        print("   CC (Complementary Contribution, parallel)...")
        # CC sampling has N layers (coalition sizes 0 to N-1), similar to stratified
        cc_mc_per_coalition = max(1, k)
        
        cc_values = compute_cc_values(x_train, y_train, x_valid, y_valid, clf, final_model, 
                                     utility_func, cc_mc_per_coalition)
        cc_mare = compute_mare(cc_values, ground_truth)
        # CC uses N * mc_per_coalition total budget
        cc_actual_budget = cc_mc_per_coalition
        results.append({'k': cc_actual_budget, 'method': 'cc', 'mare': cc_mare})
        print(f"      MARE: {cc_mare:.4f} ({cc_mare*100:.2f}%)")
        print(f"      Actual budget: {cc_actual_budget} (target: {k})")
        

        
        # 4. Simple Smart Adaptive: fixed intervals + direct sampling allocation
        print("   Simple Smart Adaptive (parallel)...")
        
        # Simple strategy: Fixed 20 intervals, allocate MC budget directly
        num_intervals = 30  # Fixed number of intervals
        
        # Estimate average sampling points based on our 4-level system
        # High(15) + Medium(9) + Low(5) + Minimal(3) distributed across 20 intervals
        # Conservative estimate: average ~8 points per interval
        estimated_total_points = num_intervals * 8  # ~160 points
        
        # Allocate MC budget: target total_points * mc_per_point = k
        mc_samples = max(1, k // estimated_total_points)
        actual_budget_estimate = estimated_total_points * mc_samples
        
        print(f"      Using {num_intervals} fixed intervals")
        print(f"      Estimated ~{estimated_total_points} total sampling points")
        print(f"      MC per point: {mc_samples}, estimated total budget: {actual_budget_estimate}")
        print(f"      Target budget: {k}, efficiency: {actual_budget_estimate/k:.2f}")
        
        adaptive_args = []
        for i in range(n_points):
            method_kwargs = {
                'num_MC': mc_samples,
                'min_samples_per_interval': 3
            }
            adaptive_args.append((i, x_train, y_train, x_valid, y_valid, clf, final_model, 
                                utility_func, 'smart_adaptive', 'probabilistic', method_kwargs))
        
        adaptive_dict = {}
        adaptive_budgets = {}
        with mp.Pool(processes=mp.cpu_count() // 10) as pool:
            for result in tqdm(pool.imap_unordered(compute_shapley_for_params_with_budget, adaptive_args),
                              total=len(adaptive_args), desc="Smart Adaptive"):
                if len(result) == 3:
                    index, value, budget = result
                    adaptive_dict[index] = value
                    adaptive_budgets[index] = budget
                else:
                    # Handle error case
                    index, value = result[:2]
                    adaptive_dict[index] = value
                    adaptive_budgets[index] = 0
        
        adaptive_values = np.array([adaptive_dict[i] for i in range(n_points)])
        adaptive_mare = compute_mare(adaptive_values, ground_truth)
        
        # Calculate actual average budget used
        actual_avg_budget = np.mean([adaptive_budgets[i] for i in range(n_points)])
        
        results.append({'k': actual_avg_budget, 'method': 'smart_adaptive', 'mare': adaptive_mare})
        print(f"      MARE: {adaptive_mare:.4f} ({adaptive_mare*100:.2f}%)")
        print(f"      Actual average budget used: {actual_avg_budget:.0f} (target: {k})")
        print(f"      Budget efficiency: {actual_avg_budget/k:.2f}")
        
        
                # 5. CC-Sparse-Integral (parallel, all players at once)
        print("   CC-Sparse-Integral (parallel, all players)...")
        # 预算映射：让“每点等价预算” ≈ m_layers * mc_per_layer ≈ k
        # 你也可以把 m_layers 设成函数：比如 min(64, max(16, int(np.sqrt(n_points)*2)))
        m_layers = min(n_points, int(20 * (k ** (1/3)))) 
        print(f"      Using m_layers={m_layers} based on k={k} and n_points={n_points}")
        
        mc_per_layer = max(1, k // m_layers * n_points)

        cc_sparse_values, cc_info = integral_cc_sparse_all_players_parallel(
            x_train, y_train, x_valid, y_valid,
            clf, final_model, utility_func,
            m_layers=m_layers,
            mc_per_layer=mc_per_layer,
            node_scheme="equispaced",   # 或 "chebyshev"
            aggregator="linear",       # 或 "voronoi"
            num_processes=args.processes,
            base_seed=42,
            verbose=False,
        )
        cc_sparse_mare = compute_mare(cc_sparse_values, ground_truth)
        cc_sparse_actual_budget = m_layers * mc_per_layer  // n_points # “每点等价预算”
        results.append({'k': cc_sparse_actual_budget, 'method': 'cc_sparse_integral', 'mare': cc_sparse_mare})
        print(f"      MARE: {cc_sparse_mare:.4f} ({cc_sparse_mare*100:.2f}%)")
        print(f"      Actual budget (per-point eqv.): {cc_sparse_actual_budget} (target: {k})")

        # CC Smart Adaptive Integral
        print("   CC Smart Adaptive Integral (all players)...")
        cc_num_intervals = 30
        cc_avg_layers_per_interval = 8
        cc_base_layers = cc_avg_layers_per_interval
        estimated_total_layers = cc_num_intervals * cc_avg_layers_per_interval
        cc_num_mc = max(5, k // max(1, estimated_total_layers) * n_points)
        cc_probe_mc = 10

        cc_smart_values, cc_smart_info = compute_integral_cc_smart_adaptive(
            x_train,
            y_train,
            x_valid,
            y_valid,
            clf,
            final_model,
            utility_func,
            num_intervals=cc_num_intervals,
            probe_layers_per_interval=3,
            probe_mc=cc_probe_mc,
            base_layers_per_interval=cc_base_layers,
            boost_high=15,
            boost_medium=7,
            boost_low=3,
            num_MC=cc_num_mc,
            aggregator="linear",
            num_processes=args.processes,
            base_seed=42,
            return_sampling_info=True,
        )
        cc_smart_mare = compute_mare(cc_smart_values, ground_truth)
        cc_smart_budget = cc_smart_info.get('integral_budget', cc_num_mc)
        results.append({'k': cc_smart_budget, 'method': 'cc_smart_adaptive', 'mare': cc_smart_mare})
        print(f"      MARE: {cc_smart_mare:.4f} ({cc_smart_mare*100:.2f}%)")
        print(f"      Actual budget (per-point eqv.): {cc_smart_budget} (target: {k})")

        
        # Generate sampling visualization for one data point if requested
        if k == 10000:
            eps = 1e-8
            mask = np.abs(ground_truth) >= eps  # 与 compute_mare 保持一致的过滤

            # 逐点绝对相对误差（ARE）
            are_trap = np.full(n_points, np.nan, dtype=float)
            are_simp = np.full(n_points, np.nan, dtype=float)
            are_trap[mask] = np.abs((trapezoid_values[mask] - ground_truth[mask]) / ground_truth[mask])
            are_simp[mask] = np.abs((simpson_values[mask]   - ground_truth[mask])   / ground_truth[mask])

            # 最差/最好（忽略 NaN）
            idx_max_trap = int(np.nanargmax(are_trap))
            idx_min_trap = int(np.nanargmin(are_trap))
            idx_max_simp = int(np.nanargmax(are_simp))
            idx_min_simp = int(np.nanargmin(are_simp))

            print("\n   ===== Extreme points (k=10000) =====")
            print(f"   Trapezoid worst idx={idx_max_trap:4d}, ARE={are_trap[idx_max_trap]:.4f}, "
                  f"GT={ground_truth[idx_max_trap]:.6g}, Est={trapezoid_values[idx_max_trap]:.6g}")
            print(f"   Trapezoid best  idx={idx_min_trap:4d}, ARE={are_trap[idx_min_trap]:.4f}, "
                  f"GT={ground_truth[idx_min_trap]:.6g}, Est={trapezoid_values[idx_min_trap]:.6g}")
            print(f"   Simpson   worst idx={idx_max_simp:4d}, ARE={are_simp[idx_max_simp]:.4f}, "
                  f"GT={ground_truth[idx_max_simp]:.6g}, Est={simpson_values[idx_max_simp]:.6g}")
            print(f"   Simpson   best  idx={idx_min_simp:4d}, ARE={are_simp[idx_min_simp]:.4f}, "
                  f"GT={ground_truth[idx_min_simp]:.6g}, Est={simpson_values[idx_min_simp]:.6g}")
        
        if visualize_sampling and k == 10000:  # Visualize for medium budget case
            print(f"      Generating Smart Adaptive sampling visualization for point {target_point_for_viz}...")
            
            # Compute Smart Adaptive with sampling info for visualization
            viz_method_kwargs = {
                'num_MC': mc_samples,
                'min_samples_per_interval': 3
            }
            
            result = compute_integral_shapley_value_with_budget(
                x_train, y_train, x_valid, y_valid, target_point_for_viz,
                clf, final_model, utility_func, 
                method='smart_adaptive', rounding_method='probabilistic',
                return_sampling_info=True, **viz_method_kwargs
            )
            
            shapley_value_viz, budget_viz, sampling_info = result
            
            # Create visualization
            viz_save_path = f"results/plots/smart_adaptive_sampling_point_{target_point_for_viz}_budget_{k}.png"
            visualize_smart_adaptive_sampling(
                sampling_info, 
                target_point_for_viz, 
                save_path=viz_save_path
            )
            
            print(f"      ✓ Sampling visualization saved to: {viz_save_path}")
            print(f"      Visualization details: Point {target_point_for_viz}, Shapley={shapley_value_viz:.6f}, Budget={budget_viz}")
    
    # 3. Create results DataFrame and plot
    results_df = pd.DataFrame(results)
    
    # Plot results with conference-style formatting
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'mathtext.fontset': 'dejavusans',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'figure.autolayout': True
    })
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Professional color palette inspired by Nature/Science journals
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#bcbd22']
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', 'x']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '--']
    
    # Line widths and marker sizes for different methods
    method_styles = {
        'monte_carlo': {'linewidth': 3.0, 'markersize': 10},
        'stratified': {'linewidth': 2.8, 'markersize': 9},
        'trapezoid': {'linewidth': 3.0, 'markersize': 9},
        'simpson': {'linewidth': 3.2, 'markersize': 10},
        'cc': {'linewidth': 3.2, 'markersize': 10},
        'cc_integral': {'linewidth': 3.0, 'markersize': 9},
        'a1_importance': {'linewidth': 3.0, 'markersize': 9},
        'smart_adaptive': {'linewidth': 3.5, 'markersize': 11},
        'sparse_residual': {'linewidth': 3.2, 'markersize': 10}
    }
    
    methods = results_df['method'].unique()
    method_labels = {
        'monte_carlo': 'Monte Carlo',
        'stratified': 'Stratified',
        'trapezoid': 'Trapezoid Integration',
        'cc': 'CC',
        'cc_integral': 'CC Integral',
        'a1_importance': 'A1 Importance Sampling',
        'simpson': 'Simpson Integration',
        'smart_adaptive': 'Smart Adaptive',
        'sparse_residual': 'Sparse Residual'
    }
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method].sort_values('k')
        style = method_styles.get(method, {'linewidth': 2.5, 'markersize': 8})
        
        ax.plot(method_data['k'], method_data['mare'], 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)], 
                linestyle=linestyles[i % len(linestyles)],
                linewidth=style['linewidth'], 
                markersize=style['markersize'],
                markerfacecolor='white',
                markeredgewidth=2.5,
                markeredgecolor=colors[i % len(colors)],
                label=method_labels.get(method, method.title()),
                alpha=0.95,
                zorder=10-i)  # Ensure important methods are on top
    
    # Formatting with professional appearance
    ax.set_xlabel('Sample Budget', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mean Absolute Relative Error', fontsize=15, fontweight='bold', labelpad=10)
    
    # Remove title for cleaner look (can be added in caption)
    # ax.set_title('Approximation Error vs Sample Budget\n(Breast Cancer Dataset)', 
    #             fontsize=16, fontweight='bold', pad=20)
    
    # Log scales with better formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Professional grid
    ax.grid(True, which="major", alpha=0.6, linestyle='-', linewidth=0.8, color='gray')
    ax.grid(True, which="minor", alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    
    # Enhanced legend
    legend = ax.legend(frameon=True, fancybox=False, shadow=False, 
                      fontsize=12, loc='upper right',
                      framealpha=0.95, edgecolor='black', facecolor='white',
                      borderpad=0.8, columnspacing=1.0, handlelength=2.5)
    
    # Professional tick formatting
    ax.tick_params(axis='both', which='major', labelsize=13, 
                   length=6, width=1.2, direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=11,
                   length=3, width=0.8, direction='in', top=True, right=True)
    
    # Set axis limits for better visualization
    y_min = results_df['mare'].min() * 0.7
    y_max = results_df['mare'].max() * 1.5
    ax.set_ylim(y_min, y_max)
    
    x_min = results_df['k'].min() * 0.8
    x_max = results_df['k'].max() * 1.2
    ax.set_xlim(x_min, x_max)
    
    # Professional spine formatting
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Tight layout with padding
    plt.tight_layout(pad=1.5)
    
    # Save results
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/csvs', exist_ok=True)
        
    # Also save as PDF for LaTeX
    plt.savefig(f'results/plots/{args.clf}_{args.dataset}_mare_comparison_t{args.t}.pdf', 
                bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='pdf', pad_inches=0.1)
    
    plt.show()
    
    results_df.to_csv(f'results/csvs/{args.clf}_{args.dataset}_mare_comparison_t{args.t}.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CANCER DATASET MARE COMPARISON SUMMARY (ACTUAL BUDGETS)")
    print("="*60)
    
    # Group by method and show actual budgets used
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method].sort_values('k')
        print(f"\n{method.upper()} METHOD:")
        for _, row in method_data.iterrows():
            print(f"  Budget k = {row['k']:8.0f}: MARE = {row['mare']:.4f} ({row['mare']*100:.2f}%)")
    
    # Find best method for each budget range
    print("\n" + "="*60)
    print("BEST METHOD BY BUDGET RANGE")
    print("="*60)
    
    # Group by approximate budget ranges
    budget_ranges = [(0, 1000), (1000, 3000), (3000, 7000), (7000, 15000), (15000, float('inf'))]
    for low, high in budget_ranges:
        range_data = results_df[(results_df['k'] >= low) & (results_df['k'] < high)]
        if not range_data.empty:
            best_method = range_data.loc[range_data['mare'].idxmin()]
            print(f"Budget range [{low}-{high}): {best_method['method']:15s} (k={best_method['k']:.0f}, MARE={best_method['mare']:.4f})")
    
    # Overall ranking by average performance
    print("\n" + "="*60)
    print("OVERALL METHOD RANKING (by average MARE)")
    print("="*60)
    
    avg_mare_by_method = results_df.groupby('method')['mare'].mean().sort_values()
    for rank, (method, avg_mare) in enumerate(avg_mare_by_method.items(), 1):
        print(f"{rank}. {method:15s}: Average MARE = {avg_mare:.4f} ({avg_mare*100:.2f}%)")
    
    print(f"\nGround truth loaded from stratified sampling (MC=1000)")
    print("Results saved to results/")
    print(f"Dataset: Cancer ({n_points} training points)")
    print("Note: Smart Adaptive uses actual sampled budgets, other methods use target budgets")
    print("MARE: Mean Absolute Relative Error - measures percentage error relative to ground truth")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MARE comparison with optional Smart Adaptive visualization")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate Smart Adaptive sampling visualization")
    parser.add_argument("--viz_point", type=int, default=0,
                       help="Data point index for visualization (default: 0)")
    parser.add_argument("--dataset", type=str, default="cancer",)
    parser.add_argument("--clf", type=str,)
    parser.add_argument("--t", type=int, default=100)
    parser.add_argument("--processes", type=int, default=150)
    parser.add_argument("--game", type=str, default="ml")
    
    args = parser.parse_args()
    
    try:
        results_df = main(visualize_sampling=args.visualize, target_point_for_viz=args.viz_point)
        print("\n✅ MARE comparison completed successfully!")
        
        if args.visualize:
            print(f"✅ Smart Adaptive sampling visualization generated for data point {args.viz_point}")
            
    except Exception as e:
        print(f"\n❌ Error during MARE comparison: {e}")
        import traceback
        traceback.print_exc()
