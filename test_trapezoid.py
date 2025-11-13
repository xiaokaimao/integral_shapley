#!/usr/bin/env python
"""
测试修正后的梯形法 Shapley 值计算
验证 (n-1)/n 系数和端点校正的效果
"""

import numpy as np
import time
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.basic_integration import compute_integral_shapley_trapezoid
from src.utils.utilities import utility_acc
from src.utils.model_utils import return_model


def load_ground_truth():
    """加载癌症数据集的真实 Shapley 值（蒙特卡洛1000000次）"""
    try:
        with open('results/pickles/svm_shapley_cancer_acc_monte_carlo_mc1000000.pkl', 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data['Monte_Carlo']
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data)
        else:
            print(f"未知的数据格式: {type(data)}")
            return None
    except FileNotFoundError:
        print("未找到真实 Shapley 值文件")
        return None
    except Exception as e:
        print(f"加载真实 Shapley 值时出错: {e}")
        return None


def prepare_cancer_data():
    """准备癌症数据集，与保存的 Shapley 值使用相同的预处理"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 使用固定随机种子，确保与保存的结果一致
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    return x_train, x_valid, y_train, y_valid


def test_cancer_dataset():
    """用癌症数据集测试，比较梯形法和真实 Shapley 值"""
    print("=" * 60)
    print("测试：梯形法 vs 真实 Shapley 值（癌症数据集）")
    print("=" * 60)
    
    # 加载真实 Shapley 值
    ground_truth = load_ground_truth()
    if ground_truth is None:
        print("无法加载真实 Shapley 值，跳过测试")
        return
    
    # 准备数据
    x_train, x_valid, y_train, y_valid = prepare_cancer_data()
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    print(f"训练集大小: {len(x_train)}")
    print(f"真实 Shapley 值数量: {len(ground_truth)}")
    
    # 确保数据大小匹配
    if len(ground_truth) != len(x_train):
        print(f"警告: 真实值数量({len(ground_truth)}) != 训练集大小({len(x_train)})")
        min_size = min(len(ground_truth), len(x_train))
        ground_truth = ground_truth[:min_size]
        x_train = x_train[:min_size]
        y_train = y_train[:min_size]
        print(f"截取到 {min_size} 个样本")
    
    # 测试几个数据点
    test_indices = [0, 1, 2, 5, 10]
    test_indices = [i for i in test_indices if i < len(x_train)]
    
    print(f"\n测试数据点: {test_indices}")
    
    results = []
    for i in test_indices:
        print(f"\n计算数据点 {i}:")
        
        # 真实值
        true_sv = ground_truth[i]
        print(f"   真实 Shapley 值: {true_sv:.6f}")
        
        # 梯形法（概率取整 + 端点校正）
        print("   计算梯形法（概率取整）...")
        start_time = time.time()
        trap_sv = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func=utility_acc,
            num_t_samples=30, num_MC=100,
            rounding_method='probabilistic'
        )
        trap_time = time.time() - start_time
        print(f"   梯形 Shapley 值: {trap_sv:.6f}")
        print(f"   计算时间: {trap_time:.2f}s")
        
        # 梯形法（标准取整）
        print("   计算梯形法（标准取整）...")
        start_time = time.time()
        trap_sv_round = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func=utility_acc,
            num_t_samples=30, num_MC=100,
            rounding_method='round'
        )
        trap_time_round = time.time() - start_time
        print(f"   梯形 Shapley 值: {trap_sv_round:.6f}")
        print(f"   计算时间: {trap_time_round:.2f}s")
        
        # 计算误差
        error_prob = abs(trap_sv - true_sv)
        error_round = abs(trap_sv_round - true_sv)
        rel_error_prob = error_prob / abs(true_sv) * 100 if true_sv != 0 else float('inf')
        rel_error_round = error_round / abs(true_sv) * 100 if true_sv != 0 else float('inf')
        
        print(f"   误差(概率):     {error_prob:.6f} ({rel_error_prob:.2f}%)")
        print(f"   误差(标准):     {error_round:.6f} ({rel_error_round:.2f}%)")
        
        results.append({
            'index': i,
            'true': true_sv,
            'trap_prob': trap_sv,
            'trap_round': trap_sv_round,
            'error_prob': error_prob,
            'error_round': error_round,
            'rel_error_prob': rel_error_prob,
            'rel_error_round': rel_error_round
        })
    
    # 总结
    print("\n" + "="*60)
    print("测试总结:")
    print("="*60)
    avg_error_prob = np.mean([r['error_prob'] for r in results])
    avg_error_round = np.mean([r['error_round'] for r in results])
    avg_rel_error_prob = np.mean([r['rel_error_prob'] for r in results if r['rel_error_prob'] != float('inf')])
    avg_rel_error_round = np.mean([r['rel_error_round'] for r in results if r['rel_error_round'] != float('inf')])
    
    print(f"平均绝对误差(概率): {avg_error_prob:.6f}")
    print(f"平均绝对误差(标准): {avg_error_round:.6f}")
    print(f"平均相对误差(概率): {avg_rel_error_prob:.2f}%")
    print(f"平均相对误差(标准): {avg_rel_error_round:.2f}%")


def test_coefficient_effect():
    """测试 (n-1)/n 系数的效果"""
    print("\n" + "=" * 60)
    print("测试：(n-1)/n 系数的效果")
    print("=" * 60)
    
    x_train, x_valid, y_train, y_valid = prepare_cancer_data()
    
    # 取子集进行快速测试
    x_train = x_train[:20]
    y_train = y_train[:20]
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    n = len(x_train)
    coef = (n-1) / n
    print(f"训练集大小 n = {n}")
    print(f"系数 (n-1)/n = {coef:.4f}")
    
    i = 0
    
    # 模拟没有系数的积分结果
    from src.core.base import compute_marginal_contribution_at_t
    t_values = np.linspace(0, 1, 20)
    integrand = []
    for t in t_values:
        val = compute_marginal_contribution_at_t(
            t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_acc, num_MC=30, rounding_method='probabilistic'
        )
        integrand.append(val)
    
    raw_integral = np.trapezoid(integrand, t_values)
    corrected_integral = coef * raw_integral
    
    print(f"\n原始积分结果:     {raw_integral:.6f}")
    print(f"乘以系数后:       {corrected_integral:.6f}")
    print(f"系数修正比例:     {coef:.4f}")


def test_multiple_points():
    """测试多个数据点的 Shapley 值"""
    print("\n" + "=" * 60)
    print("测试：多个数据点的 Shapley 值")
    print("=" * 60)
    
    x_train, x_valid, y_train, y_valid = prepare_cancer_data()
    
    # 取子集进行测试
    x_train = x_train[:15]
    y_train = y_train[:15]
    
    final_model = return_model('LinearSVC')
    final_model.fit(x_train, y_train)
    clf = return_model('LinearSVC')
    
    n_test = min(5, len(x_train))
    print(f"计算前 {n_test} 个点的 Shapley 值")
    
    shapley_values = []
    for i in range(n_test):
        print(f"\n计算点 {i}...")
        sv = compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func=utility_acc,
            num_t_samples=15, num_MC=30,
            rounding_method='probabilistic'
        )
        shapley_values.append(sv)
        print(f"  Shapley值: {sv:.6f}")
    
    print(f"\n所有 Shapley 值: {[f'{sv:.6f}' for sv in shapley_values]}")
    print(f"平均值: {np.mean(shapley_values):.6f}")
    print(f"标准差: {np.std(shapley_values):.6f}")


if __name__ == "__main__":
    print("测试修正后的梯形法 Shapley 值计算\n")
    
    try:
        # 测试1: 与真实 Shapley 值比较
        test_cancer_dataset()
        
        # 测试2: 系数效果
        test_coefficient_effect()
        
        # 测试3: 多个数据点
        test_multiple_points()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()