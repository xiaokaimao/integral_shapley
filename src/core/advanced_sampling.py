#!/usr/bin/env python
"""
Advanced sampling methods: Importance Sampling and Sparse Residual methods.

This module implements sophisticated sampling strategies for efficient Shapley value computation:
- A1 Importance Sampling over coalition sizes
- Sparse Residual method with Chebyshev nodes and residual correction
"""

import numpy as np
from sklearn.base import clone
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from typing import Optional

from .base import compute_marginal_contribution_at_t


def compute_integral_shapley_importance_sampling(
    x_train, y_train, x_valid, y_valid, i, clf, final_model,
    utility_func, num_samples=1000, num_MC=1,
    rounding_method='probabilistic',
    p_s=None,                   # 可选：规模分布（长度 n 的概率向量）；默认自动估计
    warmup_per_s=5,             # 预热条数/规模，用于估 σ_s 设定 p_s；设为0可跳过
    lambda_eps=1e-8,            # 防止 p_s 过小的下界
    separate_last=True,         # 是否把 s=n-1 的常数项精确剥离
    rng=None
):
    """
    A1: Importance Sampling over coalition sizes (pure fixed-size sampling, unbiased).
    
    基于积分公式 SV_i = ∫₀¹ f*(t) dt = E[Δ/(n·p_s)] 的重要性采样实现。
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data  
        i: Target data point index
        clf: Classifier to train on subsets
        final_model: Model trained on full data
        utility_func: Utility function
        num_samples: 总采样次数 K
        num_MC: 同一规模上的子集平均次数（降噪用）
        rounding_method: 兼容参数（未使用）
        p_s: 可选的规模分布；默认基于方差估计
        warmup_per_s: 预热阶段每个规模的采样数
        lambda_eps: p_s 下界，防止极小概率
        separate_last: 是否精确处理 s=n-1 层
        rng: 随机数生成器
        
    Returns:
        tuple: (shapley_value, info_dict)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x_train)
    assert 0 <= i < n, f"Index i={i} out of range [0, {n})"

    # 构建不含 i 的索引
    all_indices = np.arange(n)
    candidate_indices = all_indices[all_indices != i]

    # ----- 可选：精确分离 s=n-1 的常数项 -----
    constant_term = 0.0
    valid_s_range = np.arange(n)  # 默认 0..n-1
    
    if separate_last and n > 1:
        # E_{n-1} = v(N) - v(N\{i})，这是确定值
        try:
            # v(N) 由 final_model 直接给出 - 使用与其他地方一致的接口
            u_full = utility_func(x_train, y_train, x_valid, y_valid, final_model, final_model)
            
            # v(N\{i}) 需要重新训练
            X_without_i = x_train[candidate_indices]
            y_without_i = y_train[candidate_indices]
            u_without_i = utility_func(X_without_i, y_without_i, x_valid, y_valid, clone(clf), final_model)
            
            E_last = u_full - u_without_i
            constant_term = E_last / n
            valid_s_range = np.arange(n-1)  # 只在 0..n-2 上做 IS
        except Exception as e:
            print(f"Warning: Failed to separate last layer: {e}. Using full range.")
            separate_last = False

    # ----- 构造规模分布 p_s -----
    if p_s is None:
        if warmup_per_s <= 0 or len(valid_s_range) == 0:
            # 均匀分布
            p = np.ones(len(valid_s_range), dtype=float)
        else:
            # 基于方差估计的 Neyman 分配
            p = np.zeros(len(valid_s_range), dtype=float)
            
            for idx_s, s in enumerate(valid_s_range):
                warmup_deltas = []
                for _ in range(warmup_per_s):
                    # 采样 |S|=s 的子集
                    if s == 0:
                        X_S = np.empty((0, x_train.shape[1]))
                        y_S = np.empty(0)
                    else:
                        sample_idx = rng.choice(candidate_indices, size=s, replace=False)
                        X_S = x_train[sample_idx]
                        y_S = y_train[sample_idx]
                    
                    # 计算边际贡献 Δ = v(S∪{i}) - v(S)
                    try:
                        v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
                    except:
                        v_S = 0.0
                    
                    # S ∪ {i}
                    if s == 0:
                        X_S_i = x_train[i:i+1]
                        y_S_i = np.array([y_train[i]])
                    else:
                        X_S_i = np.vstack([X_S, x_train[i:i+1]])
                        y_S_i = np.append(y_S, y_train[i])
                    
                    try:
                        v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
                    except:
                        v_S_i = 0.0
                    
                    warmup_deltas.append(v_S_i - v_S)
                
                # 使用标准差作为重要性权重
                if len(warmup_deltas) > 1:
                    p[idx_s] = np.std(warmup_deltas, ddof=1)
                elif len(warmup_deltas) == 1:
                    p[idx_s] = abs(warmup_deltas[0]) + lambda_eps
                else:
                    p[idx_s] = lambda_eps
        
        # 添加下界并归一化
        p = np.maximum(p, lambda_eps)
        p = p / p.sum()
    else:
        # 用户提供的分布
        p = np.array(p_s, dtype=float)
        if separate_last and len(p) == n:
            p = p[:-1]  # 去掉最后一个
        assert len(p) == len(valid_s_range), f"p_s length mismatch: {len(p)} vs {len(valid_s_range)}"
        assert np.all(p >= 0) and p.sum() > 0, "Invalid p_s"
        p = p / p.sum()

    # ----- A1 重要性采样主循环 -----
    estimates = []
    
    for _ in range(num_samples):
        if len(valid_s_range) == 0:
            estimates.append(0.0)
            continue
        
        # 按 p 分布选择规模 s
        s_idx = rng.choice(len(valid_s_range), p=p)
        s = valid_s_range[s_idx]
        
        # 在该规模上做 num_MC 次子集采样（降噪）
        deltas = []
        for _ in range(num_MC):
            if s == 0:
                X_S = np.empty((0, x_train.shape[1]))
                y_S = np.empty(0)
            else:
                sample_idx = rng.choice(candidate_indices, size=s, replace=False)
                X_S = x_train[sample_idx]
                y_S = y_train[sample_idx]
            
            # 计算 v(S)
            try:
                v_S = utility_func(X_S, y_S, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S = 0.0
            
            # 计算 v(S ∪ {i})
            if s == 0:
                X_S_i = x_train[i:i+1]
                y_S_i = np.array([y_train[i]])
            else:
                X_S_i = np.vstack([X_S, x_train[i:i+1]])
                y_S_i = np.append(y_S, y_train[i])
            
            try:
                v_S_i = utility_func(X_S_i, y_S_i, x_valid, y_valid, clone(clf), final_model)
            except:
                v_S_i = 0.0
            
            deltas.append(v_S_i - v_S)
        
        # 平均后加权：关键的无偏权重
        delta_avg = np.mean(deltas)
        weight = 1.0 / (n * p[s_idx])
        estimates.append(delta_avg * weight)

    # 最终结果：常数项 + IS估计
    shapley_value = constant_term + np.mean(estimates)
    
    # 粗略置信区间
    if len(estimates) > 1:
        se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        ci95 = (shapley_value - 1.96 * se, shapley_value + 1.96 * se)
    else:
        ci95 = (shapley_value, shapley_value)
    
    info = {
        "method": "importance_sampling",
        "constant_term": constant_term,
        "p_s": p,
        "valid_s_range": valid_s_range,
        "ci95": ci95,
        "separate_last": separate_last,
        "num_samples": num_samples,
        "num_MC": num_MC
    }
    
    return shapley_value, info


def generate_chebyshev_nodes_with_endpoints(m_inner, interval=(0, 1)):
    """
    生成包含端点的 Chebyshev 节点
    
    Args:
        m_inner: 内部 Chebyshev 节点数量
        interval: 目标区间，默认 (0, 1)
        
    Returns:
        nodes: 包含端点的 Chebyshev 节点数组，总数 = m_inner + 2
    """
    if m_inner < 0:
        raise ValueError("m_inner must be non-negative")
    
    if m_inner == 0:
        # 只返回端点
        return np.array([interval[0], interval[1]])
    
    # 生成内部 Chebyshev 节点
    inner_nodes = generate_chebyshev_nodes(m_inner, interval)
    
    # 添加端点
    a, b = interval
    all_nodes = np.concatenate(([a], inner_nodes, [b]))
    
    # 去重并排序
    nodes = np.unique(np.clip(all_nodes, a, b))
    return np.sort(nodes)


def generate_chebyshev_nodes(m, interval=(0, 1)):
    """
    生成 [0,1] 区间上的 Chebyshev 节点
    
    Args:
        m: 节点数量
        interval: 目标区间，默认 (0, 1)
        
    Returns:
        nodes: Chebyshev 节点数组
    """
    # 生成标准 [-1, 1] 区间的 Chebyshev 节点
    nodes_std = np.cos(np.pi * (2 * np.arange(m) + 1) / (2 * m))
    
    # 映射到目标区间
    a, b = interval
    nodes = (nodes_std + 1) / 2 * (b - a) + a
    
    return np.sort(nodes)  # 从小到大排序


def fit_smooth_approximation(t_nodes, E_values, method='pchip'):
    """
    在节点上拟合光滑近似函数 h(t)
    
    Args:
        t_nodes: t 节点数组
        E_values: 对应的 E[Δ(t,i)] 估计值
        method: 拟合方法 ('pchip', 'spline')
        
    Returns:
        h_func: 拟合的函数对象
    """
    if method == 'pchip':
        # PCHIP 单调三次样条，适合单调函数
        h_func = PchipInterpolator(t_nodes, E_values, extrapolate=True)
    elif method == 'spline':
        # 三次样条
        h_func = UnivariateSpline(t_nodes, E_values, s=0, ext='extrapolate')
    else:
        raise ValueError(f"Unknown fitting method: {method}")
    
    return h_func


def compute_integral_and_segment_means(h_func, n, interval=(0, 1)):
    """
    一次性计算拟合函数的积分和各段平均值
    
    Args:
        h_func: 拟合的函数对象
        n: 数据集大小
        interval: 积分区间
        
    Returns:
        integral_value: 总积分值
        a_s: 每个分段的平均值数组
    """
    a, b = interval
    
    # 优先使用对象自带的高效积分方法
    if hasattr(h_func, "integrate"):  # PchipInterpolator
        integral_value = h_func.integrate(a, b)
        a_s = np.array([n * h_func.integrate(s/n, (s+1)/n) for s in range(n)])
    elif hasattr(h_func, "integral"):  # UnivariateSpline
        integral_value = h_func.integral(a, b)
        a_s = np.array([n * h_func.integral(s/n, (s+1)/n) for s in range(n)])
    else:  # 兜底使用数值积分
        from scipy.integrate import quad
        integral_value, _ = quad(h_func, a, b)
        a_s = np.array([n * quad(h_func, s/n, (s+1)/n)[0] for s in range(n)])
    
    return integral_value, a_s


def generate_sparse_warmup_sizes(n, num_warmup=32):
    """
    生成稀疏预热的尺寸集合
    
    Args:
        n: 数据集大小
        num_warmup: 预热尺寸数量
        
    Returns:
        warmup_sizes: 预热尺寸数组
    """
    # 端点 + Chebyshev 尺寸 + 等距尺寸
    if n <= num_warmup:
        return np.arange(n)
    
    # 端点
    sizes = [0, n-1]
    
    # Chebyshev 尺寸（映射到 [0, n-1]）
    cheb_t = generate_chebyshev_nodes(min(num_warmup-4, n-2), interval=(0, 1))
    cheb_sizes = np.round(cheb_t * (n-1)).astype(int)
    sizes.extend(cheb_sizes)
    
    # 等距尺寸
    num_uniform = max(num_warmup - len(sizes), 0)
    if num_uniform > 0:
        uniform_sizes = np.linspace(0, n-1, num_uniform, dtype=int)
        sizes.extend(uniform_sizes)
    
    # 去重并排序
    sizes = np.unique(np.array(sizes))
    return sizes[sizes < n]  # 确保所有尺寸都在有效范围内


def estimate_residual_sampling_distribution_sparse(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                                  utility_func, a_s, num_warmup=32, warmup_samples=5, 
                                                  lambda_eps=1e-8, rng=None):
    """
    使用稀疏预热估计残差重要性采样分布
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        a_s: 拟合函数的段平均值
        num_warmup: 预热尺寸数量
        warmup_samples: 每个尺寸的预热样本数
        lambda_eps: 最小概率下界
        rng: 随机数生成器
        
    Returns:
        p_s: 采样概率分布
        warmup_evaluations: 预热阶段的函数评估次数
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    candidate_indices = np.arange(n)[np.arange(n) != i]
    
    # 生成稀疏预热尺寸
    warmup_sizes = generate_sparse_warmup_sizes(n, num_warmup)
    
    # 在预热尺寸上估计方差和偏差
    sigma_warmup = {}
    bias_warmup = {}
    evaluation_count = 0
    
    for s in warmup_sizes:
        delta_values = []
        
        if s == 0:
            # s=0 时，S 为空集
            try:
                u_empty = utility_func(np.array([]).reshape(0, -1), np.array([]), 
                                     x_valid, y_valid, final_model, final_model)
                u_with_i = utility_func(x_train[i:i+1], y_train[i:i+1], 
                                      x_valid, y_valid, final_model, final_model)
                delta_values = [u_with_i - u_empty]
                evaluation_count += 2  # 两次效用函数调用
            except:
                continue  # 跳过失败的计算，不污染分布
                
        elif s == n-1:
            # s=n-1 时，S = N\{i}（唯一子集）
            try:
                X_without_i = x_train[candidate_indices]
                y_without_i = y_train[candidate_indices]
                model_without_i = clone(clf)
                model_without_i.fit(X_without_i, y_without_i)
                u_without_i = utility_func(X_without_i, y_without_i, x_valid, y_valid, 
                                         model_without_i, final_model)
                u_full = utility_func(x_train, y_train, x_valid, y_valid, final_model, final_model)
                delta_values = [u_full - u_without_i]
                evaluation_count += 2  # 两次效用函数调用
            except:
                continue
                
        else:
            # 一般情况：随机采样若干个大小为 s 的子集
            for _ in range(warmup_samples):
                try:
                    subset_indices = rng.choice(candidate_indices, size=s, replace=False)
                    
                    # 训练不包含 i 的子集模型
                    X_subset = x_train[subset_indices]
                    y_subset = y_train[subset_indices]
                    model_subset = clone(clf)
                    model_subset.fit(X_subset, y_subset)
                    u_subset = utility_func(X_subset, y_subset, x_valid, y_valid, 
                                          model_subset, final_model)
                    
                    # 训练包含 i 的子集模型
                    subset_with_i = np.concatenate([subset_indices, [i]])
                    X_subset_with_i = x_train[subset_with_i]
                    y_subset_with_i = y_train[subset_with_i]
                    model_subset_with_i = clone(clf)
                    model_subset_with_i.fit(X_subset_with_i, y_subset_with_i)
                    u_subset_with_i = utility_func(X_subset_with_i, y_subset_with_i, 
                                                 x_valid, y_valid, model_subset_with_i, final_model)
                    
                    delta = u_subset_with_i - u_subset
                    delta_values.append(delta)
                    evaluation_count += 2  # 两次效用函数调用
                except:
                    continue  # 跳过失败的计算
        
        # 计算统计量（只有成功的样本）
        if len(delta_values) > 0:
            E_s_est = np.mean(delta_values)
            sigma_warmup[s] = np.std(delta_values, ddof=1) if len(delta_values) > 1 else 0
            bias_warmup[s] = abs(E_s_est - a_s[s])
        else:
            # 如果没有成功的样本，使用保守估计
            sigma_warmup[s] = 1.0  # 保守的标准差
            bias_warmup[s] = abs(a_s[s])
    
    # 插值到所有尺寸
    sigma_s = np.zeros(n)
    bias_s = np.zeros(n)
    
    warmup_s_list = np.array(list(sigma_warmup.keys()))
    sigma_values = np.array([sigma_warmup[s] for s in warmup_s_list])
    bias_values = np.array([bias_warmup[s] for s in warmup_s_list])
    
    if len(warmup_s_list) > 1:
        # 使用 PCHIP 单调样条插值，更稳定不振荡
        try:
            sigma_interp = PchipInterpolator(warmup_s_list, sigma_values, extrapolate=True)
            bias_interp = PchipInterpolator(warmup_s_list, bias_values, extrapolate=True)
            sigma_s = sigma_interp(np.arange(n))
            bias_s = bias_interp(np.arange(n))
            # 确保插值结果为非负
            sigma_s = np.maximum(sigma_s, 0)
            bias_s = np.maximum(bias_s, 0)
        except Exception:
            # 如果PCHIP失败，使用线性插值作为备份
            sigma_s = np.interp(np.arange(n), warmup_s_list, sigma_values)
            bias_s = np.interp(np.arange(n), warmup_s_list, bias_values)
    elif len(warmup_s_list) == 1:
        # 只有一个点，使用常数
        sigma_s.fill(sigma_values[0])
        bias_s.fill(bias_values[0])
    else:
        # 没有成功的预热，使用均匀分布
        sigma_s.fill(1.0)
        bias_s = np.abs(a_s)
    
    # 计算重要性采样权重
    weights = np.sqrt(sigma_s**2 + bias_s**2) + lambda_eps
    p_s = weights / np.sum(weights)
    
    return p_s, evaluation_count


def compute_residual_correction(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                              utility_func, a_s, p_s, K=100, rng=None):
    """
    计算残差校正项：E[∫₀¹ (f*(t) - h(t)) dt] 的重要性采样估计
    
    优化：预计算端点delta值，避免重复训练
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        a_s: 拟合函数的段平均值数组
        p_s: 采样分布
        K: 残差采样次数
        rng: 随机数生成器
        
    Returns:
        residual_estimate: 残差估计值
        residual_std: 残差估计的标准误差
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    idx_wo_i = np.arange(n) != i
    candidate_indices = np.arange(n)[idx_wo_i]
    
    # ===== 端点预计算 =====
    # s=0 的情况
    try:
        u_empty = utility_func(np.empty((0, x_train.shape[1])), np.empty((0,)), 
                               x_valid, y_valid, final_model, final_model)
        u_with_i = utility_func(x_train[i:i+1], y_train[i:i+1],
                                x_valid, y_valid, final_model, final_model)
        delta_s0 = u_with_i - u_empty
    except Exception:
        delta_s0 = 0.0
    
    # s=n-1 的情况
    try:
        X_wo = x_train[candidate_indices]
        y_wo = y_train[candidate_indices]
        m_wo = clone(clf)
        m_wo.fit(X_wo, y_wo)
        u_wo = utility_func(X_wo, y_wo, x_valid, y_valid, m_wo, final_model)
        u_full = utility_func(x_train, y_train, x_valid, y_valid, final_model, final_model)
        delta_sn1 = u_full - u_wo
    except Exception:
        delta_sn1 = 0.0
    
    # ===== 重要性采样 =====
    residual_samples = []
    successes = 0
    attempts = 0
    max_attempts = K * 3  # 防止无限循环，最多尝试3倍
    
    while successes < K and attempts < max_attempts:
        attempts += 1
        # 按分布 p_s 采样尺寸 s
        s = rng.choice(n, p=p_s)
        
        try:
            if s == 0:
                delta = delta_s0  # 使用预计算值
            elif s == n-1:
                delta = delta_sn1  # 使用预计算值
            else:
                # 一般情况：随机采样大小为 s 的子集
                subset_indices = rng.choice(candidate_indices, size=s, replace=False)
                
                # 训练不包含 i 的子集模型
                X_subset = x_train[subset_indices]
                y_subset = y_train[subset_indices]
                model_subset = clone(clf)
                model_subset.fit(X_subset, y_subset)
                u_subset = utility_func(X_subset, y_subset, x_valid, y_valid, 
                                      model_subset, final_model)
                
                # 训练包含 i 的子集模型
                subset_with_i = np.concatenate([subset_indices, [i]])
                X_subset_with_i = x_train[subset_with_i]
                y_subset_with_i = y_train[subset_with_i]
                model_subset_with_i = clone(clf)
                model_subset_with_i.fit(X_subset_with_i, y_subset_with_i)
                u_subset_with_i = utility_func(X_subset_with_i, y_subset_with_i, 
                                             x_valid, y_valid, model_subset_with_i, final_model)
                
                delta = u_subset_with_i - u_subset
            
            # 计算重要性采样权重
            Y = (delta - a_s[s]) / (n * p_s[s])
            residual_samples.append(Y)
            successes += 1
            
        except Exception:
            # 失败时直接重试，不添加0，保持无偏性
            continue
    
    # 如果成功样本不足，记录警告但仍计算
    if successes < K:
        print(f"警告: 残差采样只成功 {successes}/{K} 次")
    
    if successes == 0:
        # 极端情况：所有采样都失败
        residual_estimate = 0.0
        residual_std = 0.0
    else:
        residual_samples = np.asarray(residual_samples)
        residual_estimate = float(residual_samples.mean())
        residual_std = (float(residual_samples.std(ddof=1) / np.sqrt(successes)) if successes > 1 else 0.0)
    
    return residual_estimate, residual_std


def compute_integral_shapley_sparse_residual(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                           utility_func, num_nodes=8, num_MC_per_node=50, 
                                           residual_samples=100, fitting_method='pchip',
                                           rounding_method='probabilistic', warmup_per_node=5,
                                           lambda_eps=1e-8, return_detailed_info=False, rng=None):
    """
    稀疏积分 + 无偏残差方法计算 Shapley 值（优化版本）
    
    核心思想：
    SV_i = ∫₀¹ h(t) dt + ∫₀¹ [f*(t) - h(t)] dt
           ^主积分(解析)    ^残差(重要性采样校正)
    
    优化：
    - 使用包含端点的 Chebyshev 节点
    - 一次性计算积分和段平均值
    - 稀疏预热估计采样分布，复杂度从 O(n) 降到 O(num_warmup)
    - 修正 off-by-one 错误
    
    Args:
        x_train, y_train: 训练数据
        x_valid, y_valid: 验证数据
        i: 目标数据点索引
        clf: 分类器
        final_model: 完整模型
        utility_func: 效用函数
        num_nodes: Chebyshev 节点数量 (m << n)
        num_MC_per_node: 每个节点的 MC 采样次数
        residual_samples: 残差校正的采样次数 K
        fitting_method: 拟合方法 ('pchip', 'spline')
        rounding_method: 联盟大小舍入方法
        warmup_per_node: 预热阶段每个节点的采样数
        lambda_eps: 采样概率下界
        return_detailed_info: 是否返回详细信息
        rng: 随机数生成器
        
    Returns:
        如果 return_detailed_info=False: shapley_value
        如果 return_detailed_info=True: (shapley_value, info_dict)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x_train)
    
    # ===== 第一阶段：稀疏节点拟合 =====
    
    # 1) 生成包含端点的 Chebyshev 节点
    t_nodes = generate_chebyshev_nodes_with_endpoints(max(num_nodes-2, 0), interval=(0, 1))
    
    # 2) 在每个节点高精度估计 E[Δ(t,i)]
    print(f"稀疏积分: 在 {len(t_nodes)} 个 Chebyshev 节点（含端点）上估计积分函数...")
    E_values = []
    for t in t_nodes:
        marginal_contrib = compute_marginal_contribution_at_t(
            t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC_per_node, rounding_method, rng
        )
        E_values.append(marginal_contrib)
    
    E_values = np.array(E_values)
    
    # 3) 拟合光滑函数 h(t)
    print(f"拟合光滑近似函数 (方法: {fitting_method})...")
    h_func = fit_smooth_approximation(t_nodes, E_values, method=fitting_method)
    
    # 4) 一次性计算主积分和段平均值
    print("计算主积分和段平均值...")
    main_integral, a_s = compute_integral_and_segment_means(h_func, n, interval=(0, 1))
    
    # ===== 第二阶段：残差校正 =====
    
    # 5) 稀疏预热估计最优重要性采样分布
    print("稀疏预热估计残差重要性采样分布...")
    p_s, warmup_evaluations = estimate_residual_sampling_distribution_sparse(
        x_train, y_train, x_valid, y_valid, i, clf, final_model,
        utility_func, a_s, num_warmup=min(32, n), warmup_samples=warmup_per_node, 
        lambda_eps=lambda_eps, rng=rng
    )
    
    # 6) 计算残差校正
    print(f"计算残差校正 (采样 {residual_samples} 次)...")
    residual_correction, residual_std = compute_residual_correction(
        x_train, y_train, x_valid, y_valid, i, clf, final_model,
        utility_func, a_s, p_s, K=residual_samples, rng=rng
    )
    
    # ===== 最终结果 =====
    shapley_value = main_integral + residual_correction
    
    if return_detailed_info:
        info_dict = {
            'main_integral': main_integral,
            'residual_correction': residual_correction,
            'residual_std': residual_std,
            'num_nodes': len(t_nodes),
            'num_MC_per_node': num_MC_per_node,
            'residual_samples': residual_samples,
            't_nodes': t_nodes,
            'E_values': E_values,
            'fitting_method': fitting_method,
            'sampling_distribution': p_s,
            'warmup_evaluations': warmup_evaluations,
            'total_function_evaluations': len(t_nodes) * num_MC_per_node + warmup_evaluations + residual_samples
        }
        return shapley_value, info_dict
    else:
        return shapley_value