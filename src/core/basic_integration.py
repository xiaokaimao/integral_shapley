#!/usr/bin/env python
"""
Basic integration methods for Shapley value computation.

This module contains trapezoid, Gaussian quadrature, and Simpson integration methods.
"""

import numpy as np
from scipy.integrate import simpson, fixed_quad
from tqdm import tqdm

from .base import compute_marginal_contribution_at_t, choose_optimal_t_samples


# def compute_integral_shapley_trapezoid(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_t_samples=50, num_MC=100, rounding_method='probabilistic'
# ):
#     """
#     用等距 t + 复合梯形公式估计 ∫_0^1 f(t) dt，
#     返回 ((n-1)/n) * 该积分。
#     若 rounding_method='probabilistic'，对积分结果做端点校正。
#     """
#     rng = np.random.default_rng()

#     n = x_train.shape[0]
#     coef = (n - 1) / n  # 关键系数

#     # 等距 t 节点
#     t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)

#     # 被积函数估计
#     integrand = []
#     for t in t_values:
#         val = compute_marginal_contribution_at_t(
#             t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=num_MC, rounding_method=rounding_method, rng=rng
#         )
#         integrand.append(val)

#     # 复合梯形积分
#     I = np.trapz(y=np.array(integrand, dtype=float), x=t_values)
#     shapley_est = coef * I

#     # 概率取整时做端点校正：加 (a_0 + a_{n-1}) / (2n)
#     #
#     # 数学原理：概率取整产生分段线性插值 f_n(t)，其积分等价于梯形权重求和：
#     #   ∫₀¹ f_n(t) dt = (1/(n-1)) * [a₀/2 + a₁ + ... + a_{n-2} + a_{n-1}/2]
#     # 而真实Shapley值是等权平均：
#     #   SV_i = (1/n) * [a₀ + a₁ + ... + a_{n-1}]
#     # 两者差值恰好是端点校正项：(a₀ + a_{n-1})/(2n)
#     #
#     # 注：端点处概率取整退化为确定性（t=0→s=0, t=1→s=n-1），故直接复用已算好的端点值
#     if rounding_method == 'probabilistic':
#         a0 = float(integrand[0])   # 已计算的 a_0
#         aN1 = float(integrand[-1])  # 已计算的 a_{n-1}
#         shapley_est += (a0 + aN1) / (2.0 * n)

#     elif rounding_method in ('round', 'floor', 'ceil'):
#         # 固定取整：直接返回，不做端点校正
#         pass
#     else:
#         raise ValueError(f"Unknown rounding_method: {rounding_method}")

#     return float(shapley_est)

# def compute_integral_shapley_trapezoid(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_t_samples=50, num_MC=100, rounding_method='probabilistic'
# ):
#     """
#     方案二：把 [0,1] 均分成 n 段（每段 1/n），
#     g_n(t)=a_s 在 [s/n,(s+1)/n) 上常数（最后一段 s=n-1）。
#     用复合梯形近似 ∫_0^1 g_n(t) dt，**无 (n-1)/n 系数**。
#     若 rounding_method='probabilistic'，按两点线性混合并做端点校正 +(a0 - a_{n-1})/(2n)。
#     """
#     rng = np.random.default_rng()
#     n = x_train.shape[0]
#     others = n - 1  # 用于把“层 m”映回 t' = m/(n-1) 以精确取该层

#     # 工具：拿“层 m”的 a_m（通过 t'=m/(n-1) + 固定取整 round）
#     def a_at_layer(m, mc):
#         t_prime = m / others
#         return compute_marginal_contribution_at_t(
#             t_prime, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=mc , rounding_method='round', rng=rng
#         )

#     # 等距 t（方案二：层索引的均匀坐标）
#     t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)

#     integrand = []
#     for t in t_values:
#         s = min(int(np.floor(n * t)), n - 1)
#         delta = n * t - s

#         if rounding_method == 'probabilistic':
#             if s == n - 1 or delta == 0.0:
#                 val = a_at_layer(s, num_MC)                    # 只用末层/当前层
#             elif delta == 1.0:
#                 val = a_at_layer(s + 1, num_MC)                # 只用下一层
#             else:
#                 m_s   = max(1, int(round((1.0 - delta) * num_MC)))
#                 m_sp1 = max(1, num_MC - m_s)                   # 保证总和为 num_MC
#                 a_s   = a_at_layer(s,     m_s)
#                 a_sp1 = a_at_layer(s + 1, m_sp1)
#                 val = (1.0 - delta) * a_s + delta * a_sp1    

#         elif rounding_method == 'round':
#             val = a_at_layer(s)

#         elif rounding_method == 'floor':
#             val = a_at_layer(s)

#         elif rounding_method == 'ceil':
#             val = a_at_layer(min(s + 1, n - 1))

#         else:
#             raise ValueError(f"Unknown rounding_method: {rounding_method}")

#         integrand.append(val)

#     # 复合梯形积分（**无全局系数**）
#     I = np.trapz(y=np.array(integrand, dtype=float), x=t_values)
#     shapley_est = I

#     # 概率取整下的端点校正：+(a0 - a_{n-1})/(2n)
#     if rounding_method == 'probabilistic':
#         a0  = a_at_layer(0)
#         aN1 = a_at_layer(n - 1)
#         shapley_est += (a0 - aN1) / (2.0 * n)

#     return float(shapley_est)
import numpy as np

# def compute_integral_shapley_trapezoid(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_t_samples=50, num_MC=100, rounding_method='probabilistic'
# ):
#     """
#     方案二（n 等分、无系数 exact）+ 复合梯形法。
#     - t → 层 s = floor(n * t)，最后一段 s = n-1。
#     - probabilistic：在段内按 delta = n*t - s，把 num_MC 按 (1-delta, delta) 分给层 s 与 s+1
#       （配额拆分，不翻倍），再线性混合。
#     - 端点校正（仅 probabilistic）：+(a0 - a_{n-1})/(2n)，直接复用 integrand 的端点值。
#     - round/floor/ceil：按对应层取 a_s（或 a_{s+1}），不做端点校正。
#     """
#     if num_t_samples < 2:
#         num_t_samples = 2  # 梯形法至少两个节点

#     rng = np.random.default_rng()
#     n = x_train.shape[0]
#     others = n - 1  # 层 m → t' = m/(n-1) 用于“精准取层”

#     # 评估“层 m”的 a_m（通过 t' = m/(n-1) + 固定取整 'round'）
#     def a_at_layer(m: int, mc: int) -> float:
#         t_prime = m / others
#         return compute_marginal_contribution_at_t(
#             t_prime, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=mc, rounding_method='round', rng=rng
#         )

#     # n 等分坐标上的 t 网格
#     t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)
#     integrand = []

#     for t in t_values:
#         s = min(int(np.floor(n * t)), n - 1)

#         if rounding_method == 'probabilistic':
#             delta = n * t - s

#             # 端点/整点：只用单层
#             if s == n - 1 or delta == 0.0:
#                 val = a_at_layer(s, num_MC)
#             elif delta == 1.0:
#                 val = a_at_layer(s + 1, num_MC)
#             else:
#                 # 配额拆分：两层合计评估次数仍为 num_MC
#                 m_s   = max(1, int(round((1.0 - delta) * num_MC)))
#                 m_sp1 = max(1, num_MC - m_s)
#                 # print(m_s + m_sp1)
#                 a_s   = a_at_layer(s,     m_s)
#                 a_sp1 = a_at_layer(s + 1, m_sp1)
#                 val   = (1.0 - delta) * a_s + delta * a_sp1

#         elif rounding_method in ('round', 'floor'):
#             val = a_at_layer(s, num_MC)

#         elif rounding_method == 'ceil':
#             val = a_at_layer(min(s + 1, n - 1), num_MC)

#         else:
#             raise ValueError(f"Unknown rounding_method: {rounding_method}")

#         integrand.append(float(val))

#     # 复合梯形积分（方案二：无 (n-1)/n 系数）
#     I = np.trapz(y=np.array(integrand, dtype=float), x=t_values)
#     shapley_est = float(I)

#     # 概率取整：端点校正 +(a0 - a_{n-1})/(2n)，复用端点 integrand
#     if rounding_method == 'probabilistic':
#         a0  = float(integrand[0])      # a_0
#         aN1 = float(integrand[-1])     # a_{n-1}
#         shapley_est += (a0 - aN1) / (2.0 * n)

#     return shapley_est



# def compute_integral_shapley_simpson(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_t_samples=51, num_MC=100, rounding_method='probabilistic'
# ):
#     """
#     Shapley ≈ ((n-1)/n) * ∫_0^1 f(t) dt，用 Simpson（若无 SciPy 则退化为 trapz）。
#     概率取整时自动做端点校正；固定取整时不校正。
#     """
#     # Simpson 要奇数个节点；若给偶数，补成奇数
#     if num_t_samples % 2 == 0:
#         num_t_samples += 1

#     rng = np.random.default_rng()
#     n = x_train.shape[0]
#     coef = (n - 1) / n

#     t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)
#     integrand = [
#         compute_marginal_contribution_at_t(
#             t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=num_MC, rounding_method=rounding_method, rng=rng
#         )
#         for t in t_values
#     ]


#     I = simpson(y=np.array(integrand, dtype=float), x=t_values)


#     shapley_est = coef * I

#     # 概率取整：端点校正 + (a0 + a_{n-1}) / (2n)
#     if rounding_method == 'probabilistic':
#         a0  = compute_marginal_contribution_at_t(
#             0.0, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=num_MC, rounding_method='round', rng=rng
#         )
#         aN1 = compute_marginal_contribution_at_t(
#             1.0, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=num_MC, rounding_method='round', rng=rng
#         )
#         shapley_est += (a0 + aN1) / (2.0 * n)

#     elif rounding_method in ('round', 'floor', 'ceil'):
#         pass
#     else:
#         raise ValueError(f"Unknown rounding_method: {rounding_method}")

#     return float(shapley_est)


def compute_integral_shapley_trapezoid(
    x_train, y_train, x_valid, y_valid, i, clf, final_model,
    utility_func, num_t_samples=50, num_MC=100, rounding_method='round'
):
    """
    等距 t 网格 + 直接在 t 上评估（底层按 n 分区 + round 锁层）+ 复合梯形积分
    - 不做层内线性混合
    - 不做端点校正
    - 仅用层号做缓存，避免同层重复训练
    依赖：compute_marginal_contribution_at_t 采用方案B（按 n 分区 + round）
    """
    import numpy as np

    rng = np.random.default_rng()
    n = x_train.shape[0]
    K = max(2, int(num_t_samples))

    # 等距 t（把 1 略微往回夹一下，避免 round(n*t)=n 的边界）
    t_values = np.linspace(0.0, 1.0, K, endpoint=True)
    t_values[-1] = np.nextafter(1.0, 0.0)


    integrand = []

    for t in t_values:
        # print(f"Evaluating at t={t:.4f}",f"Corresponding layer: {round(n*t)}")
        # 直接在 t 上评估（底层会按 n 分区 + round(n*t) 锁到层 m）
        val = compute_marginal_contribution_at_t(
                t, x_train, y_train, x_valid, y_valid, i, clf, final_model,
                utility_func, num_MC=num_MC, rounding_method='round', rng=rng
            )

        integrand.append(float(val))

    # 复合梯形积分（无系数、无校正）
    shapley_est = float(np.trapz(np.asarray(integrand, dtype=float), x=t_values))
    return shapley_est



def compute_integral_shapley_simpson(
    x_train, y_train, x_valid, y_valid, i, clf, final_model,
    utility_func, num_t_samples=51, num_MC=100, rounding_method='round'
):
    """
    方案二：n 等分、无系数 exact 的 Simpson 求积实现（纯 round 取整）。
    - 直接在 t 上调用 compute_marginal_contribution_at_t(..., rounding_method='round')
    - 不做概率配额拆分、不做端点校正
    - Simpson 要奇数个节点；若给偶数，自动 +1
    """
    # Simpson 需要奇数节点
    if num_t_samples % 2 == 0:
        num_t_samples += 1

    rng = np.random.default_rng()

    # 等距 t 网格（含端点 0 和 1）
    t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)
    t_values[-1] = np.nextafter(1.0, 0.0)
    # 在 t 上直接评估 integrand：round 将 t 映射到最近层规模
    integrand = [
        compute_marginal_contribution_at_t(
            float(t), x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC=num_MC, rounding_method='round', rng=rng
        )
        for t in t_values
    ]
    shapley_est = float(simpson(np.asarray(integrand, dtype=float), x=t_values))
    # Simpson 积分（无 (n-1)/n 系数、无端点校正）
    return shapley_est


# def compute_integral_shapley_simpson(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, num_t_samples=51, num_MC=100, rounding_method='probabilistic'
# ):
#     """
#     方案二（n 等分、无系数 exact）+ Simpson 求积 + 概率取整的配额拆分（不翻倍）。
#     - t 映射层：s = floor(n * t)，最后一段 s = n-1。
#     - probabilistic：在段内按 delta = n*t - s，把 num_MC 按 (1-delta, delta) 分给层 s 和 s+1，
#       分别估计 a_s 与 a_{s+1} 后做线性混合（两层合计评估次数仍为 num_MC）。
#     - 端点校正（仅 probabilistic）：+(a0 - a_{n-1})/(2n)，其中 a0 与 a_{n-1} 直接复用 integrand 的端点值。
#     - round/floor/ceil：按对应层取 a_s（或 a_{s+1}），不做端点校正。
#     """
#     # Simpson 要奇数个节点
#     if num_t_samples % 2 == 0:
#         num_t_samples += 1

#     rng = np.random.default_rng()
#     n = x_train.shape[0]
#     others = n - 1  # 把层 m 映回 t' = m/(n-1) 以“精准取层”

#     # 工具：评估“层 m”的 a_m（通过 t' = m/(n-1) + 固定取整 'round'）
#     def a_at_layer(m: int, mc: int) -> float:
#         t_prime = m / others
#         return compute_marginal_contribution_at_t(
#             t_prime, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=mc, rounding_method='round', rng=rng
#         )

#     # n 等分的 t 网格
#     t_values = np.linspace(0.0, 1.0, num_t_samples, endpoint=True)
#     integrand = []

#     for t in t_values:
#         s = min(int(np.floor(n * t)), n - 1)
#         if rounding_method == 'probabilistic':
#             delta = n * t - s

#             # 端点/整点：只用单层，避免另一层的样本数变成 0 或强行 ≥1 导致偏差
#             if s == n - 1 or delta == 0.0:
#                 val = a_at_layer(s, num_MC)
#             elif delta == 1.0:
#                 val = a_at_layer(s + 1, num_MC)
#             else:
#                 # 配额拆分：两层合计评估次数仍为 num_MC
#                 m_s   = max(1, int(round((1.0 - delta) * num_MC)))
#                 m_sp1 = max(1, num_MC - m_s)  # 保证非零且合计为 num_MC
#                 a_s   = a_at_layer(s,     m_s)
#                 a_sp1 = a_at_layer(s + 1, m_sp1)
#                 val   = (1.0 - delta) * a_s + delta * a_sp1

#         elif rounding_method in ('round', 'floor'):
#             # 与分段常数 g_n(t)=a_s 一致
#             val = a_at_layer(s, num_MC)

#         elif rounding_method == 'ceil':
#             # 右端点常数（一般只在需要一侧保守界时使用）
#             val = a_at_layer(min(s + 1, n - 1), num_MC)

#         else:
#             raise ValueError(f"Unknown rounding_method: {rounding_method}")

#         integrand.append(float(val))

#     # Simpson 积分（方案二：**无 (n-1)/n 系数**）
#     I = simpson(y=np.array(integrand, dtype=float), x=t_values)
#     shapley_est = float(I)

#     # 概率取整：端点校正 +(a0 - a_{n-1})/(2n)，直接复用 integrand 端点
#     if rounding_method == 'probabilistic':
#         a0  = float(integrand[0])       # = a_0
#         aN1 = float(integrand[-1])      # = a_{n-1}
#         shapley_est += (a0 - aN1) / (2.0 * n)

#     return shapley_est


def compute_integral_shapley_auto(x_train, y_train, x_valid, y_valid, i, clf, final_model,
                                 utility_func, method='simpson', precision='balanced', num_MC=100,
                                 rounding_method='probabilistic'):
    """
    Automatically choose optimal parameters and compute Shapley value.
    
    Args:
        x_train, y_train: Training data
        x_valid, y_valid: Validation data
        i: Target data point index
        clf: Classifier to train on subsets  
        final_model: Model trained on full data
        utility_func: Utility function
        method: 'trapezoid' or 'simpson'
        precision: 'fast', 'balanced', or 'high'
        num_MC: Monte Carlo samples per t value
        rounding_method: How to round coalition sizes
        
    Returns:
        shapley_value: Estimated Shapley value for data point i
    """
    # Choose optimal t sampling points
    optimal_t = choose_optimal_t_samples(len(x_train), method, precision)
    
    print(f"Auto-selected {optimal_t} t-samples for {method} method ({precision} precision)")
    
    # Call appropriate integration method
    if method == 'simpson':
        return compute_integral_shapley_simpson(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=optimal_t, num_MC=num_MC, 
            rounding_method=rounding_method
        )
    else:
        return compute_integral_shapley_trapezoid(
            x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_t_samples=optimal_t, num_MC=num_MC, 
            rounding_method=rounding_method
        )