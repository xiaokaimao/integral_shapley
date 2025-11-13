#!/usr/bin/env python
"""
Smart adaptive sampling methods for Shapley value computation.

This module contains intelligent local adaptive sampling with visualization support.
"""

import numpy as np
try:
    from scipy.integrate import simpson as _simpson
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from typing import Optional, Literal, Dict
from .base import compute_marginal_contribution_at_t



# def compute_integral_shapley_smart_adaptive(
#     x_train, y_train, x_valid, y_valid, i, clf, final_model,
#     utility_func, tolerance=1e-6, max_depth=10, num_MC=100,
#     rounding_method='probabilistic', min_samples_per_interval=3,
#     return_sampling_info=False
# ):
#     """
#     Smart-adaptive on scheme-B (n 等分、无系数 exact) with per-interval Simpson.

#     步骤
#       1) 把 [0,1] 先均分成若干区间（这里固定 20）
#       2) 每段用少量“探针”点估变化度，按档位分配该段的采样点数（强制奇数，>=min_samples_per_interval）
#       3) 在每段内按配额拆分估计 integrand：probabilistic 时把 num_MC 在层 s 与 s+1 按比例分配；round/floor/ceil 用单层
#       4) 对每段做 Simpson，累加得到 ∫_0^1 g_n(t) dt（方案二：**无 (n-1)/n 系数**）
#       5) 若 probabilistic，做端点校正 +(a0 - a_{n-1})/(2n)，直接复用积分阶段的端点值
#       6) 准确统计预算（积分 + 探针）

#     返回:
#       if return_sampling_info:
#           (shapley_value, actual_budget, sampling_info)
#       else:
#           (shapley_value, actual_budget)
#     """
#     rng = np.random.default_rng()
#     n = x_train.shape[0]
#     others = n - 1  # 把层 m 精准映到 t'=m/(n-1) 以取 a_m

#     # —— 工具：评估“层 m”的 a_m（固定取整），mc 是该层本次评估的 MC 次数 ——
#     def a_at_layer(m: int, mc: int) -> float:
#         t_prime = m / others
#         return compute_marginal_contribution_at_t(
#             t_prime, x_train, y_train, x_valid, y_valid, i, clf, final_model,
#             utility_func, num_MC=mc, rounding_method='round', rng=rng
#         )

#     # 固定区间划分（可替换为真正递归细分；这里保留简单稳定版本）
#     num_intervals = 20
#     intervals = [(k / num_intervals, (k + 1) / num_intervals) for k in range(num_intervals)]

#     sampling_info = {
#         'intervals': intervals,
#         'interval_info': [],
#         'all_t_values': [],
#         'all_integrand_values': [],
#         'interval_contributions': []
#     } if return_sampling_info else None

#     # —— 估计每段的“变化度”，并统计探针预算 ——
#     interval_variations = []
#     probe_points_per_interval = 5
#     probe_mc = min(10, num_MC // 2)  # 探针阶段用较小或中等 MC
#     probe_budget = 0

#     def compute_integrand_schemeB(t_values, mc_each):
#         """按方案二在给定 t 点集合上评估 integrand，mc_each 是该批评估每个 t 的总 MC。
#            返回数组，与本批次预算消耗（= len(t_values)*mc_each）。"""
#         vals = []
#         for t in t_values:
#             s = min(int(np.floor(n * t)), n - 1)
#             if rounding_method == 'probabilistic':
#                 delta = n * t - s
#                 if s == n - 1 or delta == 0.0:
#                     v = a_at_layer(s, mc_each)
#                 elif delta == 1.0:
#                     v = a_at_layer(s + 1, mc_each)
#                 else:
#                     m_s   = max(1, int(round((1.0 - delta) * mc_each)))
#                     m_sp1 = max(1, mc_each - m_s)
#                     v     = (1.0 - delta) * a_at_layer(s, m_s) + delta * a_at_layer(s + 1, m_sp1)
#             elif rounding_method in ('round', 'floor'):
#                 v = a_at_layer(s, mc_each)
#             elif rounding_method == 'ceil':
#                 v = a_at_layer(min(s + 1, n - 1), mc_each)
#             else:
#                 raise ValueError(f"Unknown rounding_method: {rounding_method}")
#             vals.append(float(v))
#         return np.array(vals, dtype=float), len(t_values) * mc_each

#     # 估变动度
#     for (a, b) in intervals:
#         t_probe = np.linspace(a, b, probe_points_per_interval)
#         probe_vals, used = compute_integrand_schemeB(t_probe, probe_mc)
#         probe_budget += used

#         if len(probe_vals) >= 3:
#             func_range = float(np.max(probe_vals) - np.min(probe_vals))
#             first_var  = float(np.sum(np.abs(np.diff(probe_vals))) / (b - a))
#             std_dev    = float(np.std(probe_vals))

#             # 二阶差分尺度
#             dt = (b - a) / (len(t_probe) - 1)
#             second = 0.0
#             for j in range(len(probe_vals) - 2):
#                 second = max(second, abs(probe_vals[j+2] - 2*probe_vals[j+1] + probe_vals[j]) / (dt**2))

#             score = func_range * 10 + second + first_var + std_dev * 5
#             interval_variations.append(score)
#         else:
#             interval_variations.append(0.0)

#     mean_var = float(np.mean(interval_variations)) if interval_variations else 0.0
#     high_th  = max(0.01, 2.0 * mean_var)
#     med_th   = max(0.005, 0.5 * mean_var)
#     low_th   = max(0.001, 0.1 * mean_var)

#     # —— 正式积分阶段：按变动度配点，用 Simpson 分段积分；同时记录 a0 与 a_{n-1} 以便端点校正 ——
#     total_integral = 0.0
#     total_points   = 0
#     a0_val = None
#     aN1_val = None

#     for idx, (a, b) in enumerate(intervals):
#         vscore = interval_variations[idx]
#         if vscore >= high_th:
#             base = 15
#         elif vscore >= med_th:
#             base = 7
#         elif vscore >= low_th:
#             base = 3
#         else:
#             base = 2

#         # base = max(base, min_samples_per_interval)
#         # if base % 2 == 0:  # Simpson 要奇数
#         #     base += 1
#         base = max(base,2)

#         t_vals = np.linspace(a, b, base, endpoint=True)
#         integ_vals, used = compute_integrand_schemeB(t_vals, num_MC)
#         total_points += base

#         # 记录端点 integrand 值（仅一次）
#         if a0_val is None:
#             a0_val = float(integ_vals[0])   # t=0 处的值
#         if idx == len(intervals) - 1:
#             aN1_val = float(integ_vals[-1]) # t=1 处的值

#         # 分段 Simpson
#         interval_I = np.trapz(integ_vals, t_vals)
#         total_integral += float(interval_I)

#         if return_sampling_info:
#             if vscore >= high_th:
#                 cat = "High"
#             elif vscore >= med_th:
#                 cat = "Medium"
#             elif vscore >= low_th:
#                 cat = "Low"
#             else:
#                 cat = "Minimal"

#             sampling_info['interval_info'].append({
#                 'interval': (a, b),
#                 'length': (b - a),
#                 'samples': base,
#                 'integral': float(interval_I),
#                 'variation': float(vscore),
#                 'category': cat,
#                 't_values': t_vals,
#                 'integrand_values': integ_vals
#             })
#             sampling_info['all_t_values'].extend(t_vals)
#             sampling_info['all_integrand_values'].extend(integ_vals)
#             sampling_info['interval_contributions'].append(float(interval_I))

#         # if idx < 5:
#         #     cat = "High" if vscore >= high_th else ("Medium" if vscore >= med_th else ("Low" if vscore >= low_th else "Minimal"))
#         #     print(f"  Interval [{a:.3f}, {b:.3f}]: variation={vscore:.2e} ({cat}), samples={base}, integral={interval_I:.6f}")

#     # 方案二：无 (n-1)/n 系数
#     shapley_value = float(total_integral)

#     # 概率取整下的端点校正，复用端点值
#     if rounding_method == 'probabilistic':
#         if a0_val is None or aN1_val is None:
#             # 理论上不会发生，因为我们每段都包含端点
#             # 为稳妥起见，如果缺失就补一次
#             a0_val  = a0_val  if a0_val  is not None else a_at_layer(0, num_MC)
#             aN1_val = aN1_val if aN1_val is not None else a_at_layer(n - 1, num_MC)
#         shapley_value += (a0_val - aN1_val) / (2.0 * n)

#     # 准确预算：积分 + 探针
#     integral_budget = total_points * num_MC
#     actual_budget   = integral_budget + probe_budget
#     # print(f"Smart adaptive sampling completed: {total_points} integral points, "
#     #       f"probe points per interval={probe_points_per_interval}, "
#     #       f"budget = integral {integral_budget} + probe {probe_budget} = {actual_budget}")

#     if return_sampling_info:
#         return shapley_value, actual_budget, sampling_info
#     else:
#         return shapley_value, actual_budget



def compute_integral_shapley_smart_adaptive(
    x_train, y_train, x_valid, y_valid, i, clf, final_model,
    utility_func,
    # 自适应参数
    tolerance: float = 1e-6,           # 这里只保留占位（当前分档用分位阈，不用递归）
    max_depth: int = 10,               # 占位
    num_MC: int = 100,                 # 每个 t 节点上做的 MC 次数
    rounding_method: str = 'round',    # 'round'/'floor'/'ceil'/'probabilistic'，直接传给你的 t 函数
    min_samples_per_interval: int = 3, # 每段至少多少个 t 节点
    return_sampling_info: bool = False,
    # 额外可选项
    num_intervals: int = 30,           # [0,1] 先均分多少段
    probe_points_per_interval: int = 5,# 每段用多少“探针点”估变化
    probe_mc: Optional[int] = None,    # 探针阶段每个 t 用多少 MC，默认 min(10, num_MC//2)
    integrator: str = 'trapezoid',       # 'simpson' 或 'trapezoid'
):
    """
    Smart-adaptive on scheme-B（n 等分映射，在 t 上直接估计 E[Δ(t,i)]）.

    思路：
      1) 先把 [0,1] 均分为 num_intervals 段
      2) 每段放 probe_points_per_interval 个 t 探针点，用较小 MC（probe_mc）评估 g(t) 以估“变化度”
      3) 按变化度分档为 High/Medium/Low/Minimal，并为该段分配正式积分节点数（强制 >= min_samples_per_interval）
      4) 正式阶段：在每段内等距取 t，直接调用 `compute_marginal_contribution_at_t(… rounding_method=…)` 评 g(t)
      5) 每段分别用数值积分（Simpson 或 Trapezoid），累加得到 ∫_0^1 g(t) dt
      6) 预算 = 探针消耗 + 正式积分消耗（节点数 × MC）

    返回:
      if return_sampling_info:
          (shapley_value, actual_budget, sampling_info)
      else:
          (shapley_value, actual_budget) 
    """
    rng = np.random.default_rng()
    # print(f"Smart adaptive sampling: num_intervals={num_intervals}, ")
    n = x_train.shape[0]
    if n < 2:
        # 退化情况：无可加入的“其他人”，即 Δ(t,i) 恒为 U({i}) - U(∅)
        # 简单起见直接在 t=0 评一次
        val = compute_marginal_contribution_at_t(
            0.0, x_train, y_train, x_valid, y_valid, i, clf, final_model,
            utility_func, num_MC=num_MC, rounding_method=rounding_method, rng=rng
        )
        if return_sampling_info:
            info = {
                'intervals': [(0.0, 1.0)],
                'interval_info': [{
                    'interval': (0.0, 1.0),
                    'length': 1.0,
                    'samples': 1,
                    'integral': float(val),
                    'variation': 0.0,
                    'category': 'Minimal',
                    't_values': np.array([0.0]),
                    'integrand_values': np.array([val]),
                }],
                'all_t_values': [0.0],
                'all_integrand_values': [val],
                'interval_contributions': [float(val)],
            }
            return float(val), int(num_MC), info
        return float(val), int(num_MC)

    # 探针 MC
    if probe_mc is None:
        probe_mc = max(1, min(20, num_MC // 2))
    # print(probe_mc)

    # 区间划分
    intervals = [(k / num_intervals, (k + 1) / num_intervals) for k in range(num_intervals)]

    sampling_info = None
    if return_sampling_info:
        sampling_info = {
            'intervals': intervals,
            'interval_info': [],
            'all_t_values': [],
            'all_integrand_values': [],
            'interval_contributions': []
        }

    # ---------- 小工具：在一批 t 上评估 g(t) ----------
    def eval_integrand_at_t_batch(t_array, mc_each):
        vals = []
        for t in t_array:
            v = compute_marginal_contribution_at_t(
                float(t), x_train, y_train, x_valid, y_valid, i, clf, final_model,
                utility_func, num_MC=mc_each, rounding_method=rounding_method, rng=rng
            )
            vals.append(float(v))
        return np.array(vals, dtype=float)

    # ---------- 1) 探针阶段：估每段的变化度 + 记录预算 ----------
    interval_variations = []
    probe_budget = 0

    for (a, b) in intervals:
        t_probe = np.linspace(a, b, probe_points_per_interval)
        probe_vals = eval_integrand_at_t_batch(t_probe, probe_mc)
        probe_budget += probe_points_per_interval * probe_mc

        if len(probe_vals) >= 3:
            func_range = float(np.max(probe_vals) - np.min(probe_vals))
            first_var = float(np.sum(np.abs(np.diff(probe_vals))) / max(b - a, 1e-12))
            std_dev = float(np.std(probe_vals))

            # 二阶差分尺度
            dt = (b - a) / (len(t_probe) - 1)
            max_second = 0.0
            for j in range(len(probe_vals) - 2):
                max_second = max(
                    max_second,
                    abs(probe_vals[j+2] - 2*probe_vals[j+1] + probe_vals[j]) / (dt**2 if dt > 0 else 1.0)
                )
            # 组合得分（与你原先一致的尺度感）
            score = func_range * 10 + max_second + first_var + std_dev * 5
            interval_variations.append(score)
        else:
            interval_variations.append(0.0)

    # 动态阈值
    mean_var = float(np.mean(interval_variations)) if interval_variations else 0.0
    high_th = max(0.01, 2.0 * mean_var)
    med_th  = max(0.005, 0.5 * mean_var)
    low_th  = max(0.001, 0.1 * mean_var)

    # ---------- 2) 正式积分阶段：按分档配点，逐段积分 ----------
    total_integral = 0.0
    integral_budget = 0
    all_t_vals = []     # 为了 info（可选）
    all_t_vals_out = [] # 同上
    for idx, (a, b) in enumerate(intervals):
        vscore = interval_variations[idx]
        if vscore >= high_th:
            base = 15
        elif vscore >= med_th:
            base = 7
        elif vscore >= low_th:
            base = 3
        else:
            base = 2

        base = max(base, min_samples_per_interval)

        # 选择积分方式
        use_simpson = (integrator.lower() == 'simpson') and _HAS_SCIPY
        if use_simpson:
            # Simpson 需要奇数且 >= 3
            base = max(base, 3)
            if base % 2 == 0:
                base += 1
        else:
            # 梯形至少 2 个点
            base = max(base, 2)

        t_vals = np.linspace(a, b, base, endpoint=True)
        vals = eval_integrand_at_t_batch(t_vals, num_MC)

        # 分段积分
        if use_simpson:
            interval_I = _simpson(vals, t_vals)
        else:
            interval_I = np.trapz(vals, t_vals)

        total_integral += float(interval_I)
        integral_budget += base * num_MC

        if return_sampling_info:
            # 分类标签
            if vscore >= high_th:
                cat = "High"
            elif vscore >= med_th:
                cat = "Medium"
            elif vscore >= low_th:
                cat = "Low"
            else:
                cat = "Minimal"

            sampling_info['interval_info'].append({
                'interval': (a, b),
                'length': (b - a),
                'samples': int(base),
                'integral': float(interval_I),
                'variation': float(vscore),
                'category': cat,
                't_values': t_vals,
                'integrand_values': vals
            })
            sampling_info['interval_contributions'].append(float(interval_I))
            sampling_info['all_t_values'].extend(t_vals)
            sampling_info['all_integrand_values'].extend(vals)

    shapley_value = float(total_integral)
    actual_budget = int(probe_budget + integral_budget)

    if return_sampling_info:
        return shapley_value, actual_budget, sampling_info
    else:
        return shapley_value, actual_budget


def compute_integral_cc_smart_adaptive(
    x_train,
    y_train,
    x_valid,
    y_valid,
    clf,
    final_model,
    utility_func,
    num_intervals: int = 16,
    probe_layers_per_interval: int = 3,
    probe_mc: int = 8,
    base_layers_per_interval: int = 3,
    boost_high: int = 3,
    boost_medium: int = 1,
    boost_low: int = 0,
    num_MC: int = 60,
    aggregator: Literal["linear", "voronoi"] = "linear",
    num_processes: Optional[int] = None,
    base_seed: int = 2048,
    return_sampling_info: bool = False,
):
    """
    自适应版稀疏 CC：在 t 轴上分区，先做探针层 wave，再按波动度重新分配层节点与 MC。
    返回 (shapley, info) 若 return_sampling_info=True，否则 (shapley, None)。
    """
    from .cc_methods import _sparse_cc_layer_worker  # 避免循环依赖
    import multiprocessing as mp

    if num_processes is None:
        num_processes = mp.cpu_count()

    n = x_train.shape[0]
    if n <= 1:
        shapley = np.zeros(n)
        info = {'reason': 'degenerate_dataset'} if return_sampling_info else None
        return (shapley, info) if return_sampling_info else shapley

    def _interval_to_layers(a: float, b: float) -> tuple[int, int]:
        lo = max(1, int(np.floor(a * n)))
        hi = max(lo, int(np.ceil(b * n)))
        return lo, min(hi, n)

    intervals = [(k / num_intervals, (k + 1) / num_intervals) for k in range(num_intervals)]

    # Pilot 阶段
    pilot_layers = set()
    interval_pilot_layers = []
    for (a, b) in intervals:
        lo, hi = _interval_to_layers(a, b)
        if hi < lo:
            interval_pilot_layers.append([])
            continue
        pts = np.linspace(lo, hi, max(1, probe_layers_per_interval), dtype=int)
        uniq = np.unique(pts)
        interval_pilot_layers.append(uniq.tolist())
        pilot_layers.update(uniq.tolist())

    if not pilot_layers:
        pilot_layers = set(np.linspace(1, n, min(n, probe_layers_per_interval * num_intervals)).astype(int))

    nodes_set = set()
    for j in pilot_layers:
        nodes_set.add(int(j))
        jj = n - int(j)
        if 1 <= jj <= n:
            nodes_set.add(jj)
    pilot_nodes = np.array(sorted(nodes_set), dtype=int)
    pilot_pos = {int(j): idx for idx, j in enumerate(pilot_nodes)}

    pilot_tasks = []
    for j in pilot_layers:
        seed = int((base_seed ^ (j * 1315423911)) & 0x7FFFFFFF)
        pilot_tasks.append(
            (
                int(j),
                int(probe_mc),
                seed,
                x_train,
                y_train,
                x_valid,
                y_valid,
                clf,
                final_model,
                utility_func,
                n,
                pilot_pos,
            )
        )

    pilot_sum = np.zeros((n, len(pilot_nodes)), dtype=float)
    pilot_cnt = np.zeros((n, len(pilot_nodes)), dtype=int)
    if pilot_tasks:
        with mp.Pool(processes=num_processes) as pool:
            for (j, col_j, sum_j, cnt_j, col_jj, sum_jj, cnt_jj) in pool.imap_unordered(_sparse_cc_layer_worker, pilot_tasks):
                if col_j is not None:
                    pilot_sum[:, col_j] += sum_j
                    pilot_cnt[:, col_j] += cnt_j
                if col_jj is not None:
                    pilot_sum[:, col_jj] += sum_jj
                    pilot_cnt[:, col_jj] += cnt_jj

    with np.errstate(divide="ignore", invalid="ignore"):
        pilot_g = np.where(pilot_cnt > 0, pilot_sum / pilot_cnt, np.nan)

    interval_scores = []
    for idx, (a, b) in enumerate(intervals):
        layer_list = interval_pilot_layers[idx]
        signals = []
        for layer in layer_list:
            col = pilot_pos.get(int(layer))
            if col is None:
                continue
            g = pilot_g[:, col]
            if np.all(np.isnan(g)):
                continue
            signals.append(np.nanmean(g))
        if len(signals) >= 2:
            arr = np.array(signals, dtype=float)
            func_range = float(np.nanmax(arr) - np.nanmin(arr))
            slope = float(np.nanmean(np.abs(np.diff(arr)))) / max(b - a, 1e-6)
            interval_scores.append(func_range * 5 + slope * 2)
        elif len(signals) == 1:
            interval_scores.append(abs(float(signals[0])))
        else:
            interval_scores.append(0.0)

    # 让对称区间共享波动度（例如 [0,0.1] 与 [0.9,1.0]）
    symmetric_scores = interval_scores.copy()
    m = len(intervals)
    for idx in range(m):
        partner = m - 1 - idx
        if partner < idx:
            continue
        pair_score = max(interval_scores[idx], interval_scores[partner])
        symmetric_scores[idx] = pair_score
        symmetric_scores[partner] = pair_score

    mean_score = float(np.mean(symmetric_scores)) if symmetric_scores else 0.0
    high_th = max(1e-4, 2.0 * mean_score)
    med_th = max(5e-5, 0.6 * mean_score)

    layer_schedule: Dict[int, Dict[str, float]] = {}
    interval_details = []
    for idx, (a, b) in enumerate(intervals):
        lo, hi = _interval_to_layers(a, b)
        score = symmetric_scores[idx]
        if hi < lo:
            interval_details.append({"interval": (a, b), "layers": [], "category": "minimal", "score": score})
            continue

        if score >= high_th:
            category = "high"
            num_layers = base_layers_per_interval + boost_high
        elif score >= med_th:
            category = "medium"
            num_layers = base_layers_per_interval + boost_medium
        elif score > 0:
            category = "low"
            num_layers = max(1, base_layers_per_interval + boost_low)
        else:
            category = "minimal"
            num_layers = max(1, base_layers_per_interval)

        num_layers = int(max(1, num_layers))
        chosen_layers = np.unique(np.linspace(lo, hi, num_layers, dtype=int))
        for layer in chosen_layers:
            layer = int(np.clip(layer, 1, n))
            layer_schedule[layer] = {"category": category, "interval_index": idx}

        interval_details.append(
            {
                "interval": (a, b),
                "layers": chosen_layers.tolist(),
                "category": category,
                "score": score,
                "num_layers": num_layers,
            }
        )

    nodes_set = set()
    for layer in layer_schedule:
        nodes_set.add(int(layer))
        jj = n - int(layer)
        if 1 <= jj <= n:
            nodes_set.add(jj)
    nodes_sorted = np.array(sorted(nodes_set), dtype=int)
    pos = {int(j): idx for idx, j in enumerate(nodes_sorted)}

    sum_mat = np.zeros((n, len(nodes_sorted)), dtype=float)
    cnt_mat = np.zeros((n, len(nodes_sorted)), dtype=int)
    tasks = []
    for layer, meta in layer_schedule.items():
        seed = int((base_seed + layer * 2654435761) & 0x7FFFFFFF)
        tasks.append(
            (
                int(layer),
                int(num_MC),
                seed,
                x_train,
                y_train,
                x_valid,
                y_valid,
                clf,
                final_model,
                utility_func,
                n,
                pos,
            )
        )

    if tasks:
        with mp.Pool(processes=num_processes) as pool:
            for (j, col_j, sum_j, cnt_j, col_jj, sum_jj, cnt_jj) in pool.imap_unordered(_sparse_cc_layer_worker, tasks):
                if col_j is not None:
                    sum_mat[:, col_j] += sum_j
                    cnt_mat[:, col_j] += cnt_j
                if col_jj is not None:
                    sum_mat[:, col_jj] += sum_jj
                    cnt_mat[:, col_jj] += cnt_jj

    with np.errstate(divide="ignore", invalid="ignore"):
        g_mat = np.where(cnt_mat > 0, sum_mat / cnt_mat, np.nan)

    t_nodes = nodes_sorted.astype(float) / n
    shapley = np.zeros(n, dtype=float)
    if aggregator == "voronoi":
        bounds = np.empty(len(t_nodes) + 1, dtype=float)
        bounds[0] = 0.0
        bounds[-1] = 1.0
        if len(t_nodes) > 1:
            bounds[1:-1] = 0.5 * (t_nodes[:-1] + t_nodes[1:])
        weights = np.diff(bounds)
        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                continue
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask], left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.sum(weights * g_fill))
    else:
        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                continue
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask], left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.trapz(g_fill, t_nodes))

    if return_sampling_info:
        info = {
            "intervals": intervals,
            "interval_details": interval_details,
            "layer_schedule": layer_schedule,
            "pilot_layers": sorted(pilot_layers),
            "nodes": nodes_sorted.tolist(),
            "t_nodes": t_nodes.tolist(),
            "pilot_budget": probe_mc * len(pilot_layers),
            "integral_budget": num_MC * len(layer_schedule) / n,
            "aggregator": aggregator,
        }
        return shapley, info

    return shapley


def visualize_smart_adaptive_sampling(sampling_info, data_point_index, save_path=None):
    """
    Visualize the adaptive sampling pattern used by Smart Adaptive method.
    
    Args:
        sampling_info: Dictionary containing sampling information from smart_adaptive method
        data_point_index: Index of the data point being analyzed
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Simple Smart Adaptive Sampling Pattern
    ax1.set_title(f'Simple Smart Adaptive Sampling for Data Point {data_point_index}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t (Coalition Proportion)')
    ax1.set_ylabel('Sampling Points')
    
    # Use categories to determine colors - clearer visualization
    category_colors = {
        'High': '#FF6B6B',      # Red - High variation
        'Medium': '#4ECDC4',    # Teal - Medium variation  
        'Low': '#45B7D1',       # Blue - Low variation
        'Minimal': '#96CEB4'    # Green - Minimal variation
    }
    
    for idx, interval_info in enumerate(sampling_info['interval_info']):
        a, b = interval_info['interval']
        samples = interval_info['samples']
        length = interval_info['length']
        variation = interval_info.get('variation', 0)
        category = interval_info.get('category', 'Minimal')
        
        # Height represents number of sampling points directly
        height = samples
        
        # Draw rectangle representing interval
        rect = patches.Rectangle((a, 0), length, height, 
                               linewidth=1, edgecolor='black', 
                               facecolor=category_colors.get(category, '#96CEB4'), alpha=0.8)
        ax1.add_patch(rect)
        
        # Add text showing number of samples and category
        ax1.text(a + length/2, height/2, f'{samples}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(a + length/2, height*0.8, f'{category}', 
                ha='center', va='center', fontsize=8, style='italic')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max([info['samples'] for info in sampling_info['interval_info']]) * 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for categories
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=category) 
                      for category, color in category_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Plot 2: Sampling points and integrand values
    ax2.set_title('Integrand Function and Sampling Points', fontsize=12)
    ax2.set_xlabel('t (Coalition Proportion)')
    ax2.set_ylabel('Integrand Value E[Δ(t,i)]')
    
    # Plot integrand values for each interval
    for idx, interval_info in enumerate(sampling_info['interval_info']):
        t_vals = interval_info['t_values']
        integrand_vals = interval_info['integrand_values']
        category = interval_info.get('category', 'Minimal')
        color = category_colors.get(category, '#96CEB4')
        
        # Plot line for this interval
        ax2.plot(t_vals, integrand_vals, 'o-', color=color, 
                linewidth=2, markersize=4, alpha=0.8)
        
        # Highlight sampling points
        ax2.scatter(t_vals, integrand_vals, color=color, 
                   s=30, zorder=5, alpha=0.9)
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Interval contributions to total integral
    ax3.set_title('Interval Contributions to Total Shapley Value', fontsize=12)
    ax3.set_xlabel('Interval Index')
    ax3.set_ylabel('Interval Contribution')
    
    interval_contributions = sampling_info['interval_contributions']
    interval_indices = range(len(interval_contributions))
    
    # Color bars by category
    bar_colors = [category_colors.get(info.get('category', 'Minimal'), '#96CEB4') 
                  for info in sampling_info['interval_info']]
    bars = ax3.bar(interval_indices, interval_contributions, 
                  color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, contrib) in enumerate(zip(bars, interval_contributions)):
        if abs(contrib) > max(abs(min(interval_contributions)), max(interval_contributions)) * 0.1:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                    (0.02 if contrib >= 0 else -0.02) * max(interval_contributions),
                    f'{contrib:.4f}', ha='center', va='bottom' if contrib >= 0 else 'top',
                    fontsize=8)
    
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Add summary statistics
    total_intervals = len(sampling_info['intervals'])
    total_samples = sum([info['samples'] for info in sampling_info['interval_info']])
    total_shapley = sum(interval_contributions)
    
    fig.suptitle(f'Smart Adaptive Sampling Visualization\\n'
                f'Total Intervals: {total_intervals}, Total Sampling Points: {total_samples}, '
                f'Shapley Value: {total_shapley:.6f}', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Smart Adaptive sampling visualization saved to: {save_path}")
    
    plt.show()
