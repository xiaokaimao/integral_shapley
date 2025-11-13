#!/usr/bin/env python
"""
Complementary Contribution (CC) methods for Shapley value computation.

This module implements various CC-based methods including:
- Basic stratified CC sampling
- Parallel CC computation
- CC integral methods with layer-wise parallelization
"""

import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn.base import clone
from typing import Optional, Tuple, Literal
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.model_utils import return_model



def cc_shapley(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    clf,
    final_model,
    utility_func,
    num_MC: int = 100,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Stratified MC (|S|=j) + 同步记录所有玩家的 CC，符合原文公式.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = x_train.shape[0]
    indices = np.arange(n)

    # 累加器：cc_sum[i,j] 记录第 j 层 CC 总和，cc_cnt[i,j] 记录样本数
    cc_sum  = np.zeros((n, n + 1))   # j 从 1..n，用 j 作列索引
    cc_cnt  = np.zeros((n, n + 1), dtype=int)

    for j in tqdm(range(1, n + 1)):# 层 j
        for _ in range(num_MC):
            S_idx = rng.choice(indices, size=j, replace=False)
            comp_idx = np.setdiff1d(indices, S_idx, assume_unique=True)

            # 计算 u = U(S) - U(N \ S)
            clf_s = clone(clf)
            clf_c = clone(clf)

            try:
                u_s = utility_func(x_train[S_idx],  y_train[S_idx],
                                       x_valid, y_valid, clf_s, final_model)
            except:
                u_s = 0.0
            
            try:
                u_c = utility_func(x_train[comp_idx], y_train[comp_idx],
                                       x_valid, y_valid, clf_c, final_model)
            except:
                u_c = 0.0
            

            u = u_s - u_c                      # CC_N(S)

            # ---- 把这一条样本同时写进两层 --------------------------
            # 对 S 中玩家：层 j 贡献 +u
            cc_sum[S_idx,  j] += u
            cc_cnt[S_idx,  j] += 1

            # 对补集玩家：层 n-j 贡献 -u (因为 CC_N(T)= -u)
            jj = n - j
            if jj > 0:  # when j = n, comp_idx is empty

                cc_sum[comp_idx, jj] += -u
                cc_cnt[comp_idx, jj] += 1
            # ------------------------------------------------------

    cc_mean = np.full_like(cc_sum, np.nan, dtype=float)

    # 对于有样本的地方，计算平均值
    mask = cc_cnt > 0
    cc_mean[mask] = cc_sum[mask] / cc_cnt[mask]

    # 跳过 j=0 列，用 nanmean 自动忽略 cc_cnt==0 的位置
    sv = np.nanmean(cc_mean[:, 1:], axis=1)
    return sv 


def _cc_layer_worker(args):
    """
    Worker function for parallel CC layer processing.
    Each process handles one coalition size (layer) with all its MC samples.
    
    Args:
        args: Tuple containing (j, num_MC, seed, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func)
    
    Returns:
        Tuple: (j, list_of_contributions)
        where list_of_contributions contains (S_idx, comp_idx, cc_value) tuples
    """
    j, num_MC, seed, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func = args
    
    # Create independent random state for this layer
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    indices = np.arange(n)
    
    layer_contributions = []
    
    # Process all MC samples for this coalition size j
    for _ in range(num_MC):
        # print("i am here")
        try:
            # Sample coalition S of size j
            S_idx = rng.choice(indices, size=j, replace=False)
            comp_idx = np.setdiff1d(indices, S_idx, assume_unique=True)
            
            # Calculate CC_N(S) = U(S) - U(N\\S)
            clf_s = clone(clf)
            clf_c = clone(clf)
            
            try:
                u_s = utility_func(x_train[S_idx], y_train[S_idx], 
                                  x_valid, y_valid, clf_s, final_model)
            except:
                u_s = 0.0
                
            try:
                u_c = utility_func(x_train[comp_idx], y_train[comp_idx], 
                                  x_valid, y_valid, clf_c, final_model)
            except:
                u_c = 0.0
                
            cc_value = u_s - u_c
            layer_contributions.append((S_idx, comp_idx, cc_value))
            
        except Exception:
            # Add zero contribution if sampling fails
            layer_contributions.append((np.array([]), np.array([]), 0.0))
    
    return j, layer_contributions


def _full_cc_layer_worker(args):
    j, mc, seed, x_train, y_train, x_valid, y_valid, clf, final_model, utility_func, n = args
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    sum_j  = np.zeros(n); cnt_j  = np.zeros(n, dtype=int)
    sum_jj = np.zeros(n); cnt_jj = np.zeros(n, dtype=int)
    jj = n - j

    got, trials = 0, 0
    while got < mc and trials < 3*mc:
        trials += 1
        try:
            S_idx = rng.choice(all_idx, size=j, replace=False)
            mask = np.zeros(n, bool); mask[S_idx] = True
            C_idx = np.where(~mask)[0]
            u = utility_func(x_train[S_idx], y_train[S_idx], x_valid, y_valid, clone(clf), final_model) \
              - utility_func(x_train[C_idx], y_train[C_idx], x_valid, y_valid, clone(clf), final_model)
            got += 1
            sum_j[S_idx]  +=  u; cnt_j[S_idx]  += 1
            if jj > 0 and C_idx.size>0:
                sum_jj[C_idx] += -u; cnt_jj[C_idx] += 1
        except Exception:
            continue
    return j, sum_j, cnt_j, (jj if jj>0 else None), (sum_jj if jj>0 else None), (cnt_jj if jj>0 else None)



def cc_shapley_parallel(x_train: np.ndarray, y_train: np.ndarray, 
                        x_valid: np.ndarray, y_valid: np.ndarray,
                        clf, final_model, utility_func,
                        num_MC: int = 100,
                        num_processes: Optional[int] = None,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Parallel CC Shapley using layer-level workers that ALREADY aggregate per player.

    Worker `_full_cc_layer_worker` returns, for each layer j:
      sum_j[n], cnt_j[n], and its dual layer jj=n-j (maybe None): sum_jj[n], cnt_jj[n].
    We just vector-accumulate them here and do per-layer means, then average across layers.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    if rng is None:
        rng = np.random.default_rng()

    n = x_train.shape[0]
    print(f"CC Layer-wise: {n} layers, {num_MC} MC/layer, {num_processes} processes")
    print(f"Total tasks: {n} layers (one per process)")

    # ---- 1) 生成任务（注意：传 n 给 worker；num_MC 改名 mc） --------------------
    layer_tasks = []
    for j in range(1, n + 1):
        layer_seed = (int(rng.integers(0, 2**31)) ^ (hash(j) & 0x7FFFFFFF)) & 0x7FFFFFFF
        task_args = (j, num_MC, layer_seed, x_train, y_train, x_valid, y_valid,
                     clf, final_model, utility_func, n)
        layer_tasks.append(task_args)

    # ---- 2) 并行执行（调用新的 worker） -----------------------------------------
    print("Starting parallel CC layer processing...")
    with mp.Pool(processes=num_processes) as pool:
        layer_results = list(tqdm(
            pool.imap_unordered(_full_cc_layer_worker, layer_tasks),
            total=len(layer_tasks),
            desc="CC layer processing"
        ))

    # ---- 3) 主进程向量化汇总：直接累加每层每玩家的 sum/cnt -----------------------
    # 列索引用层号 j（1..n）；0 列闲置
    cc_sum = np.zeros((n, n + 1), dtype=float)
    cc_cnt = np.zeros((n, n + 1), dtype=int)

    # layer_results 每项是：
    # (j, sum_j[n], cnt_j[n], jj or None, sum_jj[n] or None, cnt_jj[n] or None)
    for (j, sum_j, cnt_j, jj, sum_jj, cnt_jj) in layer_results:
        # 当前层 j
        cc_sum[:, j] += sum_j
        cc_cnt[:, j] += cnt_j
        # 对偶层 jj=n-j（可能为 None）
        if jj is not None and sum_jj is not None and cnt_jj is not None:
            cc_sum[:, jj] += sum_jj
            cc_cnt[:, jj] += cnt_jj

    # ---- 4) 分层条件平均，再跨层平均（跳过 j=0 列） -----------------------------
    cc_mean = np.full_like(cc_sum, np.nan, dtype=float)
    mask = cc_cnt > 0
    cc_mean[mask] = cc_sum[mask] / cc_cnt[mask]

    sv = np.nanmean(cc_mean[:, 1:], axis=1)  # 按层做等权平均
    print(f"CC Parallel completed. Shapley values computed for {n} data points.")
    return sv






def integral_cc_sparse_all_players(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    clf,
    final_model,
    utility_func,
    m_layers: int = 16,                 # 采样层数 K（含端点、去重后）
    mc_per_layer: int = 50,             # 每个被选层的样本数
    node_scheme: Literal["equispaced","chebyshev"] = "equispaced",
    aggregator:  Literal["voronoi","linear"] = "voronoi",  # 稳健推荐 "voronoi"；"linear" 为分段线性+trapz
    rng: Optional[np.random.Generator] = None,
    max_retry_factor: int = 3,          # 失败重试上限系数
    verbose: bool = False,
):
    """
    稀疏层 CC + 积分近似（一次性产出所有玩家 Shapley）：
      - 选 K 个层 j_k（强制包含 1 和 n，同时把对偶层 n−j_k 也纳入以复用补集）
      - 在每个选中层 j 做 mc_per_layer 次采样，计算 u=U(S)-U(N\\S)
        · i∈S → 列 j 加 +u
        · i∉S → 列 n−j 加 −u（若该层被选中）
      - 得到每个玩家在节点集合上的 b_j(i) 估计
      - 在 t 轴聚合：
        · "voronoi"：最近邻-沃罗诺伊权重（piecewise-constant）
        · "linear"：分段线性插值 + trapz
    """
    if rng is None:
        rng = np.random.default_rng()

    n = x_train.shape[0]
    all_idx = np.arange(n)

    # 1) 选层节点（含端点）
    if node_scheme == "equispaced":
        base = np.rint(np.linspace(1, n, m_layers)).astype(int)
    elif node_scheme == "chebyshev":
        # Chebyshev I 型节点在 [-1,1]，映射到 [1,n]
        k = np.linspace(0, m_layers - 1, m_layers)
        x = -np.cos(np.pi * k / (m_layers - 1))          # [-1,1]
        t = 0.5 * (x + 1.0)                              # [0,1]
        base = np.rint(1 + t * (n - 1)).astype(int)      # [1,n]
    else:
        raise ValueError("unknown node_scheme")

    j_nodes = np.unique(np.r_[1, base, n])               # 强制端点
    # 加入对偶层，便于补集样本落到我们记录的列
    nodes_set = set(j_nodes.tolist())
    for j in j_nodes:
        jj = n - j
        if 1 <= jj <= n:
            nodes_set.add(jj)
    nodes_sorted = np.array(sorted(nodes_set), dtype=int)  # 1..n 中的若干层
    M = int(nodes_sorted.size)

    if verbose:
        print(f"[nodes] picked {M} layers among 1..{n}: {nodes_sorted[:10]}{'...' if M>10 else ''}")

    # 2) 累加器：每个玩家×每个被选层的和/次数
    sum_mat = np.zeros((n, M), dtype=float)
    cnt_mat = np.zeros((n, M), dtype=int)
    pos = {int(j): k for k, j in enumerate(nodes_sorted)}

    # 3) 在“原始选层 j_nodes”上采样（补集会写到对偶层；若对偶层不在 nodes，则跳过）
    for j in j_nodes:
        target = mc_per_layer
        trials = 0
        got = 0
        while got < target and trials < max_retry_factor * target:
            trials += 1
            # 随机子集 S（|S|=j）
            S_idx = rng.choice(all_idx, size=j, replace=False)

            # 快速补集
            mask = np.zeros(n, dtype=bool)
            mask[S_idx] = True
            C_idx = np.where(~mask)[0]

            # 两次训练：U(S) 与 U(C)
            try:
                u_S = utility_func(x_train[S_idx], y_train[S_idx],
                                   x_valid, y_valid, clone(clf), final_model)
                u_C = utility_func(x_train[C_idx], y_train[C_idx],
                                   x_valid, y_valid, clone(clf), final_model)
            except Exception:
                continue  # 失败，重试

            u = float(u_S - u_C)
            got += 1

            # S 内玩家 → 层 j
            cj = pos.get(int(j), None)
            if cj is not None:
                sum_mat[S_idx, cj] += u
                cnt_mat[S_idx, cj] += 1

            # 补集玩家 → 层 n-j
            jj = n - j
            cjj = pos.get(int(jj), None)
            if cjj is not None and jj > 0 and C_idx.size > 0:
                sum_mat[C_idx, cjj] += -u
                cnt_mat[C_idx, cjj] += 1

        if verbose and got < target:
            print(f"[warn] layer j={j}: got {got}/{target} samples after {trials} trials")

    # 4) 计算节点上的 b_j(i) 估计
    with np.errstate(invalid='ignore', divide='ignore'):
        g_mat = sum_mat / cnt_mat   # (n, M)，缺失为 NaN

    # 5) 在 t 轴上做积分近似
    t_nodes = nodes_sorted.astype(float) / n  # 节点横坐标
    shapley = np.zeros(n, dtype=float)

    if aggregator == "voronoi":
        # 沃罗诺伊边界与权重
        bounds = np.empty(M + 1, dtype=float)
        bounds[0] = 0.0
        bounds[-1] = 1.0
        if M > 1:
            bounds[1:-1] = 0.5 * (t_nodes[:-1] + t_nodes[1:])
        w = np.diff(bounds)  # 每个节点的影响宽度

        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                shapley[i] = 0.0
                continue
            # 最近邻填补缺失（与沃罗诺伊权重一致）
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask],
                               left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.sum(w * g_fill))

    elif aggregator == "linear":
        # 分段线性插值 + trapz
        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                shapley[i] = 0.0
                continue
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask],
                               left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.trapz(g_fill, t_nodes))
    else:
        raise ValueError("unknown aggregator")

    return shapley, dict(j_nodes=np.array(sorted(set(j_nodes.tolist()+[n-j for j in j_nodes if 1<=n-j<=n]))),
                         nodes_sorted=nodes_sorted, t_nodes=t_nodes, g_mat=g_mat)



def _sparse_cc_layer_worker(args) -> Tuple[int, int, np.ndarray, np.ndarray, Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    处理单个选中层 j：做 mc 次采样，训练两次，聚合回传两列的 (sum, cnt)。
    返回: (j, col_j, sum_j(n,), cnt_j(n,), col_jj, sum_jj(n,), cnt_jj(n,))
    若对偶层 col_jj 不在节点集合，则对应返回 None。
    """
    (j, mc, seed, x_train, y_train, x_valid, y_valid,
     clf, final_model, utility_func, n, pos) = args

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    # 找到本层与对偶层在矩阵中的列号
    col_j  = pos.get(int(j),  None)
    jj     = n - j
    col_jj = pos.get(int(jj), None) if 1 <= jj <= n else None

    sum_j  = np.zeros(n, dtype=float) if col_j  is not None else None
    cnt_j  = np.zeros(n, dtype=int)   if col_j  is not None else None
    sum_jj = np.zeros(n, dtype=float) if col_jj is not None else None
    cnt_jj = np.zeros(n, dtype=int)   if col_jj is not None else None

    got, trials, max_trials = 0, 0, 3 * mc
    while got < mc and trials < max_trials:
        trials += 1
        try:
            S_idx = rng.choice(all_idx, size=j, replace=False)
            mask = np.zeros(n, dtype=bool); mask[S_idx] = True
            C_idx = np.where(~mask)[0]

            u_S = utility_func(x_train[S_idx], y_train[S_idx], x_valid, y_valid, clone(clf), final_model)
            u_C = utility_func(x_train[C_idx], y_train[C_idx], x_valid, y_valid, clone(clf), final_model)
            u   = float(u_S - u_C)
            got += 1

            if col_j is not None:
                sum_j[S_idx] += u
                cnt_j[S_idx] += 1
            if col_jj is not None and C_idx.size > 0 and jj > 0:
                sum_jj[C_idx] += (-u)
                cnt_jj[C_idx] += 1

        except Exception:
            continue  # 失败重试

    return (j, col_j, sum_j, cnt_j, col_jj, sum_jj, cnt_jj)


# ---------- 并行稀疏积分版：一次性输出所有玩家 ----------
def integral_cc_sparse_all_players_parallel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    clf,
    final_model,
    utility_func,
    m_layers: int = 16,                                # 选多少层（含端点，去重后）
    mc_per_layer: int = 50,                            # 每个被选层的 MC 次数
    node_scheme: Literal["equispaced","chebyshev"] = "equispaced",
    aggregator:  Literal["voronoi","linear"] = "linear",
    num_processes: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    base_seed: int = 12345,
    verbose: bool = False,
):
    """
    稀疏层 CC + 积分近似（并行）。一次训练两次，单层 worker 聚合两列并回传，主进程在 t 轴聚合。
    返回: shapley (n,), info 字典（含节点等中间量）
    """
    if rng is None:
        rng = np.random.default_rng(base_seed)
    if num_processes is None:
        num_processes = mp.cpu_count()

    n = x_train.shape[0]

    # 1) 选层节点：强制包含 1 与 n
    if node_scheme == "equispaced":
        base = np.rint(np.linspace(1, n, m_layers)).astype(int)
    elif node_scheme == "chebyshev":
        k = np.linspace(0, m_layers - 1, m_layers)
        x = -np.cos(np.pi * k / max(1, m_layers - 1))   # [-1,1]
        t = 0.5 * (x + 1.0)                             # [0,1]
        base = np.rint(1 + t * (n - 1)).astype(int)     # [1,n]
    else:
        raise ValueError("unknown node_scheme")

    j_nodes = np.unique(np.r_[1, base, n])

    # 并入对偶层，提升样本复用
    nodes_set = set(j_nodes.tolist())
    for j in j_nodes:
        jj = n - j
        if 1 <= jj <= n:
            nodes_set.add(jj)
    nodes_sorted = np.array(sorted(nodes_set), dtype=int)  # 1..n 中若干层
    M = int(nodes_sorted.size)
    pos = {int(j): k for k, j in enumerate(nodes_sorted)}  # 层号→列号

    if verbose:
        print(f"[nodes] picked {M} layers among 1..{n}")

    # 2) 并行：每个原始选层 j_nodes 启动一个任务
    tasks = []
    for j in j_nodes:
        seed = int(base_seed + j)   # 稳定分层种子
        tasks.append((int(j), mc_per_layer, seed,
                      x_train, y_train, x_valid, y_valid,
                      clf, final_model, utility_func,
                      n, pos))

    sum_mat = np.zeros((n, M), dtype=float)
    cnt_mat = np.zeros((n, M), dtype=int)

    if verbose:
        print(f"[parallel] {len(tasks)} layers, {mc_per_layer} MC/layer, procs={num_processes}")

    with mp.Pool(processes=num_processes) as pool:
        for (j, col_j, sum_j, cnt_j, col_jj, sum_jj, cnt_jj) in pool.imap_unordered(_sparse_cc_layer_worker, tasks):
            if col_j is not None:
                sum_mat[:, col_j] += sum_j
                cnt_mat[:, col_j] += cnt_j
            if col_jj is not None:
                sum_mat[:, col_jj] += sum_jj
                cnt_mat[:, col_jj] += cnt_jj

    # 3) 节点上的 b_j(i) 估计
    with np.errstate(invalid='ignore', divide='ignore'):
        g_mat = sum_mat / cnt_mat  # (n, M)，缺失处 NaN

    # 4) t 轴聚合
    t_nodes = nodes_sorted.astype(float) / n
    shapley = np.zeros(n, dtype=float)

    if aggregator == "voronoi":
        bounds = np.empty(M + 1, dtype=float)
        bounds[0] = 0.0
        bounds[-1] = 1.0
        if M > 1:
            bounds[1:-1] = 0.5 * (t_nodes[:-1] + t_nodes[1:])
        w = np.diff(bounds)  # 每个节点的权重

        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                shapley[i] = 0.0
                continue
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask],
                               left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.sum(w * g_fill))

    elif aggregator == "linear":
        for i in range(n):
            g = g_mat[i, :]
            mask = np.isfinite(g)
            if not np.any(mask):
                shapley[i] = 0.0
                continue
            g_fill = np.interp(t_nodes, t_nodes[mask], g[mask],
                               left=g[mask][0], right=g[mask][-1])
            shapley[i] = float(np.trapz(g_fill, t_nodes))
    else:
        raise ValueError("unknown aggregator")

    info = dict(
        nodes_sorted=nodes_sorted,
        t_nodes=t_nodes,
        g_mat=g_mat,
        sum_mat=sum_mat,
        cnt_mat=cnt_mat,
        j_nodes=j_nodes
    )
    return shapley, info





if __name__ == "__main__":
    from utils.data_utils import load_dataset
    from utils.utilities import utility_acc

    X_train, X_test, y_train, y_test = load_dataset('cancer')
    print("训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)


    clf1 = return_model('logistic')
    final_model = clf1.fit(X_train, y_train)

    sv = cc_shapley_parallel(
        x_train=X_train,
        y_train=y_train,
        x_valid=X_test,
        y_valid=y_test,
        clf=clf1,  # 替换为实际分类器
        final_model=final_model,  # 替换为实际最终模型
        utility_func=utility_acc,  # 替换为实际效用函数
        num_MC=100000,  # 减少 MC 次数以加快测试速度
        num_processes=15 # 使用较少的进程以节省资源
    )

    # save sv to a file
    np.save("lr_cancer_shapley_value_acc_groundtruth", sv)
