import numpy as np
import math

# Synthetic voting weights with 500 players (moderate weights)
_WEIGHT_GROUPS = [
    (10, 20.0),
    (20, 18.0),
    (30, 16.0),
    (40, 14.0),
    (50, 12.0),
    (70, 10.0),
    (80, 8.0),
    (100, 6.0),
    (100, 4.0)
]

VOTING_WEIGHTS = np.concatenate([
    np.full(count, weight, dtype=float) for count, weight in _WEIGHT_GROUPS
])


def get_voting_weights():
    """Return a copy of the canonical voting weights."""
    return VOTING_WEIGHTS.copy()


def compute_exact_voting_shapley(weights: np.ndarray | None = None) -> np.ndarray:
    """
    Compute exact Shapley values for the weighted voting game via dynamic programming.
    """
    if weights is None:
        weights = VOTING_WEIGHTS
    weights = np.asarray(weights, dtype=int)
    n = len(weights)
    total_weight = np.sum(weights)
    quota = total_weight / 2.0  # coalition must exceed half of total weight
    max_weight = int(math.floor(quota))

    fact = [math.factorial(i) for i in range(n + 1)]
    coeff = [fact[k] * fact[n - 1 - k] / fact[n] for k in range(n)]

    shapley = np.zeros(n, dtype=float)

    for idx in range(n):
        dp = np.zeros((n, max_weight + 1), dtype=object)
        for row in range(n):
            for col in range(max_weight + 1):
                dp[row, col] = 0
        dp[0, 0] = 1

        for j, w in enumerate(weights):
            if j == idx:
                continue
            for size in range(n - 2, -1, -1):
                row = dp[size]
                next_row = dp[size + 1]
                for total in range(max_weight, w - 1, -1):
                    if row[total - w] > 0:
                        next_row[total] = next_row[total] + row[total - w]

        w_i = weights[idx]
        for size in range(n):
            if coeff[size] == 0:
                continue
            row = dp[size]
            if not np.any(row):
                continue
            for total in range(max_weight + 1):
                count = row[total]
                if count == 0:
                    continue
                if total <= quota < total + w_i:
                    shapley[idx] += coeff[size] * count

    # Normalize to ensure numerical stability (should sum to 1)
    shapley_sum = np.sum(shapley)
    if shapley_sum > 0:
        shapley /= shapley_sum
    return shapley
