import numpy as np


def get_airport_costs():
    """
    Canonical 100-player airport game requirements.
    Counts (in order of runway requirement):
    1 repeated 10 times, 2 repeated 8, 3 repeated 12, 4 repeated 6,
    5 repeated 14, 6 repeated 8, 7 repeated 9, 8 repeated 10,
    9 repeated 10, 10 repeated 13.
    """
    counts = {
        1: 100,
        2: 80,
        3: 120,
        4: 60,
        5: 140,
        6: 80,
        7: 90,
        8: 100,
        9: 100,
        10: 130,
    }
    costs = []
    for value in range(1, 11):
        costs.extend([value] * counts[value])
    return np.array(costs, dtype=float)


def compute_exact_airport_shapley(costs=None):
    """
    Closed-form Shapley values for airport game (max coalition utility).
    """
    if costs is None:
        costs = get_airport_costs()
    costs = np.asarray(costs, dtype=float)
    n = len(costs)

    order = np.argsort(costs)
    sorted_costs = costs[order]

    deltas = np.empty(n, dtype=float)
    deltas[0] = sorted_costs[0]
    deltas[1:] = np.diff(sorted_costs)

    denom = n - np.arange(n)
    term = deltas / denom
    shapley_sorted = np.cumsum(term)

    shapley = np.zeros(n, dtype=float)
    shapley[order] = shapley_sorted
    return shapley
