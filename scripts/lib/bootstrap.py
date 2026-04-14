"""Bootstrap confidence intervals for means of binary (or bounded) sequences."""
import random
from typing import List, Sequence, Tuple


def bootstrap_ci(binary_values: Sequence[int], seed: int, iterations: int = 2000) -> Tuple[float, float]:
    if not binary_values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(binary_values)
    means: List[float] = []
    for _ in range(iterations):
        sample = [binary_values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations)]
    return lo, hi


def bootstrap_mean_ci(values: Sequence[float], seed: int, iterations: int = 2000) -> Tuple[float, float]:
    """Percentile CI for the sample mean under bootstrap resampling of indices."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    stats: List[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo = stats[int(0.025 * iterations)]
    hi = stats[int(0.975 * iterations)]
    return lo, hi
