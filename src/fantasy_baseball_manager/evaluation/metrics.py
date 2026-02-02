import math


def _validate_inputs(projected: list[float], actual: list[float]) -> None:
    if len(projected) != len(actual):
        raise ValueError(f"Input lengths must match: got {len(projected)} and {len(actual)}")
    if len(projected) == 0:
        raise ValueError("Inputs must not be empty")


def rmse(projected: list[float], actual: list[float]) -> float:
    _validate_inputs(projected, actual)
    mse = sum((p - a) ** 2 for p, a in zip(projected, actual, strict=True)) / len(projected)
    return math.sqrt(mse)


def mae(projected: list[float], actual: list[float]) -> float:
    _validate_inputs(projected, actual)
    return sum(abs(p - a) for p, a in zip(projected, actual, strict=True)) / len(projected)


def pearson_r(x: list[float], y: list[float]) -> float:
    _validate_inputs(x, y)
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    numerator = sum(dxi * dyi for dxi, dyi in zip(dx, dy, strict=True))
    std_x = math.sqrt(sum(dxi**2 for dxi in dx))
    std_y = math.sqrt(sum(dyi**2 for dyi in dy))
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return numerator / (std_x * std_y)


def _rank(values: list[float]) -> list[float]:
    """Rank values with average tie-breaking (1-based)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    _validate_inputs(x, y)
    return pearson_r(_rank(x), _rank(y))


def top_n_precision(projected_ids: list[str], actual_ids: list[str], n: int) -> float:
    if n <= 0:
        return 0.0
    projected_top = set(projected_ids[:n])
    actual_top = set(actual_ids[:n])
    if not projected_top:
        return 0.0
    overlap = projected_top & actual_top
    return len(overlap) / len(projected_top)
