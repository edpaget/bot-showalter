import math
from collections.abc import Callable

WeightTransform = Callable[[list[float]], list[float]]


def raw(weights: list[float]) -> list[float]:
    return weights


def sqrt(weights: list[float]) -> list[float]:
    return [math.sqrt(w) for w in weights]


def log1p(weights: list[float]) -> list[float]:
    return [math.log1p(w) for w in weights]


def clamp(lo: float, hi: float) -> WeightTransform:
    def _clamp(weights: list[float]) -> list[float]:
        return [max(lo, min(hi, w)) for w in weights]

    return _clamp


REGISTRY: dict[str, WeightTransform] = {
    "raw": raw,
    "sqrt": sqrt,
    "log1p": log1p,
    "clamp_50_200": clamp(50, 200),
    "clamp_100_400": clamp(100, 400),
}


def get_transform(name: str) -> WeightTransform:
    if name not in REGISTRY:
        msg = f"Unknown sample weight transform: {name!r}. Available: {sorted(REGISTRY)}"
        raise KeyError(msg)
    return REGISTRY[name]
