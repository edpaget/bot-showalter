from dataclasses import dataclass


@dataclass(frozen=True)
class MLEConfig:
    babip_regression_weight: float = 0.5
    k_experience_factor: float = 1.15
    min_pa: int = 100
    discount_factor: float = 0.55
    babip_stabilization_bip: float = 820.0


DEFAULT_AGE_BENCHMARKS: dict[str, float] = {
    "ROK": 18.0,
    "A": 19.0,
    "A+": 20.0,
    "AA": 21.0,
    "AAA": 22.0,
}


@dataclass(frozen=True)
class AgeAdjustmentConfig:
    benchmarks: dict[str, float]
    young_bonus_per_year: float = 0.025
    old_penalty_per_year: float = 0.010
    peak_age: float = 27.0
    development_rate_per_year: float = 0.006
    min_multiplier: float = 0.85
    max_multiplier: float = 1.25


@dataclass(frozen=True)
class TranslatedBattingLine:
    player_id: int
    season: int
    source_level: str
    pa: int
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    bb: int
    so: int
    hbp: int
    sf: int
    avg: float
    obp: float
    slg: float
    k_pct: float
    bb_pct: float
    iso: float
    babip: float
