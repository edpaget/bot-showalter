from dataclasses import dataclass


@dataclass(frozen=True)
class MLEConfig:
    babip_regression_weight: float = 0.5
    k_experience_factor: float = 1.15
    min_pa: int = 100
    discount_factor: float = 0.55
    babip_stabilization_bip: float = 820.0


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
