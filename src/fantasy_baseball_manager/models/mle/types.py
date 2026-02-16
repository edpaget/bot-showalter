from dataclasses import dataclass, field


@dataclass(frozen=True)
class MLEConfig:
    babip_regression_weight: float = 0.5
    k_experience_factor: float = 1.15
    min_pa: int = 100
    discount_factor: float = 0.55
    babip_stabilization_bip: float = 820.0
    season_weights: tuple[float, ...] = (5.0, 4.0, 3.0)
    regression_pa: float = 1200.0


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
    k_dampening: float = 0.5


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

    @property
    def avg(self) -> float:
        return self.h / self.ab if self.ab > 0 else 0.0

    @property
    def obp(self) -> float:
        denom = self.ab + self.bb + self.hbp + self.sf
        return (self.h + self.bb + self.hbp) / denom if denom > 0 else 0.0

    @property
    def slg(self) -> float:
        tb = self.h + self.doubles + self.triples * 2 + self.hr * 3
        return tb / self.ab if self.ab > 0 else 0.0

    @property
    def iso(self) -> float:
        return self.slg - self.avg

    @property
    def k_pct(self) -> float:
        return self.so / self.pa if self.pa > 0 else 0.0

    @property
    def bb_pct(self) -> float:
        return self.bb / self.pa if self.pa > 0 else 0.0

    @property
    def babip(self) -> float:
        bip = self.ab - self.so - self.hr + self.sf
        return (self.h - self.hr) / bip if bip > 0 else 0.0


DEFAULT_STABILIZATION_PA: dict[str, float] = {
    "k_pct": 60.0,
    "bb_pct": 120.0,
    "iso": 160.0,
    "babip": 820.0,
}


@dataclass(frozen=True)
class BlendConfig:
    discount_factor: float = 0.55
    stabilization_pa: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STABILIZATION_PA))


@dataclass(frozen=True)
class BlendedStatLine:
    player_id: int
    season: int
    mlb_pa: int
    mle_pa: int
    effective_pa: float
    rates: dict[str, float]
