from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    name: str
    expression: str
    player_type: str
    min_pa: int | None
    min_ip: float | None
    created_at: str


@dataclass(frozen=True, slots=True)
class CandidateValue:
    player_id: int
    season: int
    value: float | None


@dataclass(frozen=True, slots=True)
class BinnedValue:
    player_id: int
    season: int
    bin_label: str
    value: float


@dataclass(frozen=True, slots=True)
class BinTargetMean:
    bin_label: str
    target: str
    mean: float
    count: int
