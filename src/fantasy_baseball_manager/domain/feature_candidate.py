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
