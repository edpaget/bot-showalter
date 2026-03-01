from dataclasses import dataclass


@dataclass(frozen=True)
class TeamCategoryProjection:
    category: str
    projected_value: float
    league_rank_estimate: int
    strength: str


@dataclass(frozen=True)
class RosterAnalysis:
    projections: list[TeamCategoryProjection]
    strongest_categories: list[str]
    weakest_categories: list[str]
