from dataclasses import dataclass


@dataclass(frozen=True)
class TierAssignment:
    player_id: int
    tier: int


@dataclass(frozen=True)
class DraftBoardRow:
    player_id: int
    player_name: str
    rank: int
    player_type: str
    position: str
    value: float
    category_z_scores: dict[str, float]
    tier: int | None = None
    adp_overall: float | None = None
    adp_rank: int | None = None
    adp_delta: int | None = None


@dataclass(frozen=True)
class DraftBoard:
    rows: list[DraftBoardRow]
    batting_categories: tuple[str, ...]
    pitching_categories: tuple[str, ...]
