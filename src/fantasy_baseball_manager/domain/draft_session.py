from dataclasses import dataclass


@dataclass(frozen=True)
class DraftSessionRecord:
    league: str
    season: int
    teams: int
    format: str
    user_team: int
    roster_slots: dict[str, int]  # stored as JSON in DB
    budget: int
    status: str  # "in_progress" | "complete" | "abandoned"
    created_at: str
    updated_at: str
    system: str = "zar"
    version: str = "1.0"
    keeper_player_ids: list[int] | None = None
    keeper_snapshot: list[dict[str, object]] | None = None
    id: int | None = None


@dataclass(frozen=True)
class DraftSessionTrade:
    session_id: int
    trade_number: int
    team_a: int
    team_b: int
    team_a_gives: list[int]
    team_b_gives: list[int]
    id: int | None = None


@dataclass(frozen=True)
class DraftSessionPick:
    session_id: int
    pick_number: int
    team: int
    player_id: int
    player_name: str
    position: str
    price: int | None = None
    id: int | None = None
