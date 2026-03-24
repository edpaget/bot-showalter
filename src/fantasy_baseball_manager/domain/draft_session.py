from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


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
    keeper_player_ids: list[list[object]] | None = None  # [[player_id, player_type], ...]
    keeper_snapshot: list[dict[str, object]] | None = None
    team_names: dict[int, str] | None = None
    draft_order: list[int] | None = None
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
    player_type: PlayerType | None = None
    price: int | None = None
    id: int | None = None
