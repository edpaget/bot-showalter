from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime


@dataclass(frozen=True)
class Transaction:
    transaction_key: str
    league_key: str
    type: str
    timestamp: datetime.datetime
    status: str
    trader_team_key: str
    tradee_team_key: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class TransactionPlayer:
    transaction_key: str
    player_id: int | None
    yahoo_player_key: str
    player_name: str
    source_team_key: str | None
    dest_team_key: str | None
    type: str
    id: int | None = None
