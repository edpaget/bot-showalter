from dataclasses import dataclass


@dataclass(frozen=True)
class KeeperSeasonEntry:
    season: int
    cost: float
    source: str


@dataclass(frozen=True)
class KeeperHistory:
    player_id: int
    player_name: str
    league: str
    entries: tuple[KeeperSeasonEntry, ...]
