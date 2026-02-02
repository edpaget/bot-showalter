from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot

if TYPE_CHECKING:
    from pathlib import Path

    import yahoo_fantasy_api

    from fantasy_baseball_manager.marcel.models import PitchingProjection
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

_POSITION_NORMALIZATIONS: dict[str, str] = {
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "DH": "Util",
}

DEFAULT_ROSTER_CONFIG: RosterConfig = RosterConfig(
    slots=(
        RosterSlot(position="C", count=2),
        RosterSlot(position="1B", count=1),
        RosterSlot(position="2B", count=1),
        RosterSlot(position="SS", count=1),
        RosterSlot(position="3B", count=1),
        RosterSlot(position="OF", count=5),
        RosterSlot(position="Util", count=1),
        RosterSlot(position="SP", count=9),
        RosterSlot(position="RP", count=4),
        RosterSlot(position="BN", count=3),
    )
)

SP_GS_RATIO_THRESHOLD: float = 0.5


def normalize_position(pos: str) -> str:
    return _POSITION_NORMALIZATIONS.get(pos.strip(), pos.strip())


def infer_pitcher_role(proj: PitchingProjection) -> str:
    if proj.g == 0:
        return "SP"
    return "SP" if proj.gs / proj.g >= SP_GS_RATIO_THRESHOLD else "RP"


class PositionSource(Protocol):
    def fetch_positions(self) -> dict[str, tuple[str, ...]]: ...


_NON_FIELD_POSITIONS: frozenset[str] = frozenset({"BN", "IL", "DL", "IL+", "NA"})


def load_positions_file(path: Path) -> dict[str, tuple[str, ...]]:
    positions: dict[str, tuple[str, ...]] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            player_id = row[0].strip()
            raw_positions = row[1].strip().split("/") if len(row) > 1 else []
            normalized = tuple(dict.fromkeys(normalize_position(p) for p in raw_positions))
            positions[player_id] = normalized
    return positions


class CsvPositionSource:
    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        return load_positions_file(self._path)


class YahooPositionSource:
    def __init__(self, league: yahoo_fantasy_api.League, id_mapper: PlayerIdMapper) -> None:
        self._league = league
        self._id_mapper = id_mapper

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        players: list[dict[str, object]] = list(self._league.taken_players())
        players.extend(self._league.free_agents("B"))
        players.extend(self._league.free_agents("P"))

        seen: set[str] = set()
        positions: dict[str, tuple[str, ...]] = {}

        for player in players:
            yahoo_id = str(player["player_id"])
            if yahoo_id in seen:
                continue
            seen.add(yahoo_id)

            fg_id = self._id_mapper.yahoo_to_fangraphs(yahoo_id)
            if fg_id is None:
                continue

            raw_positions = [
                str(p)
                for p in player.get("eligible_positions", ())  # type: ignore[union-attr]
                if str(p) not in _NON_FIELD_POSITIONS
            ]
            normalized = tuple(dict.fromkeys(normalize_position(p) for p in raw_positions))
            positions[fg_id] = normalized

        return positions
