from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING, Protocol, cast

from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import yahoo_fantasy_api

    from fantasy_baseball_manager.marcel.models import PitchingProjection
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper

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
    """Maps FanGraphs player IDs to their eligible positions.

    Not migrated to ``DataSource[T]`` because the return type is a mapping
    (``dict[str, tuple[str, ...]]``), not a ``list[T]``.  Two lightweight
    implementations exist (``YahooPositionSource``, ``CsvPositionSource``) and
    the only consumer is ``draft/cli.py``.  Already cached via ``cached_call()``.
    """

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
    def __init__(self, league: yahoo_fantasy_api.League, id_mapper: SfbbMapper) -> None:
        self._league = league
        self._id_mapper = id_mapper

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        players: list[dict[str, object]] = list(self._league.taken_players())
        players.extend(self._league.free_agents("B"))
        players.extend(self._league.free_agents("P"))

        seen: set[str] = set()
        positions: dict[str, tuple[str, ...]] = {}

        unmapped_count = 0
        mapped_count = 0

        for player in players:
            yahoo_id = str(player["player_id"])
            if yahoo_id in seen:
                continue
            seen.add(yahoo_id)

            fg_id = self._id_mapper.yahoo_to_fangraphs(yahoo_id)
            if fg_id is None:
                unmapped_count += 1
                logger.debug(
                    "No FanGraphs ID for %s (yahoo_id=%s)",
                    player.get("name", "?"),
                    yahoo_id,
                )
                continue

            mapped_count += 1
            eligible = cast("Iterable[object]", player.get("eligible_positions", ()))
            raw_positions = [str(p) for p in eligible if str(p) not in _NON_FIELD_POSITIONS]
            normalized = tuple(dict.fromkeys(normalize_position(p) for p in raw_positions))
            positions[fg_id] = normalized

        logger.debug(
            "Yahoo positions: %d mapped, %d unmapped, %d total positions",
            mapped_count,
            unmapped_count,
            len(positions),
        )
        return positions
