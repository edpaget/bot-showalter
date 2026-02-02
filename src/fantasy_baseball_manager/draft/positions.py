from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.marcel.models import PitchingProjection

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
