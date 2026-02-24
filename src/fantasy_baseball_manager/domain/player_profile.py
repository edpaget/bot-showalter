from __future__ import annotations

import datetime
from dataclasses import dataclass


def compute_age(birth_date: str | None, season: int) -> int | None:
    """Compute age as of July 1 of the given season."""
    if birth_date is None:
        return None
    born = datetime.date.fromisoformat(birth_date)
    july_1 = datetime.date(season, 7, 1)
    return july_1.year - born.year - ((july_1.month, july_1.day) < (born.month, born.day))


@dataclass(frozen=True)
class PlayerProfile:
    player_id: int
    name: str
    age: int | None
    bats: str | None
    throws: str | None
    positions: tuple[str, ...] = ()
    pitcher_type: str | None = None
