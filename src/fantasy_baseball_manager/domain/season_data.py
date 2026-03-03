from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.roster import Roster
    from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick


@dataclass(frozen=True)
class SeasonData:
    league_key: str
    season: int
    draft_picks: list[YahooDraftPick] = field(default_factory=list)
    rosters: list[Roster] = field(default_factory=list)
