from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YahooDraftPick:
    player_id: str
    team_key: str
    round: int
    pick: int


class DraftStatus(Enum):
    PRE_DRAFT = "predraft"
    IN_PROGRESS = "draft"
    POST_DRAFT = "postdraft"


_DRAFT_STATUS_MAP: dict[str, DraftStatus] = {s.value: s for s in DraftStatus}


class DraftResultsSource(Protocol):
    def fetch_draft_results(self) -> list[YahooDraftPick]: ...

    def fetch_draft_status(self) -> DraftStatus: ...

    def fetch_user_team_key(self) -> str: ...


class YahooDraftResultsSource:
    def __init__(self, league: object) -> None:
        self._league = league

    def fetch_draft_results(self) -> list[YahooDraftPick]:
        raw_picks: list[dict[str, object]] = self._league.draft_results()  # type: ignore[union-attr]
        picks: list[YahooDraftPick] = []
        for raw in raw_picks:
            picks.append(
                YahooDraftPick(
                    player_id=str(raw["player_id"]),
                    team_key=str(raw["team_key"]),
                    round=int(raw["round"]),  # type: ignore[arg-type]
                    pick=int(raw["pick"]),  # type: ignore[arg-type]
                )
            )
        logger.debug("Fetched %d draft picks from Yahoo", len(picks))
        return picks

    def fetch_draft_status(self) -> DraftStatus:
        settings: dict[str, object] = self._league.settings()  # type: ignore[union-attr]
        raw_status = str(settings["draft_status"])
        status = _DRAFT_STATUS_MAP.get(raw_status, DraftStatus.PRE_DRAFT)
        logger.debug("Draft status: %s", status)
        return status

    def fetch_user_team_key(self) -> str:
        key: str = self._league.team_key()  # type: ignore[union-attr]
        logger.debug("User team key: %s", key)
        return key
