from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import LeagueKeeper
from fantasy_baseball_manager.ingest.adp_mapper import _build_player_lookups, _normalize_name

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player
    from fantasy_baseball_manager.repos import LeagueKeeperRepo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeagueKeeperImportResult:
    loaded: int
    skipped: int
    unmatched: list[str]


def import_league_keepers(
    rows: list[dict[str, Any]],
    repo: LeagueKeeperRepo,
    players: list[Player],
    season: int,
    league: str,
) -> LeagueKeeperImportResult:
    _, by_name = _build_player_lookups(players)

    loaded = 0
    skipped = 0
    unmatched: list[str] = []

    for row in rows:
        raw_name = row.get("player_name") or row.get("Player") or row.get("Name") or ""
        if not raw_name.strip():
            skipped += 1
            continue

        team_name = row.get("team_name") or row.get("Team") or ""
        if not team_name.strip():
            skipped += 1
            continue

        normalized = _normalize_name(raw_name)
        candidates = by_name.get(normalized, [])

        if len(candidates) != 1:
            if len(candidates) > 1:
                logger.debug("Ambiguous player name '%s' (%d matches), skipping", raw_name, len(candidates))
            else:
                logger.debug("No player match for '%s'", raw_name)
            unmatched.append(raw_name)
            continue

        player_id = candidates[0]

        cost_str = str(row.get("cost") or row.get("Cost") or "").strip().lstrip("$")
        cost: float | None = float(cost_str) if cost_str else None

        source = str(row.get("source") or row.get("Source") or "").strip() or None

        repo.upsert_batch(
            [
                LeagueKeeper(
                    player_id=player_id,
                    season=season,
                    league=league,
                    team_name=team_name.strip(),
                    cost=cost,
                    source=source,
                )
            ]
        )
        loaded += 1

    return LeagueKeeperImportResult(loaded=loaded, skipped=skipped, unmatched=unmatched)
