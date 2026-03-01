import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import KeeperCost
from fantasy_baseball_manager.ingest.adp_mapper import _build_player_lookups, _normalize_name

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player
    from fantasy_baseball_manager.repos import KeeperCostRepo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeeperImportResult:
    loaded: int
    skipped: int
    unmatched: list[str]


def import_keeper_costs(
    rows: list[dict[str, Any]],
    repo: KeeperCostRepo,
    players: list[Player],
    season: int,
    league: str,
    default_source: str = "auction",
) -> KeeperImportResult:
    _, by_name = _build_player_lookups(players)

    loaded = 0
    skipped = 0
    unmatched: list[str] = []

    for row in rows:
        raw_name = row.get("Player") or row.get("Name") or ""
        if not raw_name.strip():
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

        cost_str = str(row.get("Cost", "")).strip().lstrip("$")
        if not cost_str:
            skipped += 1
            continue
        cost = float(cost_str)

        years_str = str(row.get("Years", "1")).strip()
        years_remaining = int(years_str) if years_str else 1

        source = str(row.get("Source", default_source)).strip() or default_source

        repo.upsert_batch(
            [
                KeeperCost(
                    player_id=player_id,
                    season=season,
                    league=league,
                    cost=cost,
                    years_remaining=years_remaining,
                    source=source,
                )
            ]
        )
        loaded += 1

    return KeeperImportResult(loaded=loaded, skipped=skipped, unmatched=unmatched)
