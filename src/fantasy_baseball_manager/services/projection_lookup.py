import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerProjection, PTSourceSummary, SystemSummary
from fantasy_baseball_manager.name_utils import resolve_players

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import PlayerRepo, ProjectionRepo
logger = logging.getLogger(__name__)


class ProjectionLookupService:
    def __init__(self, player_repo: PlayerRepo, projection_repo: ProjectionRepo) -> None:
        self._player_repo = player_repo
        self._projection_repo = projection_repo

    def lookup(self, player_name: str, season: int, system: str | None = None) -> list[PlayerProjection]:
        logger.debug("Projection lookup: player=%s season=%d system=%s", player_name, season, system)
        players = resolve_players(self._player_repo, player_name)

        results: list[PlayerProjection] = []
        for player in players:
            assert player.id is not None  # noqa: S101 - type narrowing
            projections = self._projection_repo.get_by_player_season(player.id, season, system)
            for proj in projections:
                results.append(
                    PlayerProjection(
                        player_name=f"{player.name_first} {player.name_last}",
                        system=proj.system,
                        version=proj.version,
                        source_type=proj.source_type,
                        player_type=proj.player_type,
                        stats=proj.stat_json,
                    )
                )

        logger.debug("Projection lookup returned %d results", len(results))
        return results

    def list_systems(self, season: int) -> list[SystemSummary]:
        projections = self._projection_repo.get_by_season(season)

        groups: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"batter": 0, "pitcher": 0})
        for proj in projections:
            key = (proj.system, proj.version, proj.source_type)
            if proj.player_type in ("batter", "pitcher"):
                groups[key][proj.player_type] += 1

        summaries = [
            SystemSummary(
                system=system,
                version=version,
                source_type=source_type,
                batter_count=counts["batter"],
                pitcher_count=counts["pitcher"],
            )
            for (system, version, source_type), counts in sorted(groups.items())
        ]
        return summaries

    def list_pt_sources(self, season: int) -> list[PTSourceSummary]:
        projections = self._projection_repo.get_by_season(season)

        groups: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"batter": 0, "pitcher": 0})
        for proj in projections:
            pa = proj.stat_json.get("pa", 0)
            ip = proj.stat_json.get("ip", 0)
            if pa > 0:
                groups[(proj.system, proj.version)]["batter"] += 1
            if ip > 0:
                groups[(proj.system, proj.version)]["pitcher"] += 1

        return [
            PTSourceSummary(
                system=system,
                version=version,
                batter_count=counts["batter"],
                pitcher_count=counts["pitcher"],
            )
            for (system, version), counts in sorted(groups.items())
            if counts["batter"] > 0 or counts["pitcher"] > 0
        ]
