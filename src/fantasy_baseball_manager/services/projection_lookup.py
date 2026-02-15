from collections import defaultdict

from fantasy_baseball_manager.domain.projection import PlayerProjection, SystemSummary
from fantasy_baseball_manager.repos.protocols import PlayerRepo, ProjectionRepo


class ProjectionLookupService:
    def __init__(self, player_repo: PlayerRepo, projection_repo: ProjectionRepo) -> None:
        self._player_repo = player_repo
        self._projection_repo = projection_repo

    def lookup(self, player_name: str, season: int, system: str | None = None) -> list[PlayerProjection]:
        if "," in player_name:
            last, _, first = player_name.partition(",")
            last = last.strip()
            first = first.strip()
        else:
            last = player_name.strip()
            first = None

        players = self._player_repo.get_by_last_name(last)

        if first:
            players = [p for p in players if p.name_first and p.name_first.lower() == first.lower()]

        results: list[PlayerProjection] = []
        for player in players:
            assert player.id is not None
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
