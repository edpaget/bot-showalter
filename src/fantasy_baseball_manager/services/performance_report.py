from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    PitchingStatsRepo,
    PlayerRepo,
    ProjectionRepo,
)

_INVERTED_STATS: frozenset[str] = frozenset({"era", "fip", "whip", "bb_per_9", "hr_per_9"})


def _get_batter_actual(actual: BattingStats, stat_name: str) -> float | None:
    if stat_name == "iso":
        if actual.slg is not None and actual.avg is not None:
            return actual.slg - actual.avg
        return None
    if stat_name == "babip":
        if (
            actual.h is not None
            and actual.hr is not None
            and actual.ab is not None
            and actual.so is not None
            and actual.sf is not None
        ):
            denom = actual.ab - actual.so - actual.hr + actual.sf
            if denom != 0:
                return (actual.h - actual.hr) / denom
        return None
    val = getattr(actual, stat_name, None)
    if isinstance(val, int | float):
        return float(val)
    return None


def _get_pitcher_actual(actual: PitchingStats, stat_name: str) -> float | None:
    if stat_name == "hr_per_9":
        if actual.hr is not None and actual.ip is not None and actual.ip != 0:
            return actual.hr * 9 / actual.ip
        return None
    if stat_name == "babip":
        if actual.h is not None and actual.hr is not None and actual.ip is not None and actual.so is not None:
            denom = actual.ip * 3 + actual.h - actual.so - actual.hr
            if denom != 0:
                return (actual.h - actual.hr) / denom
        return None
    val = getattr(actual, stat_name, None)
    if isinstance(val, int | float):
        return float(val)
    return None


class PerformanceReportService:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        player_repo: PlayerRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
    ) -> None:
        self._projection_repo = projection_repo
        self._player_repo = player_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo

    def compute_deltas(
        self,
        system: str,
        version: str,
        season: int,
        player_type: str,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        min_pa: int | None = None,
    ) -> list[PlayerStatDelta]:
        projections = self._projection_repo.get_by_system_version(system, version)
        projections = [p for p in projections if p.season == season and p.player_type == player_type]

        proj_by_player: dict[int, Projection] = {}
        for proj in projections:
            proj_by_player[proj.player_id] = proj

        players = self._player_repo.all()
        player_by_id: dict[int, Player] = {p.id: p for p in players if p.id is not None}

        target_stats: tuple[str, ...] | list[str]
        if stats is not None:
            target_stats = stats
        elif player_type == "pitcher":
            target_stats = PITCHER_TARGETS
        else:
            target_stats = BATTER_TARGETS

        if player_type == "pitcher":
            actuals_list = self._pitching_repo.get_by_season(season, source=actuals_source)
            actuals_by_player: dict[int, BattingStats | PitchingStats] = {a.player_id: a for a in actuals_list}
        else:
            bat_actuals_list = self._batting_repo.get_by_season(season, source=actuals_source)
            actuals_by_player = {a.player_id: a for a in bat_actuals_list}

        if min_pa is not None:
            if player_type == "pitcher":
                actuals_by_player = {
                    pid: a
                    for pid, a in actuals_by_player.items()
                    if isinstance(a, PitchingStats) and a.ip is not None and a.ip >= min_pa
                }
            else:
                actuals_by_player = {
                    pid: a
                    for pid, a in actuals_by_player.items()
                    if isinstance(a, BattingStats) and a.pa is not None and a.pa >= min_pa
                }

        raw_deltas: dict[str, list[tuple[int, str, float, float, float]]] = {}

        for player_id, proj in proj_by_player.items():
            actual = actuals_by_player.get(player_id)
            if actual is None:
                continue

            player = player_by_id.get(player_id)
            player_name = f"{player.name_first} {player.name_last}" if player else str(player_id)

            for stat_name in target_stats:
                expected_val = proj.stat_json.get(stat_name)
                if expected_val is None:
                    continue

                if player_type == "pitcher":
                    assert isinstance(actual, PitchingStats)
                    actual_val = _get_pitcher_actual(actual, stat_name)
                else:
                    assert isinstance(actual, BattingStats)
                    actual_val = _get_batter_actual(actual, stat_name)

                if actual_val is None:
                    continue

                expected = float(expected_val)
                delta = actual_val - expected

                raw_deltas.setdefault(stat_name, []).append((player_id, player_name, actual_val, expected, delta))

        result: list[PlayerStatDelta] = []
        for stat_name, entries in raw_deltas.items():
            inverted = stat_name in _INVERTED_STATS
            perf_deltas = [(-e[4] if inverted else e[4]) for e in entries]

            sorted_indices = sorted(range(len(perf_deltas)), key=lambda i: perf_deltas[i])
            ranks: list[int] = [0] * len(perf_deltas)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1

            n = len(entries)
            for i, (player_id, player_name, actual_val, expected, delta) in enumerate(entries):
                perf_delta = perf_deltas[i]
                if n <= 1:
                    percentile = 50.0
                else:
                    percentile = (ranks[i] - 1) / (n - 1) * 100

                result.append(
                    PlayerStatDelta(
                        player_id=player_id,
                        player_name=player_name,
                        stat_name=stat_name,
                        actual=actual_val,
                        expected=expected,
                        delta=delta,
                        performance_delta=perf_delta,
                        percentile=percentile,
                    )
                )

        return result
