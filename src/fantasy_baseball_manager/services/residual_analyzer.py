from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    BattingStats,
    ErrorDecompositionReport,
    MissPopulationSummary,
    PitchingStats,
    PlayerResidual,
    compute_age,
    compute_miss_summary,
    rank_residuals,
    split_direction,
)
from fantasy_baseball_manager.services.performance_report import _get_batter_actual, _get_pitcher_actual

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import (
        BattingStatsRepo,
        PitchingStatsRepo,
        PlayerRepo,
        PositionAppearanceRepo,
        ProjectionRepo,
    )

logger = logging.getLogger(__name__)

_HAND_ENCODING: dict[str | None, float] = {"L": 0.0, "R": 1.0, "S": 2.0}


class ResidualAnalyzer:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
        player_repo: PlayerRepo,
        position_appearance_repo: PositionAppearanceRepo,
    ) -> None:
        self._projection_repo = projection_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo
        self._player_repo = player_repo
        self._position_appearance_repo = position_appearance_repo

    def analyze(
        self,
        system: str,
        version: str,
        season: int,
        target: str,
        player_type: str,
        top_n: int = 20,
        direction: str | None = None,
    ) -> ErrorDecompositionReport:
        projections = self._projection_repo.get_by_system_version(system, version)
        projections = [p for p in projections if p.season == season and p.player_type == player_type]

        if not projections:
            return self._empty_report(target, player_type, season, system, version)

        proj_by_player = {p.player_id: p for p in projections}
        player_ids = list(proj_by_player.keys())

        players = self._player_repo.get_by_ids(player_ids)
        player_by_id = {p.id: p for p in players if p.id is not None}

        if player_type == "pitcher":
            actuals_list = self._pitching_repo.get_by_season(season, source="fangraphs")
            actuals_by_player: dict[int, BattingStats | PitchingStats] = {a.player_id: a for a in actuals_list}
        else:
            bat_actuals_list = self._batting_repo.get_by_season(season, source="fangraphs")
            actuals_by_player = {a.player_id: a for a in bat_actuals_list}

        position_appearances = self._position_appearance_repo.get_by_season(season)
        primary_positions: dict[int, str] = {}
        position_games: dict[int, dict[str, int]] = {}
        for pa in position_appearances:
            position_games.setdefault(pa.player_id, {})[pa.position] = pa.games
        for pid, pos_map in position_games.items():
            primary_positions[pid] = max(pos_map, key=pos_map.get)  # type: ignore[arg-type]

        all_residuals: list[PlayerResidual] = []
        for player_id, proj in proj_by_player.items():
            actual = actuals_by_player.get(player_id)
            if actual is None:
                continue

            predicted_val = proj.stat_json.get(target)
            if predicted_val is None:
                continue

            if player_type == "pitcher":
                assert isinstance(actual, PitchingStats)  # noqa: S101
                actual_val = _get_pitcher_actual(actual, target)
            else:
                assert isinstance(actual, BattingStats)  # noqa: S101
                actual_val = _get_batter_actual(actual, target)

            if actual_val is None:
                continue

            predicted = float(predicted_val)
            residual = predicted - actual_val

            player = player_by_id.get(player_id)
            player_name = f"{player.name_first} {player.name_last}" if player else str(player_id)

            feature_values = self._build_feature_values(actual, player, season)

            all_residuals.append(
                PlayerResidual(
                    player_id=player_id,
                    player_name=player_name,
                    predicted=predicted,
                    actual=actual_val,
                    residual=residual,
                    feature_values=feature_values,
                )
            )

        if not all_residuals:
            return self._empty_report(target, player_type, season, system, version)

        if direction == "over":
            all_residuals = [r for r in all_residuals if r.residual > 0]
        elif direction == "under":
            all_residuals = [r for r in all_residuals if r.residual < 0]

        top_misses = rank_residuals(all_residuals, top_n)
        over, under = split_direction(all_residuals)

        top_miss_ids = {r.player_id for r in top_misses}
        rest = [r for r in all_residuals if r.player_id not in top_miss_ids]

        summary = compute_miss_summary(top_misses, rest, primary_positions)

        return ErrorDecompositionReport(
            target=target,
            player_type=player_type,
            season=season,
            system=system,
            version=version,
            top_misses=top_misses,
            over_predictions=over,
            under_predictions=under,
            summary=summary,
        )

    def _build_feature_values(
        self,
        actual: BattingStats | PitchingStats,
        player: object | None,
        season: int,
    ) -> dict[str, float]:
        features: dict[str, float] = {}

        if hasattr(player, "birth_date"):
            age = compute_age(getattr(player, "birth_date", None), season)
            if age is not None:
                features["age"] = float(age)

        if hasattr(player, "bats"):
            bats_val = getattr(player, "bats", None)
            encoded = _HAND_ENCODING.get(bats_val)
            if encoded is not None:
                features["bats"] = encoded

        if hasattr(player, "throws"):
            throws_val = getattr(player, "throws", None)
            encoded = _HAND_ENCODING.get(throws_val)
            if encoded is not None:
                features["throws"] = encoded

        if isinstance(actual, BattingStats):
            if actual.pa is not None:
                features["pa"] = float(actual.pa)
            for attr in ("avg", "obp", "slg", "woba", "hr", "sb", "bb", "so", "war"):
                val = getattr(actual, attr, None)
                if val is not None:
                    features[attr] = float(val)
        elif isinstance(actual, PitchingStats):
            if actual.ip is not None:
                features["ip"] = float(actual.ip)
            for attr in ("era", "fip", "whip", "k_per_9", "bb_per_9", "so", "bb", "hr", "war"):
                val = getattr(actual, attr, None)
                if val is not None:
                    features[attr] = float(val)

        return features

    @staticmethod
    def _empty_report(
        target: str,
        player_type: str,
        season: int,
        system: str,
        version: str,
    ) -> ErrorDecompositionReport:
        return ErrorDecompositionReport(
            target=target,
            player_type=player_type,
            season=season,
            system=system,
            version=version,
            top_misses=[],
            over_predictions=[],
            under_predictions=[],
            summary=MissPopulationSummary(
                mean_age=None,
                position_distribution={},
                mean_volume=0.0,
                distinguishing_features=[],
            ),
        )
