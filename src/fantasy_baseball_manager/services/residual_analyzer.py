from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scipy.stats import ks_2samp, ttest_1samp

from fantasy_baseball_manager.domain import (
    BattingStats,
    CohortBias,
    CohortBiasReport,
    ErrorDecompositionReport,
    FeatureGap,
    FeatureGapReport,
    MissPopulationSummary,
    PitchingStats,
    Player,
    PlayerResidual,
    PlayerType,
    bucket_by_age,
    bucket_by_experience,
    bucket_by_handedness,
    bucket_by_position,
    compute_age,
    compute_cohort_metrics,
    compute_miss_summary,
    rank_residuals,
    split_direction,
    split_residuals_by_quality,
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
        all_residuals, primary_positions, _ = self._compute_all_residuals(system, version, season, target, player_type)

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
            player_type=PlayerType(player_type),
            season=season,
            system=system,
            version=version,
            top_misses=top_misses,
            over_predictions=over,
            under_predictions=under,
            summary=summary,
        )

    def detect_feature_gaps(
        self,
        system: str,
        version: str,
        season: int,
        target: str,
        player_type: str,
        model_feature_names: frozenset[str],
        miss_percentile: float = 80.0,
        extra_features: dict[int, dict[str, float]] | None = None,
    ) -> FeatureGapReport:
        """Compare feature distributions between well-predicted and poorly-predicted players."""
        all_residuals, _, _ = self._compute_all_residuals(system, version, season, target, player_type)

        if not all_residuals:
            return FeatureGapReport(
                target=target,
                player_type=PlayerType(player_type),
                season=season,
                system=system,
                version=version,
            )

        # Merge extra features into each player's feature dict
        feature_dicts: dict[int, dict[str, float]] = {}
        for r in all_residuals:
            merged = dict(r.feature_values)
            if extra_features and r.player_id in extra_features:
                merged.update(extra_features[r.player_id])
            feature_dicts[r.player_id] = merged

        well, poor = split_residuals_by_quality(all_residuals, miss_percentile)

        if not well or not poor:
            return FeatureGapReport(
                target=target,
                player_type=PlayerType(player_type),
                season=season,
                system=system,
                version=version,
            )

        well_ids = {r.player_id for r in well}
        poor_ids = {r.player_id for r in poor}

        # Collect all feature names present in both groups
        well_feature_names: set[str] = set()
        for pid in well_ids:
            well_feature_names.update(feature_dicts[pid].keys())
        poor_feature_names: set[str] = set()
        for pid in poor_ids:
            poor_feature_names.update(feature_dicts[pid].keys())
        common_features = well_feature_names & poor_feature_names

        gaps: list[FeatureGap] = []
        for feature in common_features:
            well_vals = [feature_dicts[pid][feature] for pid in well_ids if feature in feature_dicts[pid]]
            poor_vals = [feature_dicts[pid][feature] for pid in poor_ids if feature in feature_dicts[pid]]

            if len(well_vals) < 2 or len(poor_vals) < 2:
                continue

            stat, p_value = ks_2samp(well_vals, poor_vals)
            gaps.append(
                FeatureGap(
                    feature_name=feature,
                    ks_statistic=float(stat),
                    p_value=float(p_value),
                    mean_well=sum(well_vals) / len(well_vals),
                    mean_poor=sum(poor_vals) / len(poor_vals),
                    in_model=feature in model_feature_names,
                )
            )

        gaps.sort(key=lambda g: g.ks_statistic, reverse=True)

        return FeatureGapReport(
            target=target,
            player_type=PlayerType(player_type),
            season=season,
            system=system,
            version=version,
            gaps=gaps,
        )

    def _compute_all_residuals(
        self,
        system: str,
        version: str,
        season: int,
        target: str,
        player_type: str,
    ) -> tuple[list[PlayerResidual], dict[int, str], dict[int, Player]]:
        """Compute residuals for all players and return primary positions and player objects.

        Returns (residuals, primary_positions, player_by_id).
        """
        projections = self._projection_repo.get_by_system_version(system, version)
        projections = [p for p in projections if p.season == season and p.player_type == player_type]

        if not projections:
            return [], {}, {}

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

        return all_residuals, primary_positions, player_by_id

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

    _HAND_REVERSE: dict[float, str] = {0.0: "L", 1.0: "R", 2.0: "S"}

    def bias_by_cohort(
        self,
        system: str,
        version: str,
        season: int,
        target: str,
        player_type: str,
        dimension: str,
    ) -> CohortBiasReport:
        """Compute cohort-level bias for the given dimension."""
        all_residuals, primary_positions, player_by_id = self._compute_all_residuals(
            system, version, season, target, player_type
        )

        if not all_residuals:
            return CohortBiasReport(
                target=target,
                player_type=PlayerType(player_type),
                season=season,
                system=system,
                version=version,
                dimension=dimension,
                cohorts=[],
            )

        buckets: dict[str, list[PlayerResidual]]
        if dimension == "age":
            buckets = bucket_by_age(all_residuals)
        elif dimension == "position":
            buckets = bucket_by_position(all_residuals, primary_positions)
        elif dimension == "handedness":
            handedness = self._extract_handedness(all_residuals, player_by_id, player_type)
            buckets = bucket_by_handedness(all_residuals, handedness)
        elif dimension == "experience":
            experience = self._compute_experience_years(all_residuals, season, player_type)
            buckets = bucket_by_experience(all_residuals, experience)
        else:
            msg = f"Unknown dimension: {dimension}"
            raise ValueError(msg)

        cohorts: list[CohortBias] = []
        for label, group in buckets.items():
            n = len(group)
            if n < 1:
                continue
            mean_residual, mean_abs_residual, rmse = compute_cohort_metrics(group)
            significant = False
            if n >= 2:
                residual_vals = [r.residual for r in group]
                _, p_value = ttest_1samp(residual_vals, 0.0)
                significant = float(p_value) < 0.05
            cohorts.append(
                CohortBias(
                    cohort_label=label,
                    n=n,
                    mean_residual=mean_residual,
                    mean_abs_residual=mean_abs_residual,
                    rmse=rmse,
                    significant=significant,
                )
            )

        cohorts.sort(key=lambda c: abs(c.mean_residual), reverse=True)

        return CohortBiasReport(
            target=target,
            player_type=PlayerType(player_type),
            season=season,
            system=system,
            version=version,
            dimension=dimension,
            cohorts=cohorts,
        )

    def bias_by_cohort_all_dimensions(
        self,
        system: str,
        version: str,
        season: int,
        target: str,
        player_type: str,
    ) -> list[CohortBiasReport]:
        """Run bias_by_cohort for all four dimensions."""
        return [
            self.bias_by_cohort(system, version, season, target, player_type, dim)
            for dim in ("age", "position", "handedness", "experience")
        ]

    def _extract_handedness(
        self,
        residuals: list[PlayerResidual],
        player_by_id: dict[int, Player],
        player_type: str,
    ) -> dict[int, str]:
        """Build handedness dict from player objects."""
        result: dict[int, str] = {}
        for r in residuals:
            player = player_by_id.get(r.player_id)
            if player is None:
                continue
            hand = player.throws if player_type == "pitcher" else player.bats
            if hand is not None:
                result[r.player_id] = hand
        return result

    def _compute_experience_years(
        self,
        residuals: list[PlayerResidual],
        season: int,
        player_type: str,
    ) -> dict[int, int]:
        """Count distinct seasons each player appeared in stats."""
        player_ids = {r.player_id for r in residuals}
        seasons_played: dict[int, set[int]] = {pid: set() for pid in player_ids}

        for year in range(season - 14, season + 1):
            if player_type == "pitcher":
                stats = self._pitching_repo.get_by_season(year, source="fangraphs")
            else:
                stats = self._batting_repo.get_by_season(year, source="fangraphs")
            for s in stats:
                if s.player_id in player_ids:
                    seasons_played[s.player_id].add(year)

        return {pid: len(years) for pid, years in seasons_played.items() if years}

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
            player_type=PlayerType(player_type),
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
