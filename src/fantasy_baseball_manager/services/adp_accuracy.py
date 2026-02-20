import dataclasses
import logging
import math
from typing import Any

from scipy.stats import spearmanr

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.adp_accuracy import (
    ADPAccuracyPlayer,
    ADPAccuracyReport,
    ADPAccuracyResult,
    SystemAccuracyResult,
)
from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.models.zar.engine import compute_budget_split, run_zar_pipeline
from fantasy_baseball_manager.models.zar.positions import build_position_map, build_roster_spots
from fantasy_baseball_manager.repos.protocols import (
    ADPRepo,
    BattingStatsRepo,
    PitchingStatsRepo,
    PlayerRepo,
    PositionAppearanceRepo,
    ValuationRepo,
)

logger = logging.getLogger(__name__)

_PITCHER_POSITIONS = {"SP", "RP"}
_METADATA_FIELDS = frozenset({"id", "player_id", "season", "source", "team_id", "loaded_at"})
_TOP_N_THRESHOLDS = (50, 100, 200)


def _is_pitcher_adp(adp: ADP) -> bool:
    return all(p.strip() in _PITCHER_POSITIONS for p in adp.positions.split(",") if p.strip())


def _stats_to_dict(obj: object) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if field.name in _METADATA_FIELDS:
            continue
        value = getattr(obj, field.name)
        if isinstance(value, int | float):
            result[field.name] = float(value)
    return result


class ADPAccuracyEvaluator:
    def __init__(
        self,
        adp_repo: ADPRepo,
        valuation_repo: ValuationRepo,
        player_repo: PlayerRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
        position_repo: PositionAppearanceRepo,
    ) -> None:
        self._adp_repo = adp_repo
        self._valuation_repo = valuation_repo
        self._player_repo = player_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo
        self._position_repo = position_repo

    def evaluate(
        self,
        seasons: list[int],
        league: LeagueSettings,
        provider: str = "fantasypros",
        compare_system: tuple[str, str] | None = None,
    ) -> ADPAccuracyReport:
        adp_results: list[ADPAccuracyResult] = []
        comparison: list[SystemAccuracyResult] | None = None if compare_system is None else []

        for season in seasons:
            actual_values = self._compute_actual_values(season, league)
            adp_result = self._evaluate_season(season, league, provider, actual_values)
            adp_results.append(adp_result)

            if compare_system is not None:
                sys_name, version = compare_system
                sys_result = self._evaluate_system(season, sys_name, version, actual_values)
                assert comparison is not None
                comparison.append(sys_result)

        # Aggregate across seasons
        valid_corrs = [r.rank_correlation for r in adp_results if r.n_matched >= 3]
        mean_rank_correlation = sum(valid_corrs) / len(valid_corrs) if valid_corrs else 0.0

        valid_rmses = [r.value_rmse for r in adp_results if r.n_matched > 0]
        mean_value_rmse = sum(valid_rmses) / len(valid_rmses) if valid_rmses else 0.0

        mean_top_n: dict[int, float] = {}
        for n in _TOP_N_THRESHOLDS:
            precisions = [r.top_n_precision[n] for r in adp_results if n in r.top_n_precision]
            mean_top_n[n] = sum(precisions) / len(precisions) if precisions else 0.0

        return ADPAccuracyReport(
            provider=provider,
            seasons=seasons,
            adp_results=adp_results,
            comparison=comparison,
            mean_rank_correlation=round(mean_rank_correlation, 4),
            mean_value_rmse=round(mean_value_rmse, 2),
            mean_top_n_precision=mean_top_n,
        )

    def _compute_actual_values(
        self,
        season: int,
        league: LeagueSettings,
    ) -> dict[tuple[int, str], float]:
        batting_actuals = self._batting_repo.get_by_season(season, source="fangraphs")
        pitching_actuals = self._pitching_repo.get_by_season(season, source="fangraphs")

        batter_stats: dict[int, dict[str, float]] = {}
        for bs in batting_actuals:
            batter_stats[bs.player_id] = _stats_to_dict(bs)

        pitcher_stats: dict[int, dict[str, float]] = {}
        for ps in pitching_actuals:
            pitcher_stats[ps.player_id] = _stats_to_dict(ps)

        appearances = self._position_repo.get_by_season(season)
        position_map = build_position_map(appearances, league)

        actual_batter_ids = [pid for pid in batter_stats if batter_stats[pid].get("pa", 0) > 0]
        actual_pitcher_ids = [pid for pid in pitcher_stats if pitcher_stats[pid].get("ip", 0) > 0]

        batter_budget, pitcher_budget = compute_budget_split(league)

        actual_batter_values = self._value_pool(
            player_ids=actual_batter_ids,
            stats_map=batter_stats,
            categories=list(league.batting_categories),
            position_map=position_map,
            league=league,
            budget=batter_budget,
            player_type="batter",
        )

        pitcher_position_map: dict[int, list[str]] = {pid: ["p"] for pid in actual_pitcher_ids}
        actual_pitcher_values = self._value_pool(
            player_ids=actual_pitcher_ids,
            stats_map=pitcher_stats,
            categories=list(league.pitching_categories),
            position_map=pitcher_position_map,
            league=league,
            budget=pitcher_budget,
            player_type="pitcher",
            pitcher_roster_spots={"p": league.roster_pitchers},
        )

        all_actual: dict[tuple[int, str], float] = {}
        all_actual.update(actual_batter_values)
        all_actual.update(actual_pitcher_values)
        return all_actual

    def _evaluate_season(
        self,
        season: int,
        league: LeagueSettings,
        provider: str,
        actual_values: dict[tuple[int, str], float],
    ) -> ADPAccuracyResult:
        adp_list = self._adp_repo.get_by_season(season, provider=provider)

        # Build ADP map: lowest overall_pick per player
        adp_by_player: dict[int, list[ADP]] = {}
        for adp in adp_list:
            adp_by_player.setdefault(adp.player_id, []).append(adp)

        adp_map: dict[int, ADP] = {}
        for pid, entries in adp_by_player.items():
            adp_map[pid] = min(entries, key=lambda a: a.overall_pick)

        # Rank actuals descending
        actual_ranked = sorted(actual_values.items(), key=lambda x: x[1], reverse=True)
        actual_rank_map: dict[tuple[int, str], int] = {key: rank for rank, (key, _) in enumerate(actual_ranked, 1)}

        # Build player name map
        player_name_map = self._build_player_name_map()

        # Match ADP entries to actual values
        matched: list[tuple[int, ADP, str, float, int]] = []  # (rank, adp, player_type, actual_val, actual_rank)
        for pid, adp in sorted(adp_map.items(), key=lambda x: x[1].overall_pick):
            player_type = "pitcher" if _is_pitcher_adp(adp) else "batter"
            key = (pid, player_type)
            if key in actual_values:
                matched.append((adp.rank, adp, player_type, actual_values[key], actual_rank_map[key]))

        n_matched = len(matched)

        if n_matched == 0:
            return ADPAccuracyResult(
                season=season,
                provider=provider,
                rank_correlation=0.0,
                value_rmse=0.0,
                value_mae=0.0,
                top_n_precision={n: 0.0 for n in _TOP_N_THRESHOLDS},
                n_matched=0,
                players=[],
            )

        # Sort matched by ADP rank ascending
        matched.sort(key=lambda x: x[0])

        # Build value curve: sort matched players' actual values descending
        actual_values_sorted = sorted([m[3] for m in matched], reverse=True)

        # Build per-player breakdown
        players: list[ADPAccuracyPlayer] = []
        adp_ranks: list[int] = []
        actual_ranks: list[int] = []
        squared_errors: list[float] = []
        abs_errors: list[float] = []

        for i, (adp_rank, adp, player_type, actual_val, actual_rank) in enumerate(matched):
            implied_value = actual_values_sorted[i] if i < len(actual_values_sorted) else 0.0
            value_error = implied_value - actual_val
            name = player_name_map.get(adp.player_id, f"Player {adp.player_id}")

            players.append(
                ADPAccuracyPlayer(
                    player_id=adp.player_id,
                    player_name=name,
                    player_type=player_type,
                    adp_rank=adp_rank,
                    actual_rank=actual_rank,
                    actual_value=round(actual_val, 2),
                    implied_value=round(implied_value, 2),
                    value_error=round(value_error, 2),
                )
            )

            adp_ranks.append(adp_rank)
            actual_ranks.append(actual_rank)
            squared_errors.append(value_error**2)
            abs_errors.append(abs(value_error))

        # Spearman correlation
        if n_matched >= 3:
            corr, _ = spearmanr(adp_ranks, actual_ranks)
            rank_correlation = float(corr)
        else:
            rank_correlation = 0.0

        # RMSE and MAE
        value_rmse = math.sqrt(sum(squared_errors) / n_matched)
        value_mae = sum(abs_errors) / n_matched

        # Top-N precision
        top_n_precision: dict[int, float] = {}
        for n in _TOP_N_THRESHOLDS:
            capped_n = min(n, n_matched)
            adp_top_n_ids = {(matched[i][1].player_id, matched[i][2]) for i in range(capped_n)}
            # Actual top-N: lowest actual_rank
            actual_top_n = sorted(matched, key=lambda x: x[4])[:capped_n]
            actual_top_n_ids = {(m[1].player_id, m[2]) for m in actual_top_n}
            overlap = len(adp_top_n_ids & actual_top_n_ids)
            top_n_precision[n] = overlap / capped_n if capped_n > 0 else 0.0

        # Sort players by |value_error| descending
        players.sort(key=lambda p: abs(p.value_error), reverse=True)

        return ADPAccuracyResult(
            season=season,
            provider=provider,
            rank_correlation=round(rank_correlation, 4),
            value_rmse=round(value_rmse, 2),
            value_mae=round(value_mae, 2),
            top_n_precision=top_n_precision,
            n_matched=n_matched,
            players=players,
        )

    def _evaluate_system(
        self,
        season: int,
        system: str,
        version: str,
        actual_values: dict[tuple[int, str], float],
    ) -> SystemAccuracyResult:
        valuations = self._valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        # Build predicted map
        predicted_map: dict[tuple[int, str], tuple[float, int]] = {
            (v.player_id, v.player_type): (v.value, v.rank) for v in valuations
        }

        # Rank actuals
        actual_ranked = sorted(actual_values.items(), key=lambda x: x[1], reverse=True)
        actual_rank_map: dict[tuple[int, str], int] = {key: rank for rank, (key, _) in enumerate(actual_ranked, 1)}

        # Match
        matched_pred_ranks: list[int] = []
        matched_actual_ranks: list[int] = []
        matched_keys: list[tuple[int, str]] = []
        squared_errors: list[float] = []
        abs_errors: list[float] = []

        for key, (pred_value, pred_rank) in predicted_map.items():
            if key not in actual_values:
                continue
            actual_val = actual_values[key]
            actual_rank = actual_rank_map[key]
            matched_pred_ranks.append(pred_rank)
            matched_actual_ranks.append(actual_rank)
            matched_keys.append(key)
            error = pred_value - actual_val
            squared_errors.append(error**2)
            abs_errors.append(abs(error))

        n_matched = len(matched_keys)

        if n_matched == 0:
            return SystemAccuracyResult(
                system=system,
                version=version,
                season=season,
                rank_correlation=0.0,
                value_rmse=0.0,
                value_mae=0.0,
                top_n_precision={n: 0.0 for n in _TOP_N_THRESHOLDS},
                n_matched=0,
            )

        # Spearman
        if n_matched >= 3:
            corr, _ = spearmanr(matched_pred_ranks, matched_actual_ranks)
            rank_correlation = float(corr)
        else:
            rank_correlation = 0.0

        value_rmse = math.sqrt(sum(squared_errors) / n_matched)
        value_mae = sum(abs_errors) / n_matched

        # Top-N precision
        top_n_precision: dict[int, float] = {}
        pred_sorted = sorted(zip(matched_keys, matched_pred_ranks), key=lambda x: x[1])
        actual_sorted = sorted(zip(matched_keys, matched_actual_ranks), key=lambda x: x[1])

        for n in _TOP_N_THRESHOLDS:
            capped_n = min(n, n_matched)
            pred_top = {k for k, _ in pred_sorted[:capped_n]}
            actual_top = {k for k, _ in actual_sorted[:capped_n]}
            overlap = len(pred_top & actual_top)
            top_n_precision[n] = overlap / capped_n if capped_n > 0 else 0.0

        return SystemAccuracyResult(
            system=system,
            version=version,
            season=season,
            rank_correlation=round(rank_correlation, 4),
            value_rmse=round(value_rmse, 2),
            value_mae=round(value_mae, 2),
            top_n_precision=top_n_precision,
            n_matched=n_matched,
        )

    def _value_pool(
        self,
        player_ids: list[int],
        stats_map: dict[int, dict[str, float]],
        categories: list[Any],
        position_map: dict[int, list[str]],
        league: LeagueSettings,
        budget: float,
        *,
        player_type: str,
        pitcher_roster_spots: dict[str, int] | None = None,
    ) -> dict[tuple[int, str], float]:
        if not player_ids:
            return {}

        stats_list = [stats_map[pid] for pid in player_ids]
        no_pos: list[str] = ["util"] if league.roster_util > 0 else []
        player_positions = [position_map.get(pid, no_pos) for pid in player_ids]
        roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

        result = run_zar_pipeline(stats_list, categories, player_positions, roster_spots, league.teams, budget)

        return {(pid, player_type): val for pid, val in zip(player_ids, result.dollar_values)}

    def _build_player_name_map(self) -> dict[int, str]:
        players = self._player_repo.all()
        result: dict[int, str] = {}
        for p in players:
            if p.id is not None:
                result[p.id] = f"{p.name_first} {p.name_last}"
        return result
