import dataclasses
from typing import Any

from scipy.stats import spearmanr

from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.valuation import (
    ValuationAccuracy,
    ValuationEvalResult,
)
from fantasy_baseball_manager.models.zar.engine import (
    compute_budget_split,
    run_zar_pipeline,
)
from fantasy_baseball_manager.models.zar.positions import build_position_map, build_roster_spots
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    PitchingStatsRepo,
    PlayerRepo,
    PositionAppearanceRepo,
    ValuationRepo,
)

_METADATA_FIELDS = frozenset({"id", "player_id", "season", "source", "team_id", "loaded_at"})


def _stats_to_dict(obj: object) -> dict[str, float]:
    """Extract all numeric fields from a BattingStats or PitchingStats instance."""
    result: dict[str, float] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if field.name in _METADATA_FIELDS:
            continue
        value = getattr(obj, field.name)
        if isinstance(value, int | float):
            result[field.name] = float(value)
    return result


class ValuationEvaluator:
    def __init__(
        self,
        valuation_repo: ValuationRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo,
    ) -> None:
        self._valuation_repo = valuation_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo
        self._position_repo = position_repo
        self._player_repo = player_repo

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        league: LeagueSettings,
        actuals_source: str = "fangraphs",
    ) -> ValuationEvalResult:
        # 1. Fetch predicted valuations, filter by version
        all_valuations = self._valuation_repo.get_by_season(season, system=system)
        predicted = [v for v in all_valuations if v.version == version]

        if not predicted:
            return ValuationEvalResult(
                system=system,
                version=version,
                season=season,
                value_mae=0.0,
                rank_correlation=0.0,
                n=0,
                players=[],
            )

        # 2. Fetch actual stats
        batting_actuals = self._batting_repo.get_by_season(season, source=actuals_source)
        pitching_actuals = self._pitching_repo.get_by_season(season, source=actuals_source)

        # 3. Convert actual stats to float dicts
        batter_stats: dict[int, dict[str, float]] = {}
        for bs in batting_actuals:
            batter_stats[bs.player_id] = _stats_to_dict(bs)

        pitcher_stats: dict[int, dict[str, float]] = {}
        for ps in pitching_actuals:
            pitcher_stats[ps.player_id] = _stats_to_dict(ps)

        # 4. Fetch positions
        appearances = self._position_repo.get_by_season(season)
        position_map = build_position_map(appearances, league)

        # 5. Run ZAR pipeline on actuals — same split/budget/pipeline as ZarModel.predict
        actual_batter_ids = [pid for pid in batter_stats if batter_stats[pid].get("pa", 0) > 0]
        actual_pitcher_ids = [pid for pid in pitcher_stats if pitcher_stats[pid].get("ip", 0) > 0]

        batter_budget, pitcher_budget = compute_budget_split(league)

        # Value batters
        actual_batter_values = self._value_pool(
            player_ids=actual_batter_ids,
            stats_map=batter_stats,
            categories=list(league.batting_categories),
            position_map=position_map,
            league=league,
            budget=batter_budget,
            player_type="batter",
        )

        # Value pitchers
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

        # Combine and rank — keyed by (player_id, player_type)
        all_actual: dict[tuple[int, str], float] = {}
        all_actual.update(actual_batter_values)
        all_actual.update(actual_pitcher_values)

        # Rank actuals by value descending
        actual_ranked = sorted(all_actual.items(), key=lambda x: x[1], reverse=True)
        actual_rank_map: dict[tuple[int, str], int] = {key: rank for rank, (key, _) in enumerate(actual_ranked, 1)}

        # 6. Match predicted vs actual by (player_id, player_type)
        predicted_map = {(v.player_id, v.player_type): v for v in predicted}
        # Build player name lookup
        player_name_map = self._build_player_name_map()

        matched: list[ValuationAccuracy] = []
        for (player_id, player_type), pred_val in predicted_map.items():
            key = (player_id, player_type)
            if key not in all_actual:
                continue
            actual_value = all_actual[key]
            actual_rank = actual_rank_map[key]
            surplus = round(pred_val.value - actual_value, 2)
            name = player_name_map.get(player_id, f"Player {player_id}")
            matched.append(
                ValuationAccuracy(
                    player_id=player_id,
                    player_name=name,
                    player_type=pred_val.player_type,
                    position=pred_val.position,
                    predicted_value=pred_val.value,
                    actual_value=round(actual_value, 2),
                    surplus=surplus,
                    predicted_rank=pred_val.rank,
                    actual_rank=actual_rank,
                )
            )

        if not matched:
            return ValuationEvalResult(
                system=system,
                version=version,
                season=season,
                value_mae=0.0,
                rank_correlation=0.0,
                n=0,
                players=[],
            )

        # 7. Compute metrics
        value_mae = sum(abs(p.predicted_value - p.actual_value) for p in matched) / len(matched)
        predicted_ranks = [p.predicted_rank for p in matched]
        actual_ranks = [p.actual_rank for p in matched]

        if len(matched) >= 3:
            corr, _ = spearmanr(predicted_ranks, actual_ranks)
            rank_correlation = float(corr)
        else:
            rank_correlation = 0.0

        # 8. Sort by absolute surplus descending
        matched.sort(key=lambda p: abs(p.surplus), reverse=True)

        return ValuationEvalResult(
            system=system,
            version=version,
            season=season,
            value_mae=round(value_mae, 2),
            rank_correlation=round(rank_correlation, 4),
            n=len(matched),
            players=matched,
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
        """Run ZAR pipeline on a player pool and return {(player_id, player_type): dollar_value}."""
        if not player_ids:
            return {}

        stats_list = [stats_map[pid] for pid in player_ids]
        no_pos: list[str] = ["util"] if league.roster_util > 0 else []
        player_positions = [position_map.get(pid, no_pos) for pid in player_ids]
        roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

        result = run_zar_pipeline(stats_list, categories, player_positions, roster_spots, league.teams, budget)

        return {(pid, player_type): val for pid, val in zip(player_ids, result.dollar_values)}

    def _build_player_name_map(self) -> dict[int, str]:
        """Build a mapping of player_id to display name."""
        players = self._player_repo.all()
        result: dict[int, str] = {}
        for p in players:
            if p.id is not None:
                result[p.id] = f"{p.name_first} {p.name_last}"
        return result
