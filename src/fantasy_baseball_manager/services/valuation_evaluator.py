import logging
from typing import TYPE_CHECKING, Any

from scipy.stats import spearmanr

from fantasy_baseball_manager.domain import ValuationAccuracy, ValuationComparisonResult, ValuationEvalResult
from fantasy_baseball_manager.models.zar.engine import (
    compute_budget_split,
    run_zar_pipeline,
)
from fantasy_baseball_manager.models.zar.positions import build_position_map, build_roster_spots
from fantasy_baseball_manager.services.stats_conversion import stats_to_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain import CategoryConfig, LeagueSettings
    from fantasy_baseball_manager.repos import (
        BattingStatsRepo,
        PitchingStatsRepo,
        PlayerRepo,
        PositionAppearanceRepo,
        ValuationRepo,
    )
logger = logging.getLogger(__name__)


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
        *,
        top: int | None = None,
        min_value: float | None = None,
        targets: frozenset[str] | None = None,
        stratify: str | None = None,
        tail_ns: tuple[int, ...] | None = None,
    ) -> ValuationEvalResult:
        logger.info("Evaluating valuations: %s/%s season=%d", system, version, season)
        # 1. Fetch predicted valuations, filter by version
        predicted = self._valuation_repo.get_by_season(season, system=system, version=version)

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

        # 2b. Build WAR lookup and games-started lookup
        war_lookup: dict[tuple[int, str], float] = {}
        for bs in batting_actuals:
            if bs.war is not None:
                war_lookup[(bs.player_id, "batter")] = bs.war
        gs_lookup: dict[int, int] = {}
        for ps in pitching_actuals:
            if ps.war is not None:
                war_lookup[(ps.player_id, "pitcher")] = ps.war
            if ps.gs is not None:
                gs_lookup[ps.player_id] = ps.gs

        # 3. Convert actual stats to float dicts
        batter_stats: dict[int, dict[str, float]] = {}
        for bs in batting_actuals:
            batter_stats[bs.player_id] = stats_to_dict(bs)

        pitcher_stats: dict[int, dict[str, float]] = {}
        for ps in pitching_actuals:
            pitcher_stats[ps.player_id] = stats_to_dict(ps)

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
                    actual_war=war_lookup.get(key),
                    games_started=gs_lookup.get(player_id) if player_type == "pitcher" else None,
                    category_scores=dict(pred_val.category_scores) if pred_val.category_scores else None,
                )
            )

        # 6b. Apply population filters
        filtering = min_value is not None or top is not None
        total_matched = len(matched) if filtering else None

        if min_value is not None:
            matched = [p for p in matched if p.predicted_value > min_value or p.actual_value > min_value]

        if top is not None:
            matched.sort(key=lambda p: p.predicted_rank)
            matched = matched[:top]

        # Build filter description
        filter_description: str | None = None
        if filtering:
            parts: list[str] = []
            if min_value is not None:
                parts.append(f"pred|act > ${min_value}")
            if top is not None:
                parts.append(f"top {top} by pred rank")
            filter_description = ", ".join(parts)

        if not matched:
            return ValuationEvalResult(
                system=system,
                version=version,
                season=season,
                value_mae=0.0,
                rank_correlation=0.0,
                n=0,
                players=[],
                total_matched=total_matched,
                filter_description=filter_description,
            )

        # 7. Compute metrics
        metrics = self._compute_metrics(matched, targets)

        # 7b. Cohorts (stratification)
        cohorts: dict[str, ValuationEvalResult] | None = None
        if stratify == "player_type":
            groups: dict[str, list[ValuationAccuracy]] = {}
            for p in matched:
                groups.setdefault(p.player_type, []).append(p)
            cohorts = {}
            for ptype, group in sorted(groups.items()):
                cm = self._compute_metrics(group, targets)
                cohorts[ptype] = ValuationEvalResult(
                    system=system,
                    version=version,
                    season=season,
                    value_mae=cm.value_mae,
                    rank_correlation=cm.rank_correlation,
                    n=cm.n,
                    players=sorted(group, key=lambda p: abs(p.surplus), reverse=True),
                    war_correlation=cm.war_correlation,
                    war_correlation_batters=cm.war_correlation_batters,
                    war_correlation_pitchers=cm.war_correlation_pitchers,
                    war_correlation_sp=cm.war_correlation_sp,
                    hit_rates=cm.hit_rates,
                )

        # 7c. Tail accuracy
        tail_results: dict[int, ValuationEvalResult] | None = None
        if tail_ns is not None:
            tail_results = {}
            by_pred_rank = sorted(matched, key=lambda p: p.predicted_rank)
            for n in tail_ns:
                if n > len(matched):
                    continue
                tail_slice = by_pred_rank[:n]
                tm = self._compute_metrics(tail_slice, targets)
                tail_results[n] = ValuationEvalResult(
                    system=system,
                    version=version,
                    season=season,
                    value_mae=tm.value_mae,
                    rank_correlation=tm.rank_correlation,
                    n=tm.n,
                    players=sorted(tail_slice, key=lambda p: abs(p.surplus), reverse=True),
                    war_correlation=tm.war_correlation,
                    war_correlation_batters=tm.war_correlation_batters,
                    war_correlation_pitchers=tm.war_correlation_pitchers,
                    war_correlation_sp=tm.war_correlation_sp,
                    hit_rates=tm.hit_rates,
                )

        # 7d. Per-category hit rates
        category_hit_rates: dict[str, float] | None = None
        if targets is None or "category-hit-rate" in targets:
            all_actual_stats = {**batter_stats, **pitcher_stats}
            all_categories = list(league.batting_categories) + list(league.pitching_categories)
            category_hit_rates = self._compute_category_hit_rates(matched, all_actual_stats, all_categories)
            if not category_hit_rates:
                category_hit_rates = None

        # 8. Sort by absolute surplus descending
        matched.sort(key=lambda p: abs(p.surplus), reverse=True)

        logger.debug("Valuation evaluation: %d matched players", len(matched))
        return ValuationEvalResult(
            system=system,
            version=version,
            season=season,
            value_mae=metrics.value_mae,
            rank_correlation=metrics.rank_correlation,
            n=metrics.n,
            players=matched,
            total_matched=total_matched,
            filter_description=filter_description,
            war_correlation=metrics.war_correlation,
            war_correlation_batters=metrics.war_correlation_batters,
            war_correlation_pitchers=metrics.war_correlation_pitchers,
            war_correlation_sp=metrics.war_correlation_sp,
            hit_rates=metrics.hit_rates,
            category_hit_rates=category_hit_rates,
            cohorts=cohorts,
            tail_results=tail_results,
        )

    def compare(
        self,
        baseline_system: str,
        baseline_version: str,
        candidate_system: str,
        candidate_version: str,
        season: int,
        league: LeagueSettings,
        *,
        min_value: float | None,
        top: int | None,
        targets: frozenset[str] | None,
        stratify: str | None,
        tail_ns: tuple[int, ...] | None,
    ) -> ValuationComparisonResult:
        """Evaluate two valuation systems and return both results for comparison."""
        baseline = self.evaluate(
            baseline_system,
            baseline_version,
            season,
            league,
            top=top,
            min_value=min_value,
            targets=targets,
            stratify=stratify,
            tail_ns=tail_ns,
        )
        candidate = self.evaluate(
            candidate_system,
            candidate_version,
            season,
            league,
            top=top,
            min_value=min_value,
            targets=targets,
            stratify=stratify,
            tail_ns=tail_ns,
        )
        return ValuationComparisonResult(season=season, baseline=baseline, candidate=candidate)

    @staticmethod
    def _compute_metrics(
        matched: list[ValuationAccuracy],
        targets: frozenset[str] | None,
    ) -> ValuationEvalResult:
        """Compute MAE, rank ρ, WAR ρ, and hit rates for a player list.

        Returns a minimal ValuationEvalResult holding only the metric fields.
        """
        n = len(matched)
        if n == 0:
            return ValuationEvalResult(
                system="", version="", season=0, value_mae=0.0, rank_correlation=0.0, n=0, players=[]
            )

        value_mae = round(sum(abs(p.predicted_value - p.actual_value) for p in matched) / n, 2)

        if n >= 3:
            corr, _ = spearmanr(
                [p.predicted_rank for p in matched],
                [p.actual_rank for p in matched],
            )
            rank_correlation = round(float(corr), 4)
        else:
            rank_correlation = 0.0

        war_correlation: float | None = None
        war_correlation_batters: float | None = None
        war_correlation_pitchers: float | None = None
        war_correlation_sp: float | None = None
        if targets is None or "war" in targets:
            war_correlation, war_correlation_batters, war_correlation_pitchers, war_correlation_sp = (
                ValuationEvaluator._compute_war_correlations(matched)
            )

        hit_rates: dict[int, float] | None = None
        if targets is None or "hit-rate" in targets:
            hit_rates = ValuationEvaluator._compute_hit_rates(matched)

        return ValuationEvalResult(
            system="",
            version="",
            season=0,
            value_mae=value_mae,
            rank_correlation=rank_correlation,
            n=n,
            players=[],
            war_correlation=war_correlation,
            war_correlation_batters=war_correlation_batters,
            war_correlation_pitchers=war_correlation_pitchers,
            war_correlation_sp=war_correlation_sp,
            hit_rates=hit_rates,
        )

    @staticmethod
    def _compute_war_correlations(
        matched: list[ValuationAccuracy],
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Compute Spearman ρ of predicted $ vs actual WAR (overall, batters, pitchers, SP)."""
        with_war = [(p.predicted_value, p.actual_war, p.player_type) for p in matched if p.actual_war is not None]
        if len(with_war) < 3:
            return None, None, None, None

        pred_vals = [w[0] for w in with_war]
        wars = [w[1] for w in with_war]
        corr, _ = spearmanr(pred_vals, wars)
        war_correlation = round(float(corr), 4)

        # Batters
        batter_data = [(pv, w) for pv, w, pt in with_war if pt == "batter"]
        war_correlation_batters: float | None = None
        if len(batter_data) >= 3:
            corr_b, _ = spearmanr([b[0] for b in batter_data], [b[1] for b in batter_data])
            war_correlation_batters = round(float(corr_b), 4)

        # Pitchers (all)
        pitcher_data = [(pv, w) for pv, w, pt in with_war if pt == "pitcher"]
        war_correlation_pitchers: float | None = None
        if len(pitcher_data) >= 3:
            corr_p, _ = spearmanr([p[0] for p in pitcher_data], [p[1] for p in pitcher_data])
            war_correlation_pitchers = round(float(corr_p), 4)

        # SP only (pitchers with gs >= 5)
        sp_data = [
            (p.predicted_value, p.actual_war)
            for p in matched
            if p.actual_war is not None
            and p.player_type == "pitcher"
            and p.games_started is not None
            and p.games_started >= 5
        ]
        war_correlation_sp: float | None = None
        if len(sp_data) >= 3:
            corr_sp, _ = spearmanr([s[0] for s in sp_data], [s[1] for s in sp_data])
            war_correlation_sp = round(float(corr_sp), 4)

        return war_correlation, war_correlation_batters, war_correlation_pitchers, war_correlation_sp

    @staticmethod
    def _compute_hit_rates(matched: list[ValuationAccuracy]) -> dict[int, float]:
        """Compute top-N hit rates for N in (25, 50, 100) where N ≤ len(matched)."""
        hit_rates: dict[int, float] = {}
        for n in (25, 50, 100):
            if n > len(matched):
                continue
            predicted_top = {p.player_id for p in sorted(matched, key=lambda p: p.predicted_rank)[:n]}
            actual_top = {p.player_id for p in sorted(matched, key=lambda p: p.actual_rank)[:n]}
            hit_rates[n] = round(len(predicted_top & actual_top) / n * 100, 1)
        return hit_rates

    @staticmethod
    def _compute_category_hit_rates(
        matched: list[ValuationAccuracy],
        actual_stats: dict[int, dict[str, float]],
        categories: Sequence[CategoryConfig],
        top_n: int = 20,
    ) -> dict[str, float]:
        """Compute per-category top-N hit rates.

        For each category, compare the top-N players by predicted category score
        against the top-N players by actual stat value. Returns hit rate as a
        percentage for each category.
        """
        result: dict[str, float] = {}
        for cat in categories:
            # Predicted top-N: players sorted by their category_scores[cat.key]
            with_scores = [
                (p.player_id, p.category_scores[cat.key])
                for p in matched
                if p.category_scores is not None and cat.key in p.category_scores
            ]
            if len(with_scores) < top_n:
                continue
            with_scores.sort(key=lambda x: x[1], reverse=True)
            predicted_top = {pid for pid, _ in with_scores[:top_n]}

            # Actual top-N: players sorted by actual stat value
            with_actuals = [
                (pid, actual_stats[pid].get(cat.key, 0.0))
                for pid in {p.player_id for p in matched}
                if pid in actual_stats and cat.key in actual_stats[pid]
            ]
            if len(with_actuals) < top_n:
                continue
            with_actuals.sort(key=lambda x: x[1], reverse=True)
            actual_top = {pid for pid, _ in with_actuals[:top_n]}

            result[cat.key] = round(len(predicted_top & actual_top) / top_n * 100, 1)
        return result

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

        result = run_zar_pipeline(
            stats_list, categories, player_positions, roster_spots, league.teams, budget, use_optimal_assignment=False
        )

        return {(pid, player_type): val for pid, val in zip(player_ids, result.dollar_values, strict=True)}

    def _build_player_name_map(self) -> dict[int, str]:
        """Build a mapping of player_id to display name."""
        players = self._player_repo.all()
        result: dict[int, str] = {}
        for p in players:
            if p.id is not None:
                result[p.id] = f"{p.name_first} {p.name_last}"
        return result
