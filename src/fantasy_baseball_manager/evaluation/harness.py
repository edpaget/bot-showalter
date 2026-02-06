from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.evaluation.actuals import actuals_as_projections
from fantasy_baseball_manager.evaluation.metrics import mae, pearson_r, rmse, spearman_rho, top_n_precision
from fantasy_baseball_manager.evaluation.models import (
    EvaluationResult,
    HeadToHeadResult,
    PlayerResidual,
    RankAccuracy,
    SourceEvaluation,
    StatAccuracy,
    StratumAccuracy,
)
from fantasy_baseball_manager.valuation.stat_extractors import extract_batting_stat, extract_pitching_stat
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        BattingSeasonStats,
        PitchingProjection,
        PitchingSeasonStats,
    )
    from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
    from fantasy_baseball_manager.valuation.projection_source import ProjectionSource


@dataclass(frozen=True)
class StratificationConfig:
    """Configuration for player segment stratification."""

    pa_buckets: tuple[tuple[int, int], ...] = ((200, 400), (400, 600), (600, 1500))
    ip_buckets: tuple[tuple[float, float], ...] = ((50.0, 100.0), (100.0, 150.0), (150.0, 300.0))
    age_buckets: tuple[tuple[int, int], ...] = ((21, 26), (27, 32), (33, 45))
    include_residuals: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    year: int
    batting_categories: tuple[StatCategory, ...]
    pitching_categories: tuple[StatCategory, ...]
    min_pa: int
    min_ip: float
    top_n: int
    stratification: StratificationConfig | None = None


def _compute_stat_accuracies(
    projected_values: list[list[float]],
    actual_values: list[list[float]],
    categories: tuple[StatCategory, ...],
    sample_size: int,
) -> tuple[StatAccuracy, ...]:
    accuracies: list[StatAccuracy] = []
    for i, cat in enumerate(categories):
        proj = projected_values[i]
        act = actual_values[i]
        r = pearson_r(proj, act) if sample_size >= 2 else 0.0
        accuracies.append(
            StatAccuracy(
                category=cat,
                sample_size=sample_size,
                rmse=rmse(proj, act),
                mae=mae(proj, act),
                correlation=r,
            )
        )
    return tuple(accuracies)


def _compute_rank_accuracy(
    projected_values: list[PlayerValue],
    actual_values: list[PlayerValue],
    top_n: int,
) -> RankAccuracy | None:
    sample_size = len(projected_values)
    if sample_size < 2:
        return None
    proj_totals = [pv.total_value for pv in projected_values]
    actual_totals = [pv.total_value for pv in actual_values]
    rho = spearman_rho(proj_totals, actual_totals)
    proj_sorted = sorted(projected_values, key=lambda pv: pv.total_value, reverse=True)
    actual_sorted = sorted(actual_values, key=lambda pv: pv.total_value, reverse=True)
    proj_ids = [pv.player_id for pv in proj_sorted]
    actual_ids = [pv.player_id for pv in actual_sorted]
    precision = top_n_precision(proj_ids, actual_ids, top_n)
    return RankAccuracy(
        sample_size=sample_size,
        spearman_rho=rho,
        top_n=top_n,
        top_n_precision=precision,
    )


def _bucket_players_batting(
    projections: list[BattingProjection],
    actuals: list[BattingProjection],
    buckets: tuple[tuple[int, int], ...],
    field: str,
) -> dict[str, tuple[list[BattingProjection], list[BattingProjection]]]:
    """Group matched batting players into buckets by actual field value."""
    result: dict[str, tuple[list[BattingProjection], list[BattingProjection]]] = {}
    for low, high in buckets:
        name = f"{field.upper()} {low}-{high}"
        bucket_proj: list[BattingProjection] = []
        bucket_act: list[BattingProjection] = []
        for p, a in zip(projections, actuals, strict=True):
            val = getattr(a, field)
            if low <= val < high:
                bucket_proj.append(p)
                bucket_act.append(a)
        result[name] = (bucket_proj, bucket_act)
    return result


def _bucket_players_pitching(
    projections: list[PitchingProjection],
    actuals: list[PitchingProjection],
    buckets: tuple[tuple[float, float], ...],
    field: str,
) -> dict[str, tuple[list[PitchingProjection], list[PitchingProjection]]]:
    """Group matched pitching players into buckets by actual field value."""
    result: dict[str, tuple[list[PitchingProjection], list[PitchingProjection]]] = {}
    for low, high in buckets:
        name = f"{field.upper()} {int(low)}-{int(high)}"
        bucket_proj: list[PitchingProjection] = []
        bucket_act: list[PitchingProjection] = []
        for p, a in zip(projections, actuals, strict=True):
            val = getattr(a, field)
            if low <= val < high:
                bucket_proj.append(p)
                bucket_act.append(a)
        result[name] = (bucket_proj, bucket_act)
    return result


def _compute_residuals[T: (BattingProjection, PitchingProjection)](
    projections: list[T],
    actuals: list[T],
    categories: tuple[StatCategory, ...],
    extractor: Callable[[T, StatCategory], float],
) -> tuple[PlayerResidual, ...]:
    """Compute per-player residuals for each category."""
    residuals: list[PlayerResidual] = []
    for proj, act in zip(projections, actuals, strict=True):
        for cat in categories:
            proj_val = extractor(proj, cat)
            act_val = extractor(act, cat)
            res = act_val - proj_val
            residuals.append(
                PlayerResidual(
                    player_id=proj.player_id,
                    player_name=proj.name,
                    category=cat,
                    projected=proj_val,
                    actual=act_val,
                    residual=res,
                    abs_residual=abs(res),
                )
            )
    return tuple(residuals)


def _compute_strata_batting(
    projections: list[BattingProjection],
    actuals: list[BattingProjection],
    categories: tuple[StatCategory, ...],
    strat_config: StratificationConfig,
    top_n: int,
) -> tuple[StratumAccuracy, ...]:
    """Compute accuracy metrics for each batting stratum."""
    strata: list[StratumAccuracy] = []

    # PA buckets
    pa_buckets = _bucket_players_batting(projections, actuals, strat_config.pa_buckets, "pa")
    for name, (bucket_proj, bucket_act) in pa_buckets.items():
        if len(bucket_proj) < 2:
            continue
        proj_vals = [[extract_batting_stat(b, cat) for b in bucket_proj] for cat in categories]
        act_vals = [[extract_batting_stat(b, cat) for b in bucket_act] for cat in categories]
        stat_acc = _compute_stat_accuracies(proj_vals, act_vals, categories, len(bucket_proj))
        rank_acc = None
        if categories:
            proj_pv = zscore_batting(bucket_proj, categories)
            act_pv = zscore_batting(bucket_act, categories)
            proj_by_id = {pv.player_id: pv for pv in proj_pv}
            act_by_id = {pv.player_id: pv for pv in act_pv}
            ordered_proj = [proj_by_id[b.player_id] for b in bucket_proj]
            ordered_act = [act_by_id[b.player_id] for b in bucket_proj]
            rank_acc = _compute_rank_accuracy(ordered_proj, ordered_act, top_n)
        strata.append(StratumAccuracy(name, len(bucket_proj), stat_acc, rank_acc))

    # Age buckets
    age_buckets = _bucket_players_batting(projections, actuals, strat_config.age_buckets, "age")
    for name, (bucket_proj, bucket_act) in age_buckets.items():
        if len(bucket_proj) < 2:
            continue
        proj_vals = [[extract_batting_stat(b, cat) for b in bucket_proj] for cat in categories]
        act_vals = [[extract_batting_stat(b, cat) for b in bucket_act] for cat in categories]
        stat_acc = _compute_stat_accuracies(proj_vals, act_vals, categories, len(bucket_proj))
        rank_acc = None
        if categories:
            proj_pv = zscore_batting(bucket_proj, categories)
            act_pv = zscore_batting(bucket_act, categories)
            proj_by_id = {pv.player_id: pv for pv in proj_pv}
            act_by_id = {pv.player_id: pv for pv in act_pv}
            ordered_proj = [proj_by_id[b.player_id] for b in bucket_proj]
            ordered_act = [act_by_id[b.player_id] for b in bucket_proj]
            rank_acc = _compute_rank_accuracy(ordered_proj, ordered_act, top_n)
        strata.append(StratumAccuracy(name, len(bucket_proj), stat_acc, rank_acc))

    return tuple(strata)


def _compute_strata_pitching(
    projections: list[PitchingProjection],
    actuals: list[PitchingProjection],
    categories: tuple[StatCategory, ...],
    strat_config: StratificationConfig,
    top_n: int,
) -> tuple[StratumAccuracy, ...]:
    """Compute accuracy metrics for each pitching stratum."""
    strata: list[StratumAccuracy] = []

    # IP buckets
    ip_buckets = _bucket_players_pitching(projections, actuals, strat_config.ip_buckets, "ip")
    for name, (bucket_proj, bucket_act) in ip_buckets.items():
        if len(bucket_proj) < 2:
            continue
        proj_vals = [[extract_pitching_stat(p, cat) for p in bucket_proj] for cat in categories]
        act_vals = [[extract_pitching_stat(p, cat) for p in bucket_act] for cat in categories]
        stat_acc = _compute_stat_accuracies(proj_vals, act_vals, categories, len(bucket_proj))
        rank_acc = None
        if categories:
            proj_pv = zscore_pitching(bucket_proj, categories)
            act_pv = zscore_pitching(bucket_act, categories)
            proj_by_id = {pv.player_id: pv for pv in proj_pv}
            act_by_id = {pv.player_id: pv for pv in act_pv}
            ordered_proj = [proj_by_id[p.player_id] for p in bucket_proj]
            ordered_act = [act_by_id[p.player_id] for p in bucket_proj]
            rank_acc = _compute_rank_accuracy(ordered_proj, ordered_act, top_n)
        strata.append(StratumAccuracy(name, len(bucket_proj), stat_acc, rank_acc))

    # Age buckets
    age_bucket_floats = tuple((float(low), float(high)) for low, high in strat_config.age_buckets)
    age_buckets = _bucket_players_pitching(projections, actuals, age_bucket_floats, "age")
    for name, (bucket_proj, bucket_act) in age_buckets.items():
        if len(bucket_proj) < 2:
            continue
        proj_vals = [[extract_pitching_stat(p, cat) for p in bucket_proj] for cat in categories]
        act_vals = [[extract_pitching_stat(p, cat) for p in bucket_act] for cat in categories]
        stat_acc = _compute_stat_accuracies(proj_vals, act_vals, categories, len(bucket_proj))
        rank_acc = None
        if categories:
            proj_pv = zscore_pitching(bucket_proj, categories)
            act_pv = zscore_pitching(bucket_act, categories)
            proj_by_id = {pv.player_id: pv for pv in proj_pv}
            act_by_id = {pv.player_id: pv for pv in act_pv}
            ordered_proj = [proj_by_id[p.player_id] for p in bucket_proj]
            ordered_act = [act_by_id[p.player_id] for p in bucket_proj]
            rank_acc = _compute_rank_accuracy(ordered_proj, ordered_act, top_n)
        strata.append(StratumAccuracy(name, len(bucket_proj), stat_acc, rank_acc))

    return tuple(strata)


def evaluate_source(
    source: ProjectionSource,
    source_name: str,
    batting_source: DataSource[BattingSeasonStats],
    pitching_source: DataSource[PitchingSeasonStats],
    config: EvaluationConfig,
) -> SourceEvaluation:
    proj_batting = source.batting_projections()
    proj_pitching = source.pitching_projections()
    actual_batting, actual_pitching = actuals_as_projections(batting_source, pitching_source, config.year, config.min_pa, config.min_ip)

    # Inner join batting by player_id
    actual_batting_map = {b.player_id: b for b in actual_batting}
    matched_proj_batting: list[BattingProjection] = []
    matched_act_batting: list[BattingProjection] = []
    for pb in proj_batting:
        if pb.player_id in actual_batting_map:
            matched_proj_batting.append(pb)
            matched_act_batting.append(actual_batting_map[pb.player_id])

    # Inner join pitching by player_id
    actual_pitching_map = {p.player_id: p for p in actual_pitching}
    matched_proj_pitching: list[PitchingProjection] = []
    matched_act_pitching: list[PitchingProjection] = []
    for pp in proj_pitching:
        if pp.player_id in actual_pitching_map:
            matched_proj_pitching.append(pp)
            matched_act_pitching.append(actual_pitching_map[pp.player_id])

    # Batting stat accuracy
    batting_stat_accuracy: tuple[StatAccuracy, ...] = ()
    batting_rank_accuracy: RankAccuracy | None = None
    bat_n = len(matched_proj_batting)
    if bat_n > 0:
        proj_bat_vals = [
            [extract_batting_stat(b, cat) for b in matched_proj_batting] for cat in config.batting_categories
        ]
        act_bat_vals = [
            [extract_batting_stat(b, cat) for b in matched_act_batting] for cat in config.batting_categories
        ]
        batting_stat_accuracy = _compute_stat_accuracies(proj_bat_vals, act_bat_vals, config.batting_categories, bat_n)

    # Batting rank accuracy
    if bat_n >= 2 and config.batting_categories:
        proj_bat_player_values = zscore_batting(matched_proj_batting, config.batting_categories)
        act_bat_player_values = zscore_batting(matched_act_batting, config.batting_categories)
        # Ensure same ordering by player_id for spearman
        proj_by_id = {pv.player_id: pv for pv in proj_bat_player_values}
        act_by_id = {pv.player_id: pv for pv in act_bat_player_values}
        ordered_ids = [b.player_id for b in matched_proj_batting]
        ordered_proj = [proj_by_id[pid] for pid in ordered_ids]
        ordered_act = [act_by_id[pid] for pid in ordered_ids]
        batting_rank_accuracy = _compute_rank_accuracy(ordered_proj, ordered_act, config.top_n)

    # Pitching stat accuracy
    pitching_stat_accuracy: tuple[StatAccuracy, ...] = ()
    pitching_rank_accuracy: RankAccuracy | None = None
    pitch_n = len(matched_proj_pitching)
    if pitch_n > 0:
        proj_pitch_vals = [
            [extract_pitching_stat(p, cat) for p in matched_proj_pitching] for cat in config.pitching_categories
        ]
        act_pitch_vals = [
            [extract_pitching_stat(p, cat) for p in matched_act_pitching] for cat in config.pitching_categories
        ]
        pitching_stat_accuracy = _compute_stat_accuracies(
            proj_pitch_vals, act_pitch_vals, config.pitching_categories, pitch_n
        )

    # Pitching rank accuracy
    if pitch_n >= 2 and config.pitching_categories:
        proj_pitch_player_values = zscore_pitching(matched_proj_pitching, config.pitching_categories)
        act_pitch_player_values = zscore_pitching(matched_act_pitching, config.pitching_categories)
        proj_by_id = {pv.player_id: pv for pv in proj_pitch_player_values}
        act_by_id = {pv.player_id: pv for pv in act_pitch_player_values}
        ordered_ids = [p.player_id for p in matched_proj_pitching]
        ordered_proj = [proj_by_id[pid] for pid in ordered_ids]
        ordered_act = [act_by_id[pid] for pid in ordered_ids]
        pitching_rank_accuracy = _compute_rank_accuracy(ordered_proj, ordered_act, config.top_n)

    # Compute stratification if configured
    batting_strata: tuple[StratumAccuracy, ...] = ()
    pitching_strata: tuple[StratumAccuracy, ...] = ()
    batting_residuals: tuple[PlayerResidual, ...] | None = None
    pitching_residuals: tuple[PlayerResidual, ...] | None = None

    if config.stratification is not None:
        if bat_n >= 2:
            batting_strata = _compute_strata_batting(
                matched_proj_batting,
                matched_act_batting,
                config.batting_categories,
                config.stratification,
                config.top_n,
            )
        if pitch_n >= 2:
            pitching_strata = _compute_strata_pitching(
                matched_proj_pitching,
                matched_act_pitching,
                config.pitching_categories,
                config.stratification,
                config.top_n,
            )
        if config.stratification.include_residuals:
            if bat_n > 0:
                batting_residuals = _compute_residuals(
                    matched_proj_batting,
                    matched_act_batting,
                    config.batting_categories,
                    extract_batting_stat,
                )
            if pitch_n > 0:
                pitching_residuals = _compute_residuals(
                    matched_proj_pitching,
                    matched_act_pitching,
                    config.pitching_categories,
                    extract_pitching_stat,
                )

    return SourceEvaluation(
        source_name=source_name,
        year=config.year,
        batting_stat_accuracy=batting_stat_accuracy,
        pitching_stat_accuracy=pitching_stat_accuracy,
        batting_rank_accuracy=batting_rank_accuracy,
        pitching_rank_accuracy=pitching_rank_accuracy,
        batting_strata=batting_strata,
        pitching_strata=pitching_strata,
        batting_residuals=batting_residuals,
        pitching_residuals=pitching_residuals,
    )


def evaluate(
    sources: list[tuple[str, ProjectionSource]],
    batting_source: DataSource[BattingSeasonStats],
    pitching_source: DataSource[PitchingSeasonStats],
    config: EvaluationConfig,
) -> EvaluationResult:
    evaluations = tuple(evaluate_source(source, name, batting_source, pitching_source, config) for name, source in sources)
    return EvaluationResult(evaluations=evaluations)


def compare_sources(
    eval_a: SourceEvaluation,
    eval_b: SourceEvaluation,
) -> tuple[HeadToHeadResult, ...]:
    """Compare two evaluations player-by-player using residuals."""
    no_batting = eval_a.batting_residuals is None or eval_b.batting_residuals is None
    no_pitching = eval_a.pitching_residuals is None or eval_b.pitching_residuals is None
    if no_batting and no_pitching:
        raise ValueError("Residuals required for head-to-head. Use include_residuals=True.")

    results: list[HeadToHeadResult] = []

    # Compare batting residuals
    if eval_a.batting_residuals is not None and eval_b.batting_residuals is not None:
        a_by_cat: dict[StatCategory, dict[str, PlayerResidual]] = {}
        for r in eval_a.batting_residuals:
            a_by_cat.setdefault(r.category, {})[r.player_id] = r

        b_by_cat: dict[StatCategory, dict[str, PlayerResidual]] = {}
        for r in eval_b.batting_residuals:
            b_by_cat.setdefault(r.category, {})[r.player_id] = r

        for cat in a_by_cat:
            a_resid = a_by_cat[cat]
            b_resid = b_by_cat.get(cat, {})
            common_ids = set(a_resid) & set(b_resid)

            a_wins = b_wins = ties = 0
            improvements: list[float] = []
            for pid in common_ids:
                a_err = a_resid[pid].abs_residual
                b_err = b_resid[pid].abs_residual
                if a_err < b_err - 0.001:
                    a_wins += 1
                    improvements.append(b_err - a_err)
                elif b_err < a_err - 0.001:
                    b_wins += 1
                else:
                    ties += 1

            n = len(common_ids)
            results.append(
                HeadToHeadResult(
                    source_a=eval_a.source_name,
                    source_b=eval_b.source_name,
                    category=cat,
                    sample_size=n,
                    a_wins=a_wins,
                    b_wins=b_wins,
                    ties=ties,
                    a_win_pct=a_wins / n if n > 0 else 0.0,
                    mean_improvement=sum(improvements) / len(improvements) if improvements else 0.0,
                )
            )

    # Compare pitching residuals
    if eval_a.pitching_residuals is not None and eval_b.pitching_residuals is not None:
        a_by_cat: dict[StatCategory, dict[str, PlayerResidual]] = {}
        for r in eval_a.pitching_residuals:
            a_by_cat.setdefault(r.category, {})[r.player_id] = r

        b_by_cat: dict[StatCategory, dict[str, PlayerResidual]] = {}
        for r in eval_b.pitching_residuals:
            b_by_cat.setdefault(r.category, {})[r.player_id] = r

        for cat in a_by_cat:
            a_resid = a_by_cat[cat]
            b_resid = b_by_cat.get(cat, {})
            common_ids = set(a_resid) & set(b_resid)

            a_wins = b_wins = ties = 0
            improvements: list[float] = []
            for pid in common_ids:
                a_err = a_resid[pid].abs_residual
                b_err = b_resid[pid].abs_residual
                if a_err < b_err - 0.001:
                    a_wins += 1
                    improvements.append(b_err - a_err)
                elif b_err < a_err - 0.001:
                    b_wins += 1
                else:
                    ties += 1

            n = len(common_ids)
            results.append(
                HeadToHeadResult(
                    source_a=eval_a.source_name,
                    source_b=eval_b.source_name,
                    category=cat,
                    sample_size=n,
                    a_wins=a_wins,
                    b_wins=b_wins,
                    ties=ties,
                    a_win_pct=a_wins / n if n > 0 else 0.0,
                    mean_improvement=sum(improvements) / len(improvements) if improvements else 0.0,
                )
            )

    return tuple(results)
