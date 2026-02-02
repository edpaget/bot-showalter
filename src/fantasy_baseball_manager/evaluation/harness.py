from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.evaluation.actuals import actuals_as_projections
from fantasy_baseball_manager.evaluation.metrics import mae, pearson_r, rmse, spearman_rho, top_n_precision
from fantasy_baseball_manager.evaluation.models import (
    EvaluationResult,
    RankAccuracy,
    SourceEvaluation,
    StatAccuracy,
)
from fantasy_baseball_manager.valuation.stat_extractors import extract_batting_stat, extract_pitching_stat
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
    from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
    from fantasy_baseball_manager.valuation.projection_source import ProjectionSource


@dataclass(frozen=True)
class EvaluationConfig:
    year: int
    batting_categories: tuple[StatCategory, ...]
    pitching_categories: tuple[StatCategory, ...]
    min_pa: int
    min_ip: float
    top_n: int


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


def evaluate_source(
    source: ProjectionSource,
    source_name: str,
    data_source: StatsDataSource,
    config: EvaluationConfig,
) -> SourceEvaluation:
    proj_batting = source.batting_projections()
    proj_pitching = source.pitching_projections()
    actual_batting, actual_pitching = actuals_as_projections(data_source, config.year, config.min_pa, config.min_ip)

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

    return SourceEvaluation(
        source_name=source_name,
        year=config.year,
        batting_stat_accuracy=batting_stat_accuracy,
        pitching_stat_accuracy=pitching_stat_accuracy,
        batting_rank_accuracy=batting_rank_accuracy,
        pitching_rank_accuracy=pitching_rank_accuracy,
    )


def evaluate(
    sources: list[tuple[str, ProjectionSource]],
    data_source: StatsDataSource,
    config: EvaluationConfig,
) -> EvaluationResult:
    evaluations = tuple(evaluate_source(source, name, data_source, config) for name, source in sources)
    return EvaluationResult(evaluations=evaluations)
