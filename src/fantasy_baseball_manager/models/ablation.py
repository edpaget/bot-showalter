"""Shared helpers for model evaluation and feature-ablation analysis."""

from dataclasses import dataclass
from typing import Any

from threadpoolctl import threadpool_limits

from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import FeatureSet
from fantasy_baseball_manager.models.gbm_training import (
    compute_cv_permutation_importance,
    compute_grouped_permutation_importance,
    extract_features,
    extract_targets,
    fit_models,
    identify_prune_candidates,
    validate_pruning,
)
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    Evaluator,
    ModelConfig,
    ValidationResult,
)


def evaluate_projections(
    evaluator: Evaluator | None,
    model_name: str,
    config: ModelConfig,
) -> SystemMetrics:
    """Delegate evaluation to an Evaluator — shared by all model types."""
    if evaluator is None:
        msg = "evaluate() requires an evaluator"
        raise TypeError(msg)
    version = config.version or "latest"
    season = config.seasons[0]
    return evaluator.evaluate(model_name, version, season, top=config.top)


@dataclass(frozen=True)
class PlayerTypeConfig:
    player_type: str  # "batter" or "pitcher"
    train_fs: FeatureSet
    columns: list[str]
    targets: list[str]
    params: dict[str, Any]


def single_holdout_importance(
    assembler: DatasetAssembler,
    train_fs: FeatureSet,
    columns: list[str],
    targets: list[str],
    params: dict[str, Any],
    train_seasons: list[int],
    holdout_seasons: list[int],
    n_repeats: int,
    correlation_threshold: float,
) -> Any:
    """Materialize, split, fit, then compute grouped permutation importance."""
    handle = assembler.get_or_materialize(train_fs)
    splits = assembler.split(handle, train=train_seasons, holdout=holdout_seasons)

    train_rows = assembler.read(splits.train)
    X_train = extract_features(train_rows, columns)
    y_train = extract_targets(train_rows, targets)
    models = fit_models(X_train, y_train, params)

    if splits.holdout is not None:
        holdout_rows = assembler.read(splits.holdout)
        X_holdout = extract_features(holdout_rows, columns)
        y_holdout = extract_targets(holdout_rows, targets)
        return compute_grouped_permutation_importance(
            models,
            X_holdout,
            y_holdout,
            columns,
            n_repeats=n_repeats,
            correlation_threshold=correlation_threshold,
        )
    return None


def multi_holdout_importance(
    assembler: DatasetAssembler,
    train_fs: FeatureSet,
    columns: list[str],
    targets: list[str],
    params: dict[str, Any],
    n_repeats: int,
    correlation_threshold: float,
) -> Any:
    """Materialize all rows, group by season, compute CV permutation importance."""
    handle = assembler.get_or_materialize(train_fs)
    all_rows = assembler.read(handle)

    rows_by_season: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        s = row["season"]
        rows_by_season.setdefault(s, []).append(row)

    return compute_cv_permutation_importance(
        rows_by_season,
        columns,
        targets,
        params,
        n_repeats=n_repeats,
        correlation_threshold=correlation_threshold,
    )


def run_ablation(
    assembler: DatasetAssembler,
    model_name: str,
    config: ModelConfig,
    player_configs: list[PlayerTypeConfig],
) -> AblationResult:
    """Full ablation loop shared by statcast-gbm and composite models."""
    with threadpool_limits(limits=1, user_api="openmp"):
        return _run_ablation_inner(assembler, model_name, config, player_configs)


def _run_ablation_inner(
    assembler: DatasetAssembler,
    model_name: str,
    config: ModelConfig,
    player_configs: list[PlayerTypeConfig],
) -> AblationResult:
    feature_impacts: dict[str, float] = {}
    feature_standard_errors: dict[str, float] = {}
    group_impacts: dict[str, float] = {}
    group_standard_errors: dict[str, float] = {}
    group_members: dict[str, list[str]] = {}
    validation_results: dict[str, ValidationResult] = {}

    n_repeats = int(config.model_params.get("n_repeats", 20))
    correlation_threshold = float(config.model_params.get("correlation_threshold", 0.70))
    do_validate = bool(config.model_params.get("validate", False))
    max_degradation_pct = float(config.model_params.get("max_degradation_pct", 5.0))
    multi_holdout = bool(config.model_params.get("multi_holdout", False))

    train_seasons = config.seasons[:-1]
    holdout_seasons = [config.seasons[-1]]

    for pc in player_configs:
        if multi_holdout:
            result = multi_holdout_importance(
                assembler,
                pc.train_fs,
                pc.columns,
                pc.targets,
                pc.params,
                n_repeats,
                correlation_threshold,
            )
        else:
            result = single_holdout_importance(
                assembler,
                pc.train_fs,
                pc.columns,
                pc.targets,
                pc.params,
                train_seasons,
                holdout_seasons,
                n_repeats,
                correlation_threshold,
            )

        if result is not None:
            for col, fi in result.feature_importance.items():
                feature_impacts[f"{pc.player_type}:{col}"] = fi.mean
                feature_standard_errors[f"{pc.player_type}:{col}"] = fi.se
            for g in result.groups:
                if len(g.members) > 1:
                    key = f"{pc.player_type}:{g.name}"
                    gi = result.group_importance[g.name]
                    group_impacts[key] = gi.mean
                    group_standard_errors[key] = gi.se
                    group_members[key] = [f"{pc.player_type}:{m}" for m in g.members]

            if do_validate:
                prune_set = identify_prune_candidates(result)
                if prune_set:
                    handle = assembler.get_or_materialize(pc.train_fs)
                    splits = assembler.split(handle, train=train_seasons, holdout=holdout_seasons)
                    val_train_rows = assembler.read(splits.train)
                    val_X_train = extract_features(val_train_rows, pc.columns)
                    val_y_train = extract_targets(val_train_rows, pc.targets)
                    full_models = fit_models(val_X_train, val_y_train, pc.params)
                    if splits.holdout is not None:
                        val_holdout_rows = assembler.read(splits.holdout)
                        validation_results[pc.player_type] = validate_pruning(
                            full_models=full_models,
                            train_rows=val_train_rows,
                            holdout_rows=val_holdout_rows,
                            feature_columns=pc.columns,
                            prune_set=prune_set,
                            targets=pc.targets,
                            model_params=pc.params,
                            player_type=pc.player_type,
                            max_degradation_pct=max_degradation_pct,
                        )

    return AblationResult(
        model_name=model_name,
        feature_impacts=feature_impacts,
        feature_standard_errors=feature_standard_errors,
        group_impacts=group_impacts,
        group_standard_errors=group_standard_errors,
        group_members=group_members,
        validation_results=validation_results,
    )
