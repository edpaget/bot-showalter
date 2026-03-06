"""Breakout/bust classification model.

Predicts P(breakout), P(bust), P(neutral) for each player using preseason
features and historical ADP-relative outcome labels.
"""

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from fantasy_baseball_manager.domain import LabeledSeason, OutcomeLabel, StatMetrics, SystemMetrics
from fantasy_baseball_manager.features import DatasetAssembler  # noqa: TC001 – needed at runtime by inspect.signature()
from fantasy_baseball_manager.models.breakout_bust.classification_backend import ClassificationTrainingBackend
from fantasy_baseball_manager.models.gbm_training import extract_features
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult, TrainResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.sampling import temporal_expanding_cv
from fantasy_baseball_manager.models.statcast_gbm.features import (
    build_batter_preseason_weighted_set,
    build_pitcher_preseason_averaged_set,
    preseason_averaged_pitcher_curated_columns,
    preseason_weighted_batter_curated_columns,
)
from fantasy_baseball_manager.models.statcast_gbm.serialization import load_models, save_models
from fantasy_baseball_manager.models.training_metadata import save_training_metadata, validate_no_leakage

logger = logging.getLogger(__name__)


class LabelSource(Protocol):
    def get_labels(self, season: int) -> list[LabeledSeason]: ...


LABEL_TO_INT: dict[OutcomeLabel, int] = {
    OutcomeLabel.NEUTRAL: 0,
    OutcomeLabel.BREAKOUT: 1,
    OutcomeLabel.BUST: 2,
}

INT_TO_LABEL: dict[int, OutcomeLabel] = {v: k for k, v in LABEL_TO_INT.items()}

_DEFAULT_FEATURE_COLUMNS: dict[str, list[str]] = {
    "batter": [*preseason_weighted_batter_curated_columns(), "adp_rank", "adp_pick"],
    "pitcher": [*preseason_averaged_pitcher_curated_columns(), "adp_rank", "adp_pick"],
}

_FEATURE_SET_BUILDERS: dict[str, Any] = {
    "batter": build_batter_preseason_weighted_set,
    "pitcher": build_pitcher_preseason_averaged_set,
}


def _join_labels_with_features(
    labels: list[LabeledSeason],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join labels with feature rows on (player_id, season).

    Inlined here so the models layer doesn't need to import services.
    """
    label_lookup: dict[tuple[int, int], LabeledSeason] = {(ls.player_id, ls.season): ls for ls in labels}
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("player_id"), row.get("season"))
        ls = label_lookup.get(key)  # type: ignore[arg-type]
        if ls is None:
            continue
        enriched = dict(row)
        enriched["label"] = ls.label
        enriched["adp_rank"] = ls.adp_rank
        enriched["adp_pick"] = ls.adp_pick
        result.append(enriched)
    return result


def _compute_base_rate_log_loss(y: list[int], n_classes: int = 3) -> float:
    """Compute log-loss of a classifier that always predicts class frequencies."""
    arr = np.array(y)
    n = len(arr)
    if n == 0:
        return 0.0
    proba = np.zeros((n, n_classes))
    for c in range(n_classes):
        proba[:, c] = np.sum(arr == c) / n
    return float(log_loss(arr, proba, labels=list(range(n_classes))))


def _compute_feature_importances(
    clf: HistGradientBoostingClassifier,
    X: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.integer[Any]]],
    feature_columns: list[str],
) -> list[tuple[str, float]]:
    """Compute permutation-based feature importances, sorted descending."""
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42, scoring="neg_log_loss")
    pairs = sorted(
        zip(feature_columns, result.importances_mean, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(name, float(imp)) for name, imp in pairs[:10]]


@register("breakout-bust")
class BreakoutBustModel:
    def __init__(self, assembler: DatasetAssembler, label_source: LabelSource) -> None:
        self._assembler = assembler
        self._label_source = label_source

    @property
    def name(self) -> str:
        return "breakout-bust"

    @property
    def description(self) -> str:
        return "GBM classifier predicting breakout/bust probability from preseason features"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"train", "predict", "evaluate"})

    @property
    def artifact_type(self) -> str:
        return "breakout-bust-classifier"

    def _collect_labels(self, seasons: list[int], player_type: str) -> list[LabeledSeason]:
        result: list[LabeledSeason] = []
        for season in seasons:
            result.extend(ls for ls in self._label_source.get_labels(season) if ls.player_type == player_type)
        return result

    def _get_feature_rows(self, seasons: list[int], player_type: str) -> list[dict[str, Any]]:
        builder = _FEATURE_SET_BUILDERS[player_type]
        feature_set = builder(seasons)
        handle = self._assembler.get_or_materialize(feature_set)
        return self._assembler.read(handle)

    def _build_joined_rows(
        self,
        seasons: list[int],
        player_type: str,
    ) -> list[dict[str, Any]]:
        labels = self._collect_labels(seasons, player_type)
        rows = self._get_feature_rows(seasons, player_type)
        return _join_labels_with_features(labels, rows)

    def _feature_columns(self, config: ModelConfig, player_type: str) -> list[str]:
        override = config.model_params.get("feature_columns", {}).get(player_type)
        if override is not None:
            return list(override)
        return _DEFAULT_FEATURE_COLUMNS[player_type]

    def experiment_player_types(self) -> list[str]:
        return ["batter", "pitcher"]

    def experiment_feature_columns(self, player_type: str) -> list[str]:
        return list(_DEFAULT_FEATURE_COLUMNS[player_type])

    def experiment_targets(self, player_type: str) -> list[str]:
        return ["p_breakout", "p_bust"]

    def experiment_training_data(self, player_type: str, seasons: list[int]) -> dict[int, list[dict[str, Any]]]:
        joined = self._build_joined_rows(seasons, player_type)
        rows_by_season: dict[int, list[dict[str, Any]]] = {}
        for row in joined:
            rows_by_season.setdefault(row["season"], []).append(row)
        return rows_by_season

    def experiment_training_backend(self) -> ClassificationTrainingBackend:
        return ClassificationTrainingBackend()

    def train(self, config: ModelConfig) -> TrainResult:
        seasons = sorted(config.seasons)
        if len(seasons) < 2:
            msg = f"Training requires at least 2 seasons (got {len(seasons)})"
            raise ValueError(msg)

        artifact_path = Path(config.artifacts_dir) / self.artifact_type
        artifact_path.mkdir(parents=True, exist_ok=True)
        save_training_metadata(artifact_path, seasons[:-1], [seasons[-1]])
        metrics: dict[str, float] = {}

        holdout_season = seasons[-1]

        for player_type in ("batter", "pitcher"):
            feature_columns = self._feature_columns(config, player_type)
            joined = self._build_joined_rows(seasons, player_type)

            if not joined:
                logger.warning("No joined rows for %s — skipping", player_type)
                continue

            train_rows = [r for r in joined if r["season"] != holdout_season]
            holdout_rows = [r for r in joined if r["season"] == holdout_season]

            X_train = extract_features(train_rows, feature_columns)
            y_train = [LABEL_TO_INT[r["label"]] for r in train_rows]

            clf = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=10,
            )
            clf.fit(X_train, y_train)

            # Compute feature importances on training data
            X_train_arr = np.array(X_train, dtype=np.float64)
            y_train_arr = np.array(y_train)
            top_features = _compute_feature_importances(clf, X_train_arr, y_train_arr, feature_columns)

            save_models(
                {player_type: clf, "feature_columns": feature_columns, "top_features": top_features},
                artifact_path / f"{player_type}_classifier.joblib",
            )

            # Score on holdout
            if holdout_rows:
                X_holdout = extract_features(holdout_rows, feature_columns)
                y_holdout = [LABEL_TO_INT[r["label"]] for r in holdout_rows]
                proba_holdout = clf.predict_proba(X_holdout)
                holdout_log_loss = float(log_loss(y_holdout, proba_holdout, labels=[0, 1, 2]))
                base_rate_ll = _compute_base_rate_log_loss(y_holdout)
                metrics[f"{player_type}_log_loss"] = holdout_log_loss
                metrics[f"{player_type}_base_rate_log_loss"] = base_rate_ll
                logger.info(
                    "%s holdout log-loss=%.4f base_rate=%.4f",
                    player_type,
                    holdout_log_loss,
                    base_rate_ll,
                )

            # Temporal expanding CV if enough seasons
            if len(seasons) >= 3:
                cv_losses: list[float] = []
                for cv_train_seasons, cv_test_season in temporal_expanding_cv(seasons):
                    cv_train = [r for r in joined if r["season"] in set(cv_train_seasons)]
                    cv_test = [r for r in joined if r["season"] == cv_test_season]
                    if not cv_train or not cv_test:
                        continue

                    X_cv_train = extract_features(cv_train, feature_columns)
                    y_cv_train = [LABEL_TO_INT[r["label"]] for r in cv_train]
                    X_cv_test = extract_features(cv_test, feature_columns)
                    y_cv_test = [LABEL_TO_INT[r["label"]] for r in cv_test]

                    cv_clf = HistGradientBoostingClassifier(
                        max_iter=200,
                        max_depth=5,
                        learning_rate=0.1,
                        min_samples_leaf=10,
                    )
                    cv_clf.fit(X_cv_train, y_cv_train)
                    cv_proba = cv_clf.predict_proba(X_cv_test)
                    cv_losses.append(float(log_loss(y_cv_test, cv_proba, labels=[0, 1, 2])))

                if cv_losses:
                    metrics[f"{player_type}_cv_log_loss"] = sum(cv_losses) / len(cv_losses)

        return TrainResult(
            model_name=self.name,
            metrics=metrics,
            artifacts_path=str(artifact_path),
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        artifact_path = Path(config.artifacts_dir) / self.artifact_type
        validate_no_leakage(artifact_path, config.seasons)
        seasons = sorted(config.seasons)
        predictions: list[dict[str, Any]] = []

        for player_type in ("batter", "pitcher"):
            clf_data = load_models(artifact_path / f"{player_type}_classifier.joblib")
            clf: HistGradientBoostingClassifier = clf_data[player_type]
            feature_columns: list[str] = clf_data["feature_columns"]
            top_features: list[tuple[str, float]] = clf_data["top_features"]

            joined = self._build_joined_rows(seasons, player_type)
            if not joined:
                continue

            X = extract_features(joined, feature_columns)
            proba = clf.predict_proba(X)

            # Map clf.classes_ to our label indices
            class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
            neutral_idx = class_to_idx.get(0, 0)
            breakout_idx = class_to_idx.get(1, 1)
            bust_idx = class_to_idx.get(2, 2)

            for i, row in enumerate(joined):
                predictions.append(
                    {
                        "player_id": row["player_id"],
                        "player_name": "",
                        "player_type": player_type,
                        "position": "",
                        "p_breakout": float(proba[i, breakout_idx]),
                        "p_bust": float(proba[i, bust_idx]),
                        "p_neutral": float(proba[i, neutral_idx]),
                        "top_features": top_features,
                    }
                )

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=str(artifact_path),
        )

    def evaluate(self, config: ModelConfig) -> SystemMetrics:
        """Walk-forward evaluation: train on all seasons except last, predict last, score."""
        seasons = sorted(config.seasons)
        if len(seasons) < 2:
            msg = f"Training requires at least 2 seasons (got {len(seasons)})"
            raise ValueError(msg)

        holdout_season = seasons[-1]

        all_y_holdout: list[int] = []
        all_proba: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []

        for player_type in ("batter", "pitcher"):
            feature_columns = self._feature_columns(config, player_type)
            joined = self._build_joined_rows(seasons, player_type)
            if not joined:
                continue

            train_rows = [r for r in joined if r["season"] != holdout_season]
            holdout_rows = [r for r in joined if r["season"] == holdout_season]
            if not train_rows or not holdout_rows:
                continue

            X_train = extract_features(train_rows, feature_columns)
            y_train = [LABEL_TO_INT[r["label"]] for r in train_rows]

            clf = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=10,
            )
            clf.fit(X_train, y_train)

            X_holdout = extract_features(holdout_rows, feature_columns)
            y_holdout = [LABEL_TO_INT[r["label"]] for r in holdout_rows]
            proba = clf.predict_proba(X_holdout)

            all_y_holdout.extend(y_holdout)
            all_proba.append(proba)

        n_evaluated = len(all_y_holdout)
        if n_evaluated > 0 and all_proba:
            combined_proba = np.vstack(all_proba)
            holdout_log_loss = float(log_loss(all_y_holdout, combined_proba, labels=[0, 1, 2]))
            base_rate_ll = _compute_base_rate_log_loss(all_y_holdout)
        else:
            holdout_log_loss = 0.0
            base_rate_ll = 0.0

        metrics: dict[str, StatMetrics] = {
            "log_loss": StatMetrics(
                rmse=holdout_log_loss,
                mae=0.0,
                correlation=0.0,
                rank_correlation=0.0,
                r_squared=0.0,
                mean_error=0.0,
                n=n_evaluated,
            ),
            "base_rate_log_loss": StatMetrics(
                rmse=base_rate_ll,
                mae=0.0,
                correlation=0.0,
                rank_correlation=0.0,
                r_squared=0.0,
                mean_error=0.0,
                n=n_evaluated,
            ),
        }

        return SystemMetrics(
            system=self.name,
            version=config.version or "",
            source_type="breakout-bust-classifier",
            metrics=metrics,
        )
