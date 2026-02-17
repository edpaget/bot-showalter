"""Playing-time model — projects PA (batters) and IP (pitchers) via OLS regression."""

from pathlib import Path
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import AnyFeature, FeatureSet, SpineFilter
from fantasy_baseball_manager.models.sampling import temporal_holdout_split
from fantasy_baseball_manager.models.playing_time.aging import (
    enrich_rows_with_age_pt_factor,
    fit_playing_time_aging_curve,
)
from fantasy_baseball_manager.models.playing_time.convert import pt_projection_to_domain
from fantasy_baseball_manager.models.playing_time.engine import (
    ablation_study,
    coefficient_report,
    compute_residual_buckets,
    evaluate_holdout,
    fit_playing_time,
    predict_playing_time,
    predict_playing_time_distribution,
    select_alpha,
)
from fantasy_baseball_manager.models.playing_time.features import (
    batting_pt_feature_columns,
    build_batting_pt_derived_transforms,
    build_batting_pt_features,
    build_batting_pt_training_features,
    build_pitching_pt_derived_transforms,
    build_pitching_pt_features,
    build_pitching_pt_training_features,
    pitching_pt_feature_columns,
)
from fantasy_baseball_manager.models.playing_time.serialization import (
    load_aging_curves,
    load_coefficients,
    load_residual_buckets,
    save_aging_curves,
    save_coefficients,
    save_residual_buckets,
)
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register

_ARTIFACT_FILENAME = "pt_coefficients.joblib"
_AGING_CURVES_FILENAME = "pt_aging_curves.joblib"
_RESIDUAL_BUCKETS_FILENAME = "pt_residual_buckets.joblib"


@register("playing_time")
class PlayingTimeModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "playing_time"

    @property
    def description(self) -> str:
        return "Playing-time projection — projects PA (batters) and IP (pitchers) via OLS regression."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "predict", "ablate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.FILE.value

    def _build_feature_sets(
        self,
        seasons: list[int],
        *,
        training: bool = False,
        lags: int = 3,
    ) -> tuple[FeatureSet, FeatureSet]:
        if training:
            bat_features = build_batting_pt_training_features(lags)
            pitch_features = build_pitching_pt_training_features(lags)
        else:
            bat_features: list[AnyFeature] = list(build_batting_pt_features(lags))
            bat_features.extend(build_batting_pt_derived_transforms(lags))
            pitch_features: list[AnyFeature] = list(build_pitching_pt_features(lags))
            pitch_features.extend(build_pitching_pt_derived_transforms(lags))

        batting_fs = FeatureSet(
            name="playing_time_batting_train" if training else "playing_time_batting",
            features=tuple(bat_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_pa=50, player_type="batter"),
        )
        pitching_fs = FeatureSet(
            name="playing_time_pitching_train" if training else "playing_time_pitching",
            features=tuple(pitch_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_ip=10.0, player_type="pitcher"),
        )
        return batting_fs, pitching_fs

    def _artifact_path(self, config: ModelConfig) -> Path:
        return Path(config.artifacts_dir) / self.name / (config.version or "latest")

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        lags = config.model_params.get("lags", 3)
        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        assert self._assembler is not None, "assembler is required for train"
        lags = config.model_params.get("lags", 3)
        aging_min_samples: int = config.model_params.get("aging_min_samples", 30)
        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, training=True, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        # Fit aging curves from training data
        bat_curve = fit_playing_time_aging_curve(
            bat_rows,
            "batter",
            "target_pa",
            "pa_1",
            min_pt=50.0,
            min_samples=aging_min_samples,
        )
        pitch_curve = fit_playing_time_aging_curve(
            pitch_rows,
            "pitcher",
            "target_ip",
            "ip_1",
            min_pt=10.0,
            min_samples=aging_min_samples,
        )

        # Enrich rows with age_pt_factor
        bat_rows = enrich_rows_with_age_pt_factor(bat_rows, bat_curve)
        pitch_rows = enrich_rows_with_age_pt_factor(pitch_rows, pitch_curve)

        bat_columns = batting_pt_feature_columns(lags) + ["age_pt_factor"]
        pitch_columns = pitching_pt_feature_columns(lags) + ["age_pt_factor"]

        alpha_override: float | None = config.model_params.get("alpha", None)

        bat_alpha = (
            alpha_override if alpha_override is not None else select_alpha(bat_rows, bat_columns, "target_pa", "batter")
        )
        pitch_alpha = (
            alpha_override
            if alpha_override is not None
            else select_alpha(pitch_rows, pitch_columns, "target_ip", "pitcher")
        )

        bat_coeff = fit_playing_time(bat_rows, bat_columns, "target_pa", "batter", alpha=bat_alpha)
        pitch_coeff = fit_playing_time(pitch_rows, pitch_columns, "target_ip", "pitcher", alpha=pitch_alpha)

        artifact_path = self._artifact_path(config)
        artifact_path.mkdir(parents=True, exist_ok=True)
        save_coefficients(
            {"batter": bat_coeff, "pitcher": pitch_coeff},
            artifact_path / _ARTIFACT_FILENAME,
        )
        save_aging_curves(
            {"batter": bat_curve, "pitcher": pitch_curve},
            artifact_path / _AGING_CURVES_FILENAME,
        )

        bat_residuals = compute_residual_buckets(bat_rows, bat_coeff, "target_pa")
        pitch_residuals = compute_residual_buckets(pitch_rows, pitch_coeff, "target_ip")
        save_residual_buckets(
            {"batter": bat_residuals, "pitcher": pitch_residuals},
            artifact_path / _RESIDUAL_BUCKETS_FILENAME,
        )

        metrics: dict[str, float] = {
            "r_squared_batter": bat_coeff.r_squared,
            "r_squared_pitcher": pitch_coeff.r_squared,
            "bat_peak_age": bat_curve.peak_age,
            "pitch_peak_age": pitch_curve.peak_age,
            "alpha_batter": bat_alpha,
            "alpha_pitcher": pitch_alpha,
        }

        # Holdout evaluation when enough seasons are available
        if len(config.seasons) >= 4:
            bat_train, bat_holdout = temporal_holdout_split(bat_rows)
            pitch_train, pitch_holdout = temporal_holdout_split(pitch_rows)

            if bat_train and bat_holdout:
                bat_ho = evaluate_holdout(bat_train, bat_holdout, bat_columns, "target_pa", "batter", alpha=bat_alpha)
                metrics["rmse_batter_holdout"] = bat_ho["rmse"]
                metrics["r_squared_batter_holdout"] = bat_ho["r_squared"]

            if pitch_train and pitch_holdout:
                pitch_ho = evaluate_holdout(
                    pitch_train, pitch_holdout, pitch_columns, "target_ip", "pitcher", alpha=pitch_alpha
                )
                metrics["rmse_pitcher_holdout"] = pitch_ho["rmse"]
                metrics["r_squared_pitcher_holdout"] = pitch_ho["r_squared"]

        # Coefficient report — log feature counts
        bat_report = coefficient_report(bat_coeff)
        pitch_report = coefficient_report(pitch_coeff)
        # Subtract 1 for intercept entry
        metrics["n_batter_features"] = float(len(bat_report) - 1)
        metrics["n_pitcher_features"] = float(len(pitch_report) - 1)

        return TrainResult(
            model_name=self.name,
            metrics=metrics,
            artifacts_path=str(artifact_path),
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        assert self._assembler is not None, "assembler is required for predict"
        lags = config.model_params.get("lags", 3)

        artifact_path = self._artifact_path(config)
        coefficients = load_coefficients(artifact_path / _ARTIFACT_FILENAME)
        bat_coeff = coefficients["batter"]
        pitch_coeff = coefficients["pitcher"]

        aging_curves = load_aging_curves(artifact_path / _AGING_CURVES_FILENAME)

        residual_buckets_path = artifact_path / _RESIDUAL_BUCKETS_FILENAME
        residual_buckets = load_residual_buckets(residual_buckets_path) if residual_buckets_path.exists() else None

        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = enrich_rows_with_age_pt_factor(
            self._assembler.read(bat_handle),
            aging_curves["batter"],
        )
        pitch_rows = enrich_rows_with_age_pt_factor(
            self._assembler.read(pitch_handle),
            aging_curves["pitcher"],
        )

        if not config.seasons:
            msg = "config.seasons must not be empty"
            raise ValueError(msg)
        projected_season = max(config.seasons) + 1
        version = config.version or "latest"

        predictions: list[dict[str, Any]] = []
        all_distributions: list[dict[str, Any]] | None = [] if residual_buckets else None

        for row in bat_rows:
            pt = predict_playing_time(row, bat_coeff, clamp_min=0.0, clamp_max=750.0)
            domain = pt_projection_to_domain(
                row["player_id"],
                projected_season,
                pt,
                pitcher=False,
                version=version,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "batter",
                    **domain.stat_json,
                }
            )
            if residual_buckets and all_distributions is not None:
                dist = predict_playing_time_distribution(
                    pt,
                    row,
                    residual_buckets["batter"],
                    clamp_min=0.0,
                    clamp_max=750.0,
                )
                all_distributions.append(
                    {
                        "player_id": row["player_id"],
                        "player_type": "batter",
                        "season": projected_season,
                        "stat": dist.stat,
                        "p10": dist.p10,
                        "p25": dist.p25,
                        "p50": dist.p50,
                        "p75": dist.p75,
                        "p90": dist.p90,
                        "mean": dist.mean,
                        "std": dist.std,
                    }
                )

        for row in pitch_rows:
            pt = predict_playing_time(row, pitch_coeff, clamp_min=0.0, clamp_max=250.0)
            domain = pt_projection_to_domain(
                row["player_id"],
                projected_season,
                pt,
                pitcher=True,
                version=version,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "pitcher",
                    **domain.stat_json,
                }
            )
            if residual_buckets and all_distributions is not None:
                dist = predict_playing_time_distribution(
                    pt,
                    row,
                    residual_buckets["pitcher"],
                    clamp_min=0.0,
                    clamp_max=250.0,
                )
                all_distributions.append(
                    {
                        "player_id": row["player_id"],
                        "player_type": "pitcher",
                        "season": projected_season,
                        "stat": dist.stat,
                        "p10": dist.p10,
                        "p25": dist.p25,
                        "p50": dist.p50,
                        "p75": dist.p75,
                        "p90": dist.p90,
                        "mean": dist.mean,
                        "std": dist.std,
                    }
                )

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=config.output_dir or config.artifacts_dir,
            distributions=all_distributions if all_distributions else None,
        )

    def ablate(self, config: ModelConfig) -> AblationResult:
        assert self._assembler is not None, "assembler is required for ablate"
        lags = config.model_params.get("lags", 3)
        aging_min_samples: int = config.model_params.get("aging_min_samples", 30)

        if len(config.seasons) < 2:
            return AblationResult(model_name=self.name, feature_impacts={})

        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, training=True, lags=lags)
        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)
        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        # Fit aging curves and enrich
        bat_curve = fit_playing_time_aging_curve(
            bat_rows, "batter", "target_pa", "pa_1", min_pt=50.0, min_samples=aging_min_samples
        )
        pitch_curve = fit_playing_time_aging_curve(
            pitch_rows, "pitcher", "target_ip", "ip_1", min_pt=10.0, min_samples=aging_min_samples
        )
        bat_rows = enrich_rows_with_age_pt_factor(bat_rows, bat_curve)
        pitch_rows = enrich_rows_with_age_pt_factor(pitch_rows, pitch_curve)

        bat_train, bat_holdout = temporal_holdout_split(bat_rows)
        pitch_train, pitch_holdout = temporal_holdout_split(pitch_rows)

        alpha_override: float | None = config.model_params.get("alpha", None)

        batting_groups: list[tuple[str, list[str]]] = [
            ("base", ["age", "pa_1", "pa_2", "pa_3"]),
            ("war", ["war_1", "war_2"]),
            ("war_thresholds", ["war_above_2", "war_above_4", "war_below_0"]),
            ("interactions", ["war_trend", "pt_trend"]),
            ("aging", ["age_pt_factor"]),
            ("consensus_pt", ["consensus_pa"]),
        ]

        pitching_groups: list[tuple[str, list[str]]] = [
            ("base", ["age", "ip_1", "ip_2", "ip_3", "g_1", "g_2", "g_3", "gs_1"]),
            ("war", ["war_1", "war_2"]),
            ("war_thresholds", ["war_above_2", "war_above_4", "war_below_0"]),
            ("interactions", ["war_trend", "pt_trend"]),
            ("aging", ["age_pt_factor"]),
            ("starter_ratio", ["starter_ratio"]),
            ("consensus_pt", ["consensus_ip"]),
        ]

        feature_impacts: dict[str, float] = {}
        use_auto_alpha = alpha_override is None

        if bat_train and bat_holdout:
            bat_results = ablation_study(
                bat_train,
                bat_holdout,
                batting_groups,
                "target_pa",
                "batter",
                alpha=alpha_override or 0.0,
                auto_alpha=use_auto_alpha,
            )
            for i, entry in enumerate(bat_results):
                if i == 0:
                    delta = entry["rmse"]
                else:
                    delta = bat_results[i - 1]["rmse"] - entry["rmse"]
                feature_impacts[f"batter:{entry['group']}"] = delta

        if pitch_train and pitch_holdout:
            pitch_results = ablation_study(
                pitch_train,
                pitch_holdout,
                pitching_groups,
                "target_ip",
                "pitcher",
                alpha=alpha_override or 0.0,
                auto_alpha=use_auto_alpha,
            )
            for i, entry in enumerate(pitch_results):
                if i == 0:
                    delta = entry["rmse"]
                else:
                    delta = pitch_results[i - 1]["rmse"] - entry["rmse"]
                feature_impacts[f"pitcher:{entry['group']}"] = delta

        return AblationResult(model_name=self.name, feature_impacts=feature_impacts)
