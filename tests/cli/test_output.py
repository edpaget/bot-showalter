from typing import TYPE_CHECKING, Any

from rich.console import Console

from fantasy_baseball_manager.cli import _output
from fantasy_baseball_manager.cli._output import (
    diff_records,
    print_ablation_result,
    print_adjusted_rankings,
    print_adp_accuracy_report,
    print_adp_movers_report,
    print_batch_simulation_result,
    print_bin_target_means,
    print_breakout_candidates,
    print_candidate_values,
    print_cascade_result,
    print_category_needs,
    print_checkpoint_detail,
    print_checkpoint_list,
    print_classifier_evaluation,
    print_cohort_bias_report,
    print_cohort_bias_summary,
    print_compare_features_result,
    print_comparison_result,
    print_coverage_matrix,
    print_dataset_list,
    print_draft_board,
    print_draft_report,
    print_draft_tiers,
    print_error,
    print_error_decomposition_report,
    print_experiment_detail,
    print_experiment_search_results,
    print_experiment_summary,
    print_feature_gap_report,
    print_features,
    print_gate_result,
    print_import_result,
    print_ingest_result,
    print_keeper_decisions,
    print_keeper_scenarios,
    print_keeper_solution,
    print_keeper_trade_impact,
    print_opportunity_costs,
    print_performance_report,
    print_pick_trade_evaluation,
    print_player_projections,
    print_player_valuations,
    print_position_check,
    print_predict_result,
    print_preflight_result,
    print_prepare_result,
    print_projection_confidence,
    print_regression_check_result,
    print_residual_analysis_report,
    print_residual_persistence_report,
    print_routing_table,
    print_run_detail,
    print_run_diff,
    print_run_inspect,
    print_run_list,
    print_scarcity_rankings,
    print_scarcity_report,
    print_stratified_comparison_result,
    print_system_disagreements,
    print_system_metrics,
    print_system_summaries,
    print_talent_delta_report,
    print_talent_quality_report,
    print_trade_evaluation,
    print_train_result,
    print_tune_result,
    print_upgrades,
    print_validation_result,
    print_valuation_eval_result,
    print_valuation_rankings,
    print_value_curve,
    print_value_over_adp,
    print_variance_targets,
)
from fantasy_baseball_manager.domain import (
    BreakoutPrediction,
    ClassifierCalibrationBin,
    ClassifierEvaluation,
    KeeperDecision,
    KeeperScenario,
    KeeperSet,
    KeeperSolution,
    KeeperTradeImpact,
    LiftResult,
    SensitivityEntry,
    ThresholdMetrics,
)
from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.adp_accuracy import ADPAccuracyReport, ADPAccuracyResult, SystemAccuracyResult
from fantasy_baseball_manager.domain.adp_movers import ADPMover, ADPMoversReport
from fantasy_baseball_manager.domain.adp_report import ValueOverADP, ValueOverADPReport
from fantasy_baseball_manager.domain.category_tracker import CategoryNeed, PlayerRecommendation
from fantasy_baseball_manager.domain.checkpoint import FeatureCheckpoint
from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.draft_report import CategoryStanding, DraftReport, PickGrade, StealOrReach
from fantasy_baseball_manager.domain.error_decomposition import (
    CohortBias,
    CohortBiasReport,
    DistinguishingFeature,
    ErrorDecompositionReport,
    FeatureGap,
    FeatureGapReport,
    MissPopulationSummary,
    PlayerResidual,
)
from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    RegressionCheckResult,
    StatMetrics,
    StratifiedComparisonResult,
    SystemMetrics,
    TailAccuracy,
)
from fantasy_baseball_manager.domain.experiment import (
    Experiment,
    ExplorationSummary,
    FeatureExplorationResult,
    TargetExplorationResult,
    TargetResult,
)
from fantasy_baseball_manager.domain.feature_candidate import BinTargetMean, CandidateValue
from fantasy_baseball_manager.domain.keeper import AdjustedValuation, TradeEvaluation, TradePlayerDetail
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.mock_draft import (
    BatchSimulationResult,
    DraftPick,
    SimulationSummary,
)
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
from fantasy_baseball_manager.domain.pick_value import (
    CascadeResult,
    CascadeRoster,
    PickTrade,
    PickTradeEvaluation,
)
from fantasy_baseball_manager.domain.pick_value import (
    PickValue as DomainPickValue,
)
from fantasy_baseball_manager.domain.positional_scarcity import (
    PositionScarcity,
    PositionValueCurve,
    ScarcityAdjustedPlayer,
)
from fantasy_baseball_manager.domain.positional_upgrade import (
    MarginalValue,
    OpportunityCost,
    PositionUpgrade,
)
from fantasy_baseball_manager.domain.projection import PlayerProjection, Projection, SystemSummary
from fantasy_baseball_manager.domain.projection_confidence import (
    ClassifiedPlayer,
    ConfidenceReport,
    PlayerConfidence,
    StatSpread,
    VarianceClassification,
)
from fantasy_baseball_manager.domain.residual_analysis import (
    CalibrationBin,
    ResidualAnalysisReport,
    ResidualAnalysisSummary,
    StatResidualAnalysis,
)
from fantasy_baseball_manager.domain.residual_persistence import (
    ChronicPerformer,
    ResidualPersistenceReport,
    ResidualPersistenceSummary,
    StatResidualPersistence,
)
from fantasy_baseball_manager.domain.talent_quality import (
    StatTalentMetrics,
    TalentQualitySummary,
    TrueTalentQualityReport,
)
from fantasy_baseball_manager.domain.tier import PlayerTier
from fantasy_baseball_manager.domain.valuation import PlayerValuation, ValuationAccuracy, ValuationEvalResult
from fantasy_baseball_manager.features.types import (
    DeltaFeature,
    DerivedTransformFeature,
    Feature,
    Source,
    TransformFeature,
)
from fantasy_baseball_manager.models.gbm_training import PerTargetBest
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    PredictResult,
    PrepareResult,
    TargetComparison,
    TrainResult,
    TuneResult,
    ValidationResult,
)
from fantasy_baseball_manager.services.dataset_catalog import DatasetInfo
from fantasy_baseball_manager.services.quick_eval import FeatureSetComparisonResult, TargetDelta
from fantasy_baseball_manager.services.regression_gate import GateResult, GateSegmentResult
from fantasy_baseball_manager.services.validation_gate import (
    PreflightResult,
    TargetPreflightDetail,
    ValidationSegmentResult,
)
from fantasy_baseball_manager.services.validation_gate import ValidationResult as GateValidationResult

if TYPE_CHECKING:
    import pytest


def _plain_feature() -> Feature:
    return Feature(name="hr_1", source=Source.BATTING, column="hr", lag=0)


def _lag_feature() -> Feature:
    return Feature(name="hr_prev", source=Source.BATTING, column="hr", lag=1)


def _system_feature() -> Feature:
    return Feature(name="proj_hr", source=Source.PROJECTION, column="hr", system="steamer")


def _computed_feature() -> Feature:
    return Feature(name="age", source=Source.PLAYER, column="", computed="age")


def _delta_feature() -> DeltaFeature:
    left = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
    right = Feature(name="hr_2", source=Source.BATTING, column="hr", lag=2)
    return DeltaFeature(name="hr_diff", left=left, right=right)


def _transform_feature() -> TransformFeature:
    def dummy(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    return TransformFeature(
        name="rate_calc",
        source=Source.STATCAST,
        columns=("ev", "la"),
        group_by=("player_id", "season"),
        transform=dummy,
        outputs=("barrel_pct", "hard_hit_pct"),
    )


class TestPrintError:
    def test_print_error_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error("something went wrong")
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "something went wrong" in captured.err
        assert captured.out == ""


class TestPrintFeatures:
    def test_print_features_plain(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_plain_feature(),))
        captured = capsys.readouterr()
        assert "batting.hr" in captured.out
        assert "1 features" in captured.out

    def test_print_features_lag(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_lag_feature(),))
        captured = capsys.readouterr()
        assert "lag=1" in captured.out

    def test_print_features_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_system_feature(),))
        captured = capsys.readouterr()
        assert "system=steamer" in captured.out

    def test_print_features_computed(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_computed_feature(),))
        captured = capsys.readouterr()
        assert "computed=age" in captured.out

    def test_print_features_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_delta_feature(),))
        captured = capsys.readouterr()
        assert "delta(hr_1 - hr_2)" in captured.out

    def test_print_features_transform(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_transform_feature(),))
        captured = capsys.readouterr()
        assert "transform" in captured.out
        assert "barrel_pct" in captured.out
        assert "hard_hit_pct" in captured.out

    def test_print_features_mixed(self, capsys: pytest.CaptureFixture[str]) -> None:
        features = (
            _plain_feature(),
            _lag_feature(),
            _system_feature(),
            _computed_feature(),
            _delta_feature(),
            _transform_feature(),
        )
        print_features("test_model", features)
        captured = capsys.readouterr()
        assert "6 features" in captured.out
        assert "batting.hr" in captured.out
        assert "lag=1" in captured.out
        assert "system=steamer" in captured.out
        assert "computed=age" in captured.out
        assert "delta(hr_1 - hr_2)" in captured.out
        assert "barrel_pct, hard_hit_pct" in captured.out


def _make_player_projection(
    stats: dict[str, Any],
    system: str = "steamer",
    version: str = "2025.1",
    source_type: str = "third_party",
    player_type: str = "batter",
) -> PlayerProjection:
    return PlayerProjection(
        player_name="Mike Trout",
        system=system,
        version=version,
        source_type=source_type,
        player_type=player_type,
        stats=stats,
    )


class TestPrintPlayerProjectionsLineage:
    def test_print_player_projections_ensemble_lineage(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "rbi": 90.0,
                "_components": {"marcel": 0.6, "steamer": 0.4},
                "_mode": "weighted_average",
            },
            system="ensemble",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "Sources:" in captured.out
        assert "marcel 60%" in captured.out
        assert "steamer 40%" in captured.out
        assert "weighted_average" in captured.out

    def test_print_player_projections_composite_lineage(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "_pt_system": "playing_time",
            },
            system="composite",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "PT source:" in captured.out
        assert "playing_time" in captured.out

    def test_print_player_projections_hides_metadata_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "avg": 0.280,
                "_components": {"marcel": 0.6, "steamer": 0.4},
                "_mode": "weighted_average",
                "_pt_system": "playing_time",
                "rates": {"hr_rate": 0.05},
            },
            system="ensemble",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "_components" not in captured.out
        assert "_mode" not in captured.out
        assert "_pt_system" not in captured.out
        assert "rates" not in captured.out
        assert "hr" in captured.out
        assert "avg" in captured.out

    def test_print_player_projections_plain_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={"hr": 30.0, "avg": 0.280},
            system="steamer",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "Sources:" not in captured.out
        assert "PT source:" not in captured.out


def _make_player_valuation(
    player_name: str = "Juan Soto",
    system: str = "zar",
    version: str = "1.0",
    projection_system: str = "steamer",
    projection_version: str = "2025.1",
    player_type: str = "batter",
    position: str = "OF",
    value: float = 42.5,
    rank: int = 1,
    category_scores: dict[str, float] | None = None,
) -> PlayerValuation:
    return PlayerValuation(
        player_name=player_name,
        system=system,
        version=version,
        projection_system=projection_system,
        projection_version=projection_version,
        player_type=player_type,
        position=position,
        value=value,
        rank=rank,
        category_scores=category_scores or {"hr": 2.1, "sb": 0.5},
    )


class TestPrintPlayerValuations:
    def test_empty_valuations(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_player_valuations([])
        captured = capsys.readouterr()
        assert "No valuations found" in captured.out

    def test_single_valuation_shows_breakdown(self, capsys: pytest.CaptureFixture[str]) -> None:
        val = _make_player_valuation()
        print_player_valuations([val])
        captured = capsys.readouterr()
        assert "Juan Soto" in captured.out
        assert "zar" in captured.out
        assert "42.5" in captured.out
        assert "hr" in captured.out
        assert "2.1" in captured.out


class TestPrintValuationRankings:
    def test_empty_rankings(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_valuation_rankings([])
        captured = capsys.readouterr()
        assert "No valuations found" in captured.out

    def test_rankings_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        vals = [
            _make_player_valuation(player_name="Juan Soto", rank=1, value=42.5),
            _make_player_valuation(player_name="Aaron Judge", rank=2, value=38.0),
        ]
        print_valuation_rankings(vals)
        captured = capsys.readouterr()
        assert "Juan Soto" in captured.out
        assert "Aaron Judge" in captured.out
        assert "42.5" in captured.out
        assert "1" in captured.out


def _make_eval_result(
    n: int = 2,
    players: list[ValuationAccuracy] | None = None,
) -> ValuationEvalResult:
    if players is None and n > 0:
        players = [
            ValuationAccuracy(
                player_id=1,
                player_name="Mike Trout",
                player_type="batter",
                position="of",
                predicted_value=40.0,
                actual_value=35.0,
                surplus=5.0,
                predicted_rank=1,
                actual_rank=2,
            ),
            ValuationAccuracy(
                player_id=2,
                player_name="Aaron Judge",
                player_type="batter",
                position="util",
                predicted_value=30.0,
                actual_value=38.0,
                surplus=-8.0,
                predicted_rank=2,
                actual_rank=1,
            ),
        ]
    return ValuationEvalResult(
        system="zar",
        version="1.0",
        season=2025,
        value_mae=6.5,
        rank_correlation=0.85,
        n=n,
        players=players or [],
    )


class TestPrintValuationEvalResult:
    def test_empty_eval_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_eval_result(n=0, players=[])
        print_valuation_eval_result(result)
        captured = capsys.readouterr()
        assert "No matched players found" in captured.out

    def test_eval_result_shows_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_eval_result()
        print_valuation_eval_result(result)
        captured = capsys.readouterr()
        assert "6.5" in captured.out
        assert "0.85" in captured.out

    def test_eval_result_shows_player_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_eval_result()
        print_valuation_eval_result(result)
        captured = capsys.readouterr()
        assert "Mike Trout" in captured.out
        assert "Aaron Judge" in captured.out
        assert "40.0" in captured.out
        assert "35.0" in captured.out


def _make_stat_metrics(
    rmse: float = 0.05,
    mae: float = 0.04,
    correlation: float = 0.9,
    rank_correlation: float = 0.9,
    r_squared: float = 0.75,
    mean_error: float = 0.0,
    n: int = 100,
) -> StatMetrics:
    return StatMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        rank_correlation=rank_correlation,
        r_squared=r_squared,
        mean_error=mean_error,
        n=n,
    )


def _make_system_metrics(
    system: str = "steamer",
    version: str = "2025",
    source_type: str = "third_party",
    stats: dict[str, StatMetrics] | None = None,
) -> SystemMetrics:
    if stats is None:
        stats = {"hr": _make_stat_metrics(), "avg": _make_stat_metrics(r_squared=0.65)}
    return SystemMetrics(system=system, version=version, source_type=source_type, metrics=stats)


class TestPrintSystemMetrics:
    def test_r_squared_header_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_metrics(_make_system_metrics())
        captured = capsys.readouterr()
        assert "R\u00b2" in captured.out

    def test_r_squared_value_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_metrics(_make_system_metrics())
        captured = capsys.readouterr()
        assert "0.7500" in captured.out
        assert "0.6500" in captured.out

    def test_system_metrics_shows_rho(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_metrics(_make_system_metrics())
        captured = capsys.readouterr()
        assert "\u03c1" in captured.out

    def test_system_metrics_shows_rank_correlation_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_metrics(_make_system_metrics(stats={"hr": _make_stat_metrics(rank_correlation=0.85)}))
        captured = capsys.readouterr()
        assert "0.8500" in captured.out

    def test_bias_column_in_system_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        metrics = _make_stat_metrics()
        print_system_metrics(_make_system_metrics(stats={"hr": metrics}))
        captured = capsys.readouterr()
        assert "Bias" in captured.out
        assert "+0.0000" in captured.out


class TestPrintComparisonResult:
    def test_comparison_shows_r_squared(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(r_squared=0.60)})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics(r_squared=0.45)})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "R\u00b2" in captured.out
        assert "0.6000" in captured.out
        assert "0.4500" in captured.out

    def test_two_system_comparison_shows_delta_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(rmse=0.10)})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics(rmse=0.08)})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "Δ" in captured.out
        assert "%Δ" in captured.out

    def test_two_system_comparison_shows_summary_footer(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(
            system="steamer", version="2025", stats={"hr": _make_stat_metrics(rmse=0.10, r_squared=0.60)}
        )
        sys_b = _make_system_metrics(
            system="zips", version="2025", stats={"hr": _make_stat_metrics(rmse=0.08, r_squared=0.70)}
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "zips/2025 vs steamer/2025" in captured.out
        assert "RMSE" in captured.out

    def test_three_system_comparison_no_deltas(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(system="steamer", version="2025")
        sys_b = _make_system_metrics(system="zips", version="2025")
        sys_c = _make_system_metrics(system="marcel", version="latest")
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b, sys_c])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "Δ" not in captured.out

    def test_improvement_colored_green(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(force_terminal=True, highlight=False, width=200))
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(rmse=0.10)})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics(rmse=0.05)})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "\x1b[32m" in captured.out  # ANSI green

    def test_regression_colored_red(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(force_terminal=True, highlight=False, width=200))
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(rmse=0.05)})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics(rmse=0.10)})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "\x1b[31m" in captured.out  # ANSI red

    def test_two_system_shows_rank_correlation_delta(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        sys_a = _make_system_metrics(
            system="steamer", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.80)}
        )
        sys_b = _make_system_metrics(
            system="zips", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.90)}
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "\u03c1" in captured.out
        assert "0.8000" in captured.out
        assert "0.9000" in captured.out

    def test_summary_footer_includes_rank_correlation(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(
            system="steamer", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.70)}
        )
        sys_b = _make_system_metrics(
            system="zips", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.85)}
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "on \u03c1" in captured.out

    def test_three_system_shows_rank_correlation(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        sys_a = _make_system_metrics(
            system="steamer", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.80)}
        )
        sys_b = _make_system_metrics(
            system="zips", version="2025", stats={"hr": _make_stat_metrics(rank_correlation=0.85)}
        )
        sys_c = _make_system_metrics(
            system="marcel", version="latest", stats={"hr": _make_stat_metrics(rank_correlation=0.75)}
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b, sys_c])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "\u03c1" in captured.out
        assert "0.8000" in captured.out
        assert "0.8500" in captured.out
        assert "0.7500" in captured.out

    def test_tail_section_shown_when_tail_data_present(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        tail = TailAccuracy(ns=(25,), rmse_by_stat={"hr": {25: 5.1234}})
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics()})
        sys_a_with_tail = SystemMetrics(
            system=sys_a.system, version=sys_a.version, source_type=sys_a.source_type, metrics=sys_a.metrics, tail=tail
        )
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics()})
        sys_b_with_tail = SystemMetrics(
            system=sys_b.system, version=sys_b.version, source_type=sys_b.source_type, metrics=sys_b.metrics, tail=tail
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a_with_tail, sys_b_with_tail])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "Tail accuracy" in captured.out
        assert "top-25" in captured.out
        assert "5.1234" in captured.out

    def test_tail_section_absent_when_no_tail_data(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics()})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics()})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "Tail accuracy" not in captured.out


class TestPrintStratifiedComparisonResult:
    def test_stratified_shows_r_squared(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(r_squared=0.55)})
        cohort = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a])
        result = StratifiedComparisonResult(dimension="pa_bucket", season=2024, stats=["hr"], cohorts={"top": cohort})
        print_stratified_comparison_result(result)
        captured = capsys.readouterr()
        assert "R²" in captured.out
        assert "0.5500" in captured.out


class TestPrintAblationResult:
    def test_print_ablation_with_se(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.0592},
            feature_standard_errors={"batter:hr": 0.0045},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "SE:" in captured.out
        assert "95% CI:" in captured.out

    def test_print_ablation_without_se_backward_compat(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.0592},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "SE:" not in captured.out
        assert "+0.0592" in captured.out

    def test_print_ablation_prune_candidate_marked(self, capsys: pytest.CaptureFixture[str]) -> None:
        # mean + 2*SE <= 0: mean=-0.01, SE=0.005 → -0.01 + 0.01 = 0.0 <= 0
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:noise": -0.01},
            feature_standard_errors={"batter:noise": 0.005},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "PRUNE" in captured.out

    def test_print_ablation_borderline_not_pruned(self, capsys: pytest.CaptureFixture[str]) -> None:
        # mean + 2*SE > 0: mean=0.001, SE=0.005 → 0.001 + 0.01 = 0.011 > 0
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:borderline": 0.001},
            feature_standard_errors={"batter:borderline": 0.005},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "PRUNE" not in captured.out

    def test_print_ablation_with_groups(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:a": 0.03, "batter:b": 0.02},
            feature_standard_errors={"batter:a": 0.005, "batter:b": 0.004},
            group_impacts={"batter:group_0": 0.06},
            group_standard_errors={"batter:group_0": 0.008},
            group_members={"batter:group_0": ["batter:a", "batter:b"]},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Feature Groups:" in captured.out
        assert "batter:group_0" in captured.out

    def test_print_ablation_group_prune_marked(self, capsys: pytest.CaptureFixture[str]) -> None:
        # mean + 2*SE <= 0: mean=-0.02, SE=0.005 → -0.02 + 0.01 = -0.01 <= 0
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:a": -0.01, "batter:b": -0.01},
            feature_standard_errors={"batter:a": 0.005, "batter:b": 0.005},
            group_impacts={"batter:group_0": -0.02},
            group_standard_errors={"batter:group_0": 0.005},
            group_members={"batter:group_0": ["batter:a", "batter:b"]},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "GROUP PRUNE" in captured.out

    def test_print_ablation_no_groups_omits_section(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            feature_standard_errors={"batter:hr": 0.01},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Feature Groups:" not in captured.out

    def test_print_ablation_group_members_listed(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:a": 0.03, "batter:b": 0.02},
            feature_standard_errors={"batter:a": 0.005, "batter:b": 0.004},
            group_impacts={"batter:group_0": 0.06},
            group_standard_errors={"batter:group_0": 0.008},
            group_members={"batter:group_0": ["batter:a", "batter:b"]},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "batter:a" in captured.out
        assert "batter:b" in captured.out


def _make_validation_result(
    player_type: str = "batter",
    go: bool = True,
    n_improved: int = 4,
    n_degraded: int = 2,
    max_degradation_pct: float = 1.3,
    pruned_features: tuple[str, ...] = ("sprint_speed", "pull_pct"),
    comparisons: tuple[TargetComparison, ...] | None = None,
) -> ValidationResult:
    if comparisons is None:
        comparisons = (
            TargetComparison(target="avg", full_rmse=0.0312, pruned_rmse=0.0310, delta_pct=-0.6),
            TargetComparison(target="obp", full_rmse=0.0289, pruned_rmse=0.0292, delta_pct=1.0),
        )
    return ValidationResult(
        player_type=player_type,
        comparisons=comparisons,
        pruned_features=pruned_features,
        n_improved=n_improved,
        n_degraded=n_degraded,
        max_degradation_pct=max_degradation_pct,
        go=go,
    )


class TestPrintAblationValidation:
    def test_validation_go_shows_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result(go=True)
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "GO" in captured.out

    def test_validation_nogo_shows_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result(go=False, n_improved=2, n_degraded=5, max_degradation_pct=8.2)
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "NO-GO" in captured.out

    def test_no_validation_omits_section(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Pruning Validation" not in captured.out

    def test_validation_shows_pruned_features(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result(pruned_features=("sprint_speed", "pull_pct"))
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "sprint_speed" in captured.out
        assert "pull_pct" in captured.out

    def test_validation_shows_rmse_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result()
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Full RMSE" in captured.out
        assert "Pruned RMSE" in captured.out


class TestPrintDraftBoard:
    def test_basic_output_contains_player_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Mike Trout",
                rank=1,
                player_type="batter",
                position="OF",
                value=42.5,
                category_z_scores={"hr": 2.10, "r": 1.00},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Gerrit Cole",
                rank=2,
                player_type="pitcher",
                position="SP",
                value=35.0,
                category_z_scores={"w": 1.50},
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr", "r"), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "Mike Trout" in captured.out
        assert "Gerrit Cole" in captured.out
        assert "$42.5" in captured.out
        assert "1" in captured.out  # rank

    def test_empty_board_prints_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        board = DraftBoard(rows=[], batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "No players on draft board." in captured.out

    def test_adp_delta_shows_signed_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Buy Target",
                rank=1,
                player_type="batter",
                position="OF",
                value=40.0,
                category_z_scores={},
                adp_overall=20.0,
                adp_rank=20,
                adp_delta=19,
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Avoid Player",
                rank=2,
                player_type="batter",
                position="1B",
                value=30.0,
                category_z_scores={},
                adp_overall=1.0,
                adp_rank=1,
                adp_delta=-1,
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        # Positive delta has + sign, negative has - sign
        assert "+19" in captured.out
        assert "-1" in captured.out
        assert "ADP" in captured.out
        assert "Delta" in captured.out

    def test_tier_column_shown_when_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Player A",
                rank=1,
                player_type="batter",
                position="OF",
                value=40.0,
                category_z_scores={},
                tier=1,
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "Tier" in captured.out

    def test_adp_delta_zero_shows_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Even Player",
                rank=1,
                player_type="batter",
                position="OF",
                value=30.0,
                category_z_scores={},
                adp_overall=1.0,
                adp_rank=1,
                adp_delta=0,
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "0" in captured.out


class TestPrintPrepareResult:
    def test_shows_model_name_and_rows(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = PrepareResult(model_name="xgb-v1", rows_processed=5000, artifacts_path="/tmp/artifacts")
        print_prepare_result(result)
        captured = capsys.readouterr()
        assert "xgb-v1" in captured.out
        assert "5000" in captured.out
        assert "/tmp/artifacts" in captured.out


class TestPrintTrainResult:
    def test_shows_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TrainResult(
            model_name="xgb-v1",
            metrics={"rmse": 0.045, "r2": 0.82},
            artifacts_path="/tmp/model",
        )
        print_train_result(result)
        captured = capsys.readouterr()
        assert "xgb-v1" in captured.out
        assert "rmse" in captured.out
        assert "0.045" in captured.out
        assert "/tmp/model" in captured.out

    def test_empty_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TrainResult(model_name="xgb-v1", metrics={}, artifacts_path="/tmp/model")
        print_train_result(result)
        captured = capsys.readouterr()
        assert "xgb-v1" in captured.out
        assert "/tmp/model" in captured.out


class TestPrintPredictResult:
    def test_shows_prediction_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = PredictResult(
            model_name="xgb-v1",
            predictions=[{"player_id": 1}, {"player_id": 2}, {"player_id": 3}],
            output_path="/tmp/preds",
        )
        print_predict_result(result)
        captured = capsys.readouterr()
        assert "xgb-v1" in captured.out
        assert "3 predictions" in captured.out


class TestPrintRoutingTable:
    def test_shows_stat_and_system(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        routes = {"hr": "steamer", "obp": "statcast-gbm"}
        print_routing_table(routes)

    def test_empty_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        print_routing_table({})


class TestPrintCoverageMatrix:
    def test_shows_systems_and_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        system_stats = {
            "steamer": {"hr", "rbi", "obp"},
            "statcast-gbm": {"obp", "avg"},
        }
        print_coverage_matrix(system_stats)

    def test_with_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        system_stats = {
            "steamer": {"hr", "rbi"},
            "statcast-gbm": {"obp"},
        }
        routes = {"hr": "steamer", "obp": "statcast-gbm"}
        print_coverage_matrix(system_stats, routes=routes)

    def test_with_required_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        system_stats = {
            "steamer": {"hr", "rbi"},
            "statcast-gbm": {"obp"},
        }
        routes = {"hr": "steamer", "obp": "statcast-gbm"}
        required = frozenset({"hr", "obp", "rbi"})
        print_coverage_matrix(system_stats, routes=routes, required_stats=required)

    def test_highlights_uncovered_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_output, "console", Console(width=120, force_terminal=True))
        system_stats = {
            "steamer": {"hr"},
        }
        routes = {"hr": "steamer"}
        required = frozenset({"hr", "rbi"})  # rbi is required but not routed
        print_coverage_matrix(system_stats, routes=routes, required_stats=required)


class TestPrintTuneResult:
    def test_shows_batter_and_pitcher_params(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"max_depth": 6, "learning_rate": 0.1},
            pitcher_params={"max_depth": 4, "learning_rate": 0.05},
            batter_cv_rmse={"hr": 0.0450, "avg": 0.0312},
            pitcher_cv_rmse={"era": 0.5200, "whip": 0.0800},
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        assert "xgb-v1" in captured.out
        assert "Batter best params:" in captured.out
        assert "Pitcher best params:" in captured.out
        assert "learning_rate" in captured.out
        assert "max_depth" in captured.out
        assert "Batter CV RMSE:" in captured.out
        assert "0.0450" in captured.out
        assert "TOML snippet" in captured.out

    def test_none_param_shows_unlimited_comment(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"max_depth": None},
            pitcher_params={"max_depth": 4},
            batter_cv_rmse={"hr": 0.05},
            pitcher_cv_rmse={"era": 0.5},
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        assert "unlimited" in captured.out

    def test_float_param_in_toml(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"learning_rate": 0.1},
            pitcher_params={},
            batter_cv_rmse={},
            pitcher_cv_rmse={},
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        assert "learning_rate = 0.1" in captured.out


class TestPrintImportResult:
    def test_shows_import_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = LoadLog(
            source_type="csv",
            source_detail="steamer_2025.csv",
            target_table="projections",
            rows_loaded=500,
            started_at="2025-01-01",
            finished_at="2025-01-01",
            status="success",
        )
        print_import_result(log)
        captured = capsys.readouterr()
        assert "500 projections loaded" in captured.out
        assert "steamer_2025.csv" in captured.out
        assert "success" in captured.out


class TestPrintIngestResult:
    def test_shows_ingest_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = LoadLog(
            source_type="csv",
            source_detail="batting.csv",
            target_table="batting_stats",
            rows_loaded=1200,
            started_at="2025-01-01",
            finished_at="2025-01-01",
            status="success",
        )
        print_ingest_result(log)
        captured = capsys.readouterr()
        assert "1200 rows loaded into batting_stats" in captured.out
        assert "batting.csv" in captured.out

    def test_shows_error_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = LoadLog(
            source_type="csv",
            source_detail="bad.csv",
            target_table="batting_stats",
            rows_loaded=0,
            started_at="2025-01-01",
            finished_at="2025-01-01",
            status="error",
            error_message="Invalid CSV format",
        )
        print_ingest_result(log)
        captured = capsys.readouterr()
        assert "Invalid CSV format" in captured.out


class TestPrintRunList:
    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_run_list([])
        captured = capsys.readouterr()
        assert "No runs found" in captured.out

    def test_shows_run_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        records = [
            ModelRunRecord(
                system="xgb",
                version="1.0",
                config_json={},
                artifact_type="model",
                created_at="2025-03-01",
                operation="train",
                tags_json={"env": "prod"},
            ),
            ModelRunRecord(
                system="xgb",
                version="1.0",
                config_json={},
                artifact_type="model",
                created_at="2025-03-02",
                operation="predict",
            ),
        ]
        print_run_list(records)
        captured = capsys.readouterr()
        assert "xgb" in captured.out
        assert "train" in captured.out
        assert "predict" in captured.out
        assert "env=prod" in captured.out


class TestPrintRunDetail:
    def test_shows_all_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        record = ModelRunRecord(
            system="xgb",
            version="1.0",
            config_json={"max_depth": 6},
            artifact_type="model",
            created_at="2025-03-01",
            operation="train",
            artifact_path="/tmp/model.pkl",
            git_commit="abc123",
            metrics_json={"rmse": 0.05},
            tags_json={"env": "prod"},
        )
        print_run_detail(record)
        captured = capsys.readouterr()
        assert "xgb" in captured.out
        assert "1.0" in captured.out
        assert "train" in captured.out
        assert "/tmp/model.pkl" in captured.out
        assert "abc123" in captured.out
        assert "max_depth" in captured.out
        assert "rmse" in captured.out
        assert "env=prod" in captured.out

    def test_missing_optional_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        record = ModelRunRecord(
            system="xgb",
            version="1.0",
            config_json={},
            artifact_type="model",
            created_at="2025-03-01",
        )
        print_run_detail(record)
        captured = capsys.readouterr()
        assert "N/A" in captured.out  # git_commit and artifact_path


class TestPrintPerformanceReport:
    def test_empty_deltas(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_performance_report("Breakouts", [])
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_shows_delta_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        deltas = [
            PlayerStatDelta(1, "J. Soto", "avg", 0.310, 0.280, 0.030, 0.030, 95.0),
            PlayerStatDelta(2, "M. Trout", "avg", 0.220, 0.260, -0.040, -0.040, 5.0),
        ]
        print_performance_report("Breakouts", deltas)
        captured = capsys.readouterr()
        assert "Breakouts" in captured.out
        assert "J. Soto" in captured.out
        assert "M. Trout" in captured.out
        assert "0.310" in captured.out
        assert "95" in captured.out

    def test_zero_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        deltas = [PlayerStatDelta(1, "P1", "avg", 0.280, 0.280, 0.0, 0.0, 50.0)]
        print_performance_report("Report", deltas)
        captured = capsys.readouterr()
        assert "+0.000" in captured.out


class TestPrintSystemSummaries:
    def test_empty_summaries(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_summaries([])
        captured = capsys.readouterr()
        assert "No projection systems found" in captured.out

    def test_shows_summary_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        summaries = [
            SystemSummary(
                system="steamer", version="2025.1", source_type="third_party", batter_count=300, pitcher_count=200
            ),
            SystemSummary(
                system="zips", version="2025.1", source_type="third_party", batter_count=280, pitcher_count=180
            ),
        ]
        print_system_summaries(summaries)
        captured = capsys.readouterr()
        assert "steamer" in captured.out
        assert "zips" in captured.out
        assert "300" in captured.out
        assert "500" in captured.out  # total


class TestPrintDatasetList:
    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_dataset_list([])
        captured = capsys.readouterr()
        assert "No cached datasets found" in captured.out

    def test_shows_dataset_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        datasets = [
            DatasetInfo(
                dataset_id=1,
                feature_set_id=10,
                feature_set_name="batting_v2",
                feature_set_version="2.0",
                split="train",
                table_name="ds_batting_v2_train",
                row_count=5000,
                seasons=[2022, 2023],
                created_at="2025-03-01",
            ),
        ]
        print_dataset_list(datasets)
        captured = capsys.readouterr()
        assert "batting_v2" in captured.out
        assert "2.0" in captured.out
        assert "train" in captured.out
        assert "5000" in captured.out
        assert "2022, 2023" in captured.out

    def test_null_split_shows_dash(self, capsys: pytest.CaptureFixture[str]) -> None:
        datasets = [
            DatasetInfo(
                dataset_id=1,
                feature_set_id=10,
                feature_set_name="fs",
                feature_set_version="1.0",
                split=None,
                table_name="tbl",
                row_count=100,
                seasons=[],
                created_at="2025-01-01",
            ),
        ]
        print_dataset_list(datasets)
        captured = capsys.readouterr()
        # The dash character used for null split
        assert "\u2014" in captured.out


def _make_talent_quality_report(
    *,
    stat_metrics: list[StatTalentMetrics] | None = None,
    passes: bool = True,
) -> TrueTalentQualityReport:
    if stat_metrics is None:
        stat_metrics = [
            StatTalentMetrics(
                stat_name="avg",
                model_next_season_corr=0.85,
                raw_next_season_corr=0.70,
                predictive_validity_pass=True,
                residual_yoy_corr=0.10,
                residual_non_persistence_pass=True,
                shrinkage_ratio=0.80,
                estimate_raw_corr=0.90,
                shrinkage_pass=True,
                r_squared=0.75,
                residual_by_bucket={},
                r_squared_pass=True,
                regression_rate=0.90,
                regression_rate_pass=True,
                n_season_n=300,
                n_returning=250,
            ),
        ]
    p = 1 if passes else 0
    summary = TalentQualitySummary(
        predictive_validity_passes=p,
        predictive_validity_total=1,
        residual_non_persistence_passes=p,
        residual_non_persistence_total=1,
        shrinkage_passes=p,
        shrinkage_total=1,
        r_squared_passes=p,
        r_squared_total=1,
        regression_rate_passes=p,
        regression_rate_total=1,
    )
    return TrueTalentQualityReport(
        system="fbm",
        version="1.0",
        season_n=2023,
        season_n1=2024,
        player_type="batter",
        stat_metrics=stat_metrics,
        summary=summary,
    )


class TestPrintTalentQualityReport:
    def test_empty_reports(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_talent_quality_report([])
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_shows_summary_and_stat_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_talent_quality_report()
        print_talent_quality_report([report])
        captured = capsys.readouterr()
        assert "True-Talent Quality" in captured.out
        assert "fbm/1.0" in captured.out
        assert "2023 -> 2024" in captured.out
        assert "Summary:" in captured.out
        assert "Predictive validity:" in captured.out
        assert "Residual non-persistence:" in captured.out
        assert "Shrinkage quality:" in captured.out
        assert "avg" in captured.out
        assert "0.850" in captured.out  # model_next_season_corr

    def test_failing_passes_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        metric = StatTalentMetrics(
            stat_name="hr",
            model_next_season_corr=0.50,
            raw_next_season_corr=0.60,
            predictive_validity_pass=False,
            residual_yoy_corr=0.25,
            residual_non_persistence_pass=False,
            shrinkage_ratio=0.95,
            estimate_raw_corr=0.40,
            shrinkage_pass=False,
            r_squared=0.50,
            residual_by_bucket={},
            r_squared_pass=False,
            regression_rate=0.70,
            regression_rate_pass=False,
            n_season_n=200,
            n_returning=150,
        )
        report = _make_talent_quality_report(stat_metrics=[metric], passes=False)
        print_talent_quality_report([report])
        captured = capsys.readouterr()
        assert "0/1" in captured.out


def _make_residual_persistence_report(
    *,
    go: bool = True,
    chronic: bool = False,
) -> ResidualPersistenceReport:
    overperformers = [ChronicPerformer(1, "J. Soto", 0.030, 0.025, 0.028, 600, 550)] if chronic else []
    underperformers = [ChronicPerformer(2, "M. Trout", -0.040, -0.035, -0.038, 500, 450)] if chronic else []
    stat_metrics = [
        StatResidualPersistence(
            stat_name="avg",
            residual_corr_overall=0.08,
            residual_corr_by_bucket={"<200": 0.05, "200-400": 0.08, "400+": 0.10},
            n_by_bucket={"<200": 50, "200-400": 100, "400+": 150},
            chronic_overperformers=overperformers,
            chronic_underperformers=underperformers,
            rmse_baseline=0.0400,
            rmse_corrected=0.0385,
            rmse_improvement_pct=3.8,
            persistence_pass=True,
            ceiling_pass=True,
            n_returning=250,
        ),
    ]
    summary = ResidualPersistenceSummary(
        persistence_passes=4 if go else 1,
        persistence_total=5,
        ceiling_passes=3 if go else 0,
        ceiling_total=5,
        go=go,
    )
    return ResidualPersistenceReport(
        system="fbm",
        version="1.0",
        season_n=2023,
        season_n1=2024,
        stat_metrics=stat_metrics,
        summary=summary,
    )


class TestPrintResidualPersistenceReport:
    def test_shows_correlation_and_rmse_tables(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_persistence_report()
        print_residual_persistence_report(report)
        captured = capsys.readouterr()
        assert "Residual Persistence Diagnostic" in captured.out
        assert "Residual Correlation" in captured.out
        assert "RMSE Improvement Ceiling" in captured.out
        assert "avg" in captured.out
        assert "0.080" in captured.out  # overall corr
        assert "0.0400" in captured.out  # baseline
        assert "3.8%" in captured.out  # improvement

    def test_go_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_persistence_report(go=True)
        print_residual_persistence_report(report)
        captured = capsys.readouterr()
        assert "GO" in captured.out
        assert "4/5" in captured.out

    def test_nogo_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_persistence_report(go=False)
        print_residual_persistence_report(report)
        captured = capsys.readouterr()
        assert "NO-GO" in captured.out

    def test_chronic_performers_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_persistence_report(chronic=True)
        print_residual_persistence_report(report)
        captured = capsys.readouterr()
        assert "Chronic Performers" in captured.out
        assert "Overperformers:" in captured.out
        assert "J. Soto" in captured.out
        assert "Underperformers:" in captured.out
        assert "M. Trout" in captured.out


def _make_value_over_adp_entry(
    player_name: str = "J. Soto",
    rank_delta: int = 20,
    zar_value: float = 42.5,
    zar_rank: int = 1,
    adp_rank: int = 21,
    adp_pick: float = 21.3,
) -> ValueOverADP:
    return ValueOverADP(
        player_id=1,
        player_name=player_name,
        player_type="batter",
        position="OF",
        adp_positions="OF",
        zar_rank=zar_rank,
        zar_value=zar_value,
        adp_rank=adp_rank,
        adp_pick=adp_pick,
        rank_delta=rank_delta,
        provider="espn",
    )


class TestPrintValueOverADP:
    def test_all_sections(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ValueOverADPReport(
            season=2025,
            system="zar",
            version="1.0",
            provider="espn",
            buy_targets=[_make_value_over_adp_entry(player_name="Buy Guy", rank_delta=30)],
            avoid_list=[_make_value_over_adp_entry(player_name="Avoid Guy", rank_delta=-25)],
            unranked_valuable=[_make_value_over_adp_entry(player_name="Sleeper Guy")],
            n_matched=100,
        )
        print_value_over_adp(report)
        captured = capsys.readouterr()
        assert "Value-Over-ADP" in captured.out
        assert "Buy Targets" in captured.out
        assert "Buy Guy" in captured.out
        assert "Avoid List" in captured.out
        assert "Avoid Guy" in captured.out
        assert "Unranked Sleepers" in captured.out
        assert "Sleeper Guy" in captured.out

    def test_empty_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ValueOverADPReport(
            season=2025,
            system="zar",
            version="1.0",
            provider="espn",
            buy_targets=[],
            avoid_list=[],
            unranked_valuable=[],
            n_matched=0,
        )
        print_value_over_adp(report)
        captured = capsys.readouterr()
        assert "No discrepancies found" in captured.out


def _make_adp_accuracy_result(
    season: int = 2024,
    n_matched: int = 150,
) -> ADPAccuracyResult:
    return ADPAccuracyResult(
        season=season,
        provider="espn",
        rank_correlation=0.72,
        value_rmse=8.50,
        value_mae=6.20,
        top_n_precision={10: 0.60, 50: 0.55},
        n_matched=n_matched,
        players=[],
    )


class TestPrintADPAccuracyReport:
    def test_single_season_no_comparison(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ADPAccuracyReport(
            provider="espn",
            seasons=[2024],
            adp_results=[_make_adp_accuracy_result()],
            comparison=None,
            mean_rank_correlation=0.72,
            mean_value_rmse=8.50,
            mean_top_n_precision={10: 0.60, 50: 0.55},
        )
        print_adp_accuracy_report(report)
        captured = capsys.readouterr()
        assert "ADP Accuracy" in captured.out
        assert "season 2024" in captured.out
        assert "0.7200" in captured.out
        assert "$8.50" in captured.out
        assert "60.0%" in captured.out

    def test_single_season_with_comparison(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_result = SystemAccuracyResult(
            system="zar",
            version="1.0",
            season=2024,
            rank_correlation=0.80,
            value_rmse=7.00,
            value_mae=5.50,
            top_n_precision={10: 0.70, 50: 0.60},
            n_matched=150,
        )
        report = ADPAccuracyReport(
            provider="espn",
            seasons=[2024],
            adp_results=[_make_adp_accuracy_result()],
            comparison=[sys_result],
            mean_rank_correlation=0.72,
            mean_value_rmse=8.50,
            mean_top_n_precision={10: 0.60, 50: 0.55},
        )
        print_adp_accuracy_report(report)
        captured = capsys.readouterr()
        assert "ADP" in captured.out
        assert "zar/1.0" in captured.out
        assert "0.8000" in captured.out
        assert "$7.00" in captured.out

    def test_multi_season(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ADPAccuracyReport(
            provider="espn",
            seasons=[2023, 2024],
            adp_results=[
                _make_adp_accuracy_result(season=2023),
                _make_adp_accuracy_result(season=2024),
            ],
            comparison=None,
            mean_rank_correlation=0.72,
            mean_value_rmse=8.50,
            mean_top_n_precision={10: 0.60, 50: 0.55},
        )
        print_adp_accuracy_report(report)
        captured = capsys.readouterr()
        assert "2 seasons" in captured.out
        assert "2023" in captured.out
        assert "2024" in captured.out
        assert "Mean" in captured.out

    def test_multi_season_low_n_shows_dash(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ADPAccuracyReport(
            provider="espn",
            seasons=[2023, 2024],
            adp_results=[
                _make_adp_accuracy_result(season=2023, n_matched=2),
                _make_adp_accuracy_result(season=2024),
            ],
            comparison=None,
            mean_rank_correlation=0.72,
            mean_value_rmse=8.50,
            mean_top_n_precision={10: 0.60},
        )
        print_adp_accuracy_report(report)
        captured = capsys.readouterr()
        # n_matched < 3 → Spearman shows dash
        assert "\u2014" in captured.out


class TestPrintADPMoversReport:
    def test_all_sections(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ADPMoversReport(
            season=2025,
            provider="espn",
            current_as_of="2025-03-15",
            previous_as_of="2025-03-01",
            risers=[ADPMover("Rising Star", "OF", 10, 30, 20, "up")],
            fallers=[ADPMover("Falling Star", "SP", 40, 15, -25, "down")],
            new_entries=[ADPMover("New Guy", "SS", 50, 0, 50, "new")],
            dropped_entries=[ADPMover("Gone Guy", "1B", 0, 45, -45, "dropped")],
        )
        print_adp_movers_report(report)
        captured = capsys.readouterr()
        assert "ADP Movers" in captured.out
        assert "2025-03-01 -> 2025-03-15" in captured.out
        assert "Risers" in captured.out
        assert "Rising Star" in captured.out
        assert "+20" in captured.out
        assert "Fallers" in captured.out
        assert "Falling Star" in captured.out
        assert "-25" in captured.out
        assert "New Entries" in captured.out
        assert "New Guy" in captured.out
        assert "Dropped" in captured.out
        assert "Gone Guy" in captured.out

    def test_empty_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ADPMoversReport(
            season=2025,
            provider="espn",
            current_as_of="2025-03-15",
            previous_as_of="2025-03-01",
            risers=[],
            fallers=[],
            new_entries=[],
            dropped_entries=[],
        )
        print_adp_movers_report(report)
        captured = capsys.readouterr()
        assert "No movers found" in captured.out


class TestPrintProjectionConfidence:
    def test_empty_players(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = ConfidenceReport(season=2025, systems=["steamer", "zips"], players=[])
        print_projection_confidence(report)
        captured = capsys.readouterr()
        assert "No players with sufficient projection systems" in captured.out

    def test_shows_confidence_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        players = [
            PlayerConfidence(
                player_id=1,
                player_name="J. Soto",
                player_type="batter",
                position="OF",
                spreads=[
                    StatSpread(stat="hr", min_value=25, max_value=35, mean=30, std=5.0, cv=0.167, systems={}),
                    StatSpread(
                        stat="avg", min_value=0.270, max_value=0.300, mean=0.285, std=0.015, cv=0.053, systems={}
                    ),
                ],
                overall_cv=0.110,
                agreement_level="high",
            ),
            PlayerConfidence(
                player_id=2,
                player_name="M. Trout",
                player_type="batter",
                position="CF",
                spreads=[
                    StatSpread(stat="hr", min_value=15, max_value=40, mean=27.5, std=12.5, cv=0.455, systems={}),
                ],
                overall_cv=0.455,
                agreement_level="low",
            ),
        ]
        report = ConfidenceReport(season=2025, systems=["steamer", "zips"], players=players)
        print_projection_confidence(report)
        captured = capsys.readouterr()
        assert "Projection Confidence" in captured.out
        assert "2 systems" in captured.out
        assert "2 players" in captured.out
        assert "J. Soto" in captured.out
        assert "M. Trout" in captured.out
        assert "high" in captured.out
        assert "low" in captured.out
        assert "25-35" in captured.out
        assert "0.110" in captured.out


class TestPrintPlayerProjectionsEmpty:
    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_player_projections([])
        captured = capsys.readouterr()
        assert "No projections found" in captured.out


class TestPrintValuationEvalResultTop:
    def test_top_limits_player_rows(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _make_eval_result()
        print_valuation_eval_result(result, top=1)
        captured = capsys.readouterr()
        assert "Mike Trout" in captured.out
        assert "Aaron Judge" not in captured.out


class TestPrintValidationEdgeCases:
    def test_empty_pruned_features(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result(pruned_features=())
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Pruned:" not in captured.out

    def test_empty_comparisons(self, capsys: pytest.CaptureFixture[str]) -> None:
        vr = _make_validation_result(comparisons=())
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "Full RMSE" not in captured.out

    def test_zero_delta_pct(self, capsys: pytest.CaptureFixture[str]) -> None:
        comp = TargetComparison(target="avg", full_rmse=0.03, pruned_rmse=0.03, delta_pct=0.0)
        vr = _make_validation_result(comparisons=(comp,))
        result = AblationResult(
            model_name="test-model",
            feature_impacts={"batter:hr": 0.05},
            validation_results={"batter": vr},
        )
        print_ablation_result(result)
        captured = capsys.readouterr()
        assert "+0.0%" in captured.out


class TestPrintTuneResultPitcherNone:
    def test_pitcher_none_param_shows_unlimited(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"max_depth": 4},
            pitcher_params={"max_depth": None},
            batter_cv_rmse={},
            pitcher_cv_rmse={},
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        # Both the batter (int) and pitcher (None) branches in TOML section
        assert captured.out.count("unlimited") == 1


class TestPrintTuneResultPerTargetBest:
    def test_shows_per_target_optimal(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"max_depth": 4},
            pitcher_params={"max_depth": 3},
            batter_cv_rmse={"hr": 0.05},
            pitcher_cv_rmse={"era": 0.50, "whip": 0.08},
            pitcher_per_target_best={
                "era": PerTargetBest(
                    target="era",
                    best_params={"max_depth": 5, "loss": "absolute_error"},
                    best_rmse=0.45,
                    joint_rmse=0.50,
                    delta_pct=11.1,
                ),
                "whip": PerTargetBest(
                    target="whip",
                    best_params={"max_depth": 3},
                    best_rmse=0.08,
                    joint_rmse=0.08,
                    delta_pct=0.0,
                ),
            },
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        assert "per-target optimal" in captured.out
        assert "era" in captured.out
        assert "11.1" in captured.out
        assert "max_depth" in captured.out


class TestPrintPlayerProjectionsIntStat:
    def test_int_stat_shown_as_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(stats={"pa": 600})
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "600" in captured.out


class TestPrintDraftBoardAdpDeltaNone:
    def test_adp_present_but_delta_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="No Delta",
                rank=1,
                player_type="batter",
                position="OF",
                value=30.0,
                category_z_scores={},
                adp_overall=5.0,
                adp_rank=5,
                adp_delta=None,
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "No Delta" in captured.out
        assert "5.0" in captured.out


class TestPrintFeaturesDerivedTransform:
    def test_derived_transform_feature(self, capsys: pytest.CaptureFixture[str]) -> None:
        def dummy(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {}

        feature = DerivedTransformFeature(
            name="derived_calc",
            inputs=("hr", "ab"),
            group_by=("player_id", "season"),
            transform=dummy,
            outputs=("hr_rate", "iso"),
        )
        print_features("test_model", (feature,))
        captured = capsys.readouterr()
        assert "derived transform" in captured.out
        assert "hr_rate" in captured.out
        assert "iso" in captured.out


def _make_player_confidence(
    player_id: int = 1,
    player_name: str = "J. Soto",
    player_type: str = "batter",
    position: str = "OF",
    overall_cv: float = 0.110,
    agreement_level: str = "high",
    spreads: list[StatSpread] | None = None,
) -> PlayerConfidence:
    if spreads is None:
        spreads = [
            StatSpread(stat="hr", min_value=25, max_value=35, mean=30, std=5.0, cv=0.167, systems={}),
        ]
    return PlayerConfidence(
        player_id=player_id,
        player_name=player_name,
        player_type=player_type,
        position=position,
        spreads=spreads,
        overall_cv=overall_cv,
        agreement_level=agreement_level,
    )


def _make_classified_player(
    player_id: int = 1,
    player_name: str = "J. Soto",
    classification: VarianceClassification = VarianceClassification.SAFE_CONSENSUS,
    adp_rank: int | None = 5,
    value_rank: int = 3,
    risk_reward_score: float = 2.5,
    **kwargs: Any,
) -> ClassifiedPlayer:
    return ClassifiedPlayer(
        player=_make_player_confidence(player_id=player_id, player_name=player_name, **kwargs),
        classification=classification,
        adp_rank=adp_rank,
        value_rank=value_rank,
        risk_reward_score=risk_reward_score,
    )


class TestPrintVarianceTargets:
    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_variance_targets([])
        captured = capsys.readouterr()
        assert "No classified players" in captured.out

    def test_groups_by_classification(self, capsys: pytest.CaptureFixture[str]) -> None:
        classified = [
            _make_classified_player(
                player_id=1,
                player_name="Upside Guy",
                classification=VarianceClassification.UPSIDE_GAMBLE,
                risk_reward_score=5.0,
            ),
            _make_classified_player(
                player_id=2,
                player_name="Safe Guy",
                classification=VarianceClassification.SAFE_CONSENSUS,
                risk_reward_score=1.0,
            ),
            _make_classified_player(
                player_id=3,
                player_name="Risky Guy",
                classification=VarianceClassification.RISKY_AVOID,
                risk_reward_score=-3.0,
            ),
        ]
        print_variance_targets(classified)
        captured = capsys.readouterr()
        assert "Upside Guy" in captured.out
        assert "Safe Guy" in captured.out
        assert "Risky Guy" in captured.out
        assert "UPSIDE_GAMBLE" in captured.out or "Upside Gamble" in captured.out
        assert "SAFE_CONSENSUS" in captured.out or "Safe Consensus" in captured.out
        assert "RISKY_AVOID" in captured.out or "Risky Avoid" in captured.out

    def test_rr_score_signed(self, capsys: pytest.CaptureFixture[str]) -> None:
        classified = [
            _make_classified_player(
                player_id=1,
                player_name="Positive RR",
                classification=VarianceClassification.UPSIDE_GAMBLE,
                risk_reward_score=5.0,
            ),
            _make_classified_player(
                player_id=2,
                player_name="Negative RR",
                classification=VarianceClassification.RISKY_AVOID,
                risk_reward_score=-3.0,
            ),
        ]
        print_variance_targets(classified)
        captured = capsys.readouterr()
        assert "+5.0" in captured.out
        assert "-3.0" in captured.out

    def test_header_shows_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        classified = [
            _make_classified_player(player_id=1, player_name="Player A"),
            _make_classified_player(player_id=2, player_name="Player B"),
        ]
        print_variance_targets(classified)
        captured = capsys.readouterr()
        assert "2 players classified" in captured.out


class TestPrintSystemDisagreements:
    def test_shows_per_system_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        player = _make_player_confidence(
            player_name="J. Soto",
            position="OF",
            spreads=[
                StatSpread(
                    stat="hr",
                    min_value=25,
                    max_value=35,
                    mean=30,
                    std=5.0,
                    cv=0.167,
                    systems={"steamer": 25.0, "zips": 35.0, "atc": 30.0},
                ),
            ],
        )
        projections = [
            Projection(
                player_id=1, season=2025, system="steamer", version="1.0", player_type="batter", stat_json={"hr": 25.0}
            ),
            Projection(
                player_id=1, season=2025, system="zips", version="1.0", player_type="batter", stat_json={"hr": 35.0}
            ),
            Projection(
                player_id=1, season=2025, system="atc", version="1.0", player_type="batter", stat_json={"hr": 30.0}
            ),
        ]
        print_system_disagreements(player, projections)
        captured = capsys.readouterr()
        assert "J. Soto" in captured.out
        assert "OF" in captured.out
        assert "hr" in captured.out
        assert "steamer" in captured.out
        assert "zips" in captured.out
        assert "atc" in captured.out

    def test_sorted_by_cv_descending(self, capsys: pytest.CaptureFixture[str]) -> None:
        player = _make_player_confidence(
            spreads=[
                StatSpread(
                    stat="avg",
                    min_value=0.270,
                    max_value=0.300,
                    mean=0.285,
                    std=0.015,
                    cv=0.053,
                    systems={"steamer": 0.270, "zips": 0.300},
                ),
                StatSpread(
                    stat="hr",
                    min_value=15,
                    max_value=40,
                    mean=27.5,
                    std=12.5,
                    cv=0.455,
                    systems={"steamer": 15.0, "zips": 40.0},
                ),
            ],
        )
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="1.0",
                player_type="batter",
                stat_json={"hr": 15.0, "avg": 0.270},
            ),
            Projection(
                player_id=1,
                season=2025,
                system="zips",
                version="1.0",
                player_type="batter",
                stat_json={"hr": 40.0, "avg": 0.300},
            ),
        ]
        print_system_disagreements(player, projections)
        captured = capsys.readouterr()
        # hr (CV=0.455) should appear before avg (CV=0.053) in the output
        hr_pos = captured.out.find("hr")
        avg_pos = captured.out.find("avg")
        assert hr_pos < avg_pos

    def test_empty_spreads(self, capsys: pytest.CaptureFixture[str]) -> None:
        player = _make_player_confidence(spreads=[])
        print_system_disagreements(player, [])
        captured = capsys.readouterr()
        assert "J. Soto" in captured.out


def _make_residual_analysis_report(
    *,
    bias_significant: bool = True,
    hetero_significant: bool = False,
) -> ResidualAnalysisReport:
    bins = [
        CalibrationBin(bin_center=0.250, mean_predicted=0.248, mean_actual=0.258, mean_residual=0.010, count=50),
        CalibrationBin(bin_center=0.280, mean_predicted=0.279, mean_actual=0.286, mean_residual=0.007, count=50),
    ]
    stat_analyses = [
        StatResidualAnalysis(
            stat_name="avg",
            player_type="batter",
            n_observations=100,
            mean_residual=0.0085,
            std_residual=0.025,
            bias_significant=bias_significant,
            heteroscedasticity_corr=0.12,
            heteroscedasticity_significant=hetero_significant,
            calibration_bins=bins,
        ),
    ]
    summary = ResidualAnalysisSummary(
        n_bias_significant=1 if bias_significant else 0,
        n_bias_total=1,
        n_hetero_significant=1 if hetero_significant else 0,
        n_hetero_total=1,
        calibration_recommended=bias_significant,
    )
    return ResidualAnalysisReport(
        system="test-sys",
        version="v1",
        seasons=[2023, 2024],
        top=300,
        stat_analyses=stat_analyses,
        summary=summary,
    )


class TestPrintRegressionCheckResult:
    def test_pass_shows_green(self, capsys: pytest.CaptureFixture[str]) -> None:
        check = RegressionCheckResult(
            passed=True,
            rmse_passed=True,
            rank_correlation_passed=True,
            explanation="PASS: candidate RMSE 5/8 wins, \u03c1 6/8 wins",
        )
        print_regression_check_result(check)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_fail_shows_red(self, capsys: pytest.CaptureFixture[str]) -> None:
        check = RegressionCheckResult(
            passed=False,
            rmse_passed=False,
            rank_correlation_passed=True,
            explanation="FAIL: candidate RMSE 5/8 losses, \u03c1 3/8 wins",
        )
        print_regression_check_result(check)
        captured = capsys.readouterr()
        assert "FAIL" in captured.out


class TestPrintResidualAnalysisReport:
    def test_shows_stat_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_analysis_report()
        print_residual_analysis_report(report)
        captured = capsys.readouterr()
        assert "Residual Analysis" in captured.out
        assert "avg" in captured.out
        assert "batter" in captured.out
        assert "0.0085" in captured.out  # mean residual

    def test_shows_calibration_recommendation(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_analysis_report(bias_significant=True)
        print_residual_analysis_report(report)
        captured = capsys.readouterr()
        assert "Calibration recommended" in captured.out

    def test_shows_no_calibration_needed(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_analysis_report(bias_significant=False)
        print_residual_analysis_report(report)
        captured = capsys.readouterr()
        assert "No calibration needed" in captured.out

    def test_shows_calibration_bins(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _make_residual_analysis_report()
        print_residual_analysis_report(report)
        captured = capsys.readouterr()
        assert "Calibration Bins" in captured.out
        assert "0.2500" in captured.out  # bin center


def _make_keeper_decision(
    player_id: int = 1,
    player_name: str = "Mike Trout",
    position: str = "cf",
    cost: float = 10.0,
    projected_value: float = 25.0,
    surplus: float = 15.0,
    years_remaining: int = 1,
    recommendation: str = "keep",
) -> KeeperDecision:
    return KeeperDecision(
        player_id=player_id,
        player_name=player_name,
        position=position,
        cost=cost,
        projected_value=projected_value,
        surplus=surplus,
        years_remaining=years_remaining,
        recommendation=recommendation,
    )


def _make_keeper_set(players: tuple[KeeperDecision, ...] | None = None) -> KeeperSet:
    if players is None:
        players = (
            _make_keeper_decision(),
            _make_keeper_decision(
                player_id=2, player_name="Aaron Judge", position="of", cost=15.0, projected_value=30.0, surplus=15.0
            ),
        )
    total_surplus = sum(p.surplus for p in players)
    total_cost = sum(p.cost for p in players)
    return KeeperSet(
        players=players,
        total_surplus=total_surplus,
        total_cost=total_cost,
        positions_filled={"cf": 1, "of": 1},
        score=total_surplus,
    )


def _make_keeper_solution() -> KeeperSolution:
    optimal = _make_keeper_set()
    alt_player = _make_keeper_decision(
        player_id=3, player_name="Mookie Betts", position="of", cost=20.0, projected_value=28.0, surplus=8.0
    )
    alt_set = _make_keeper_set(
        players=(
            _make_keeper_decision(),
            alt_player,
        )
    )
    return KeeperSolution(
        optimal=optimal,
        alternatives=[alt_set],
        sensitivity=[
            SensitivityEntry(player_name="Aaron Judge", player_id=2, surplus_gap=7.0),
            SensitivityEntry(player_name="Mike Trout", player_id=1, surplus_gap=15.0),
        ],
    )


class TestPrintKeeperSolution:
    def test_shows_optimal_set(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "Optimal Keeper Set" in captured.out
        assert "Mike Trout" in captured.out
        assert "Aaron Judge" in captured.out

    def test_shows_cost_and_surplus(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "$10" in captured.out  # Trout cost
        assert "$25" in captured.out  # Trout value
        assert "$15.0" in captured.out  # Trout surplus

    def test_shows_alternatives(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "Alt 1" in captured.out
        assert "Mookie Betts" in captured.out

    def test_shows_sensitivity(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "Sensitivity" in captured.out
        assert "Aaron Judge" in captured.out
        assert "7.0" in captured.out  # surplus gap

    def test_shows_summary_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "$30.0" in captured.out  # total surplus
        assert "$25" in captured.out  # total cost


class TestPrintKeeperScenarios:
    def test_shows_scenario_ranking(self, capsys: pytest.CaptureFixture[str]) -> None:
        scenarios = [
            KeeperScenario(
                name="Best",
                keepers=[1, 2],
                keeper_set=_make_keeper_set(),
                delta_vs_optimal=0.0,
            ),
            KeeperScenario(
                name="Worse",
                keepers=[1, 3],
                keeper_set=_make_keeper_set(
                    players=(
                        _make_keeper_decision(),
                        _make_keeper_decision(player_id=3, player_name="Mookie Betts", position="of", surplus=8.0),
                    )
                ),
                delta_vs_optimal=7.0,
            ),
        ]
        print_keeper_scenarios(scenarios)
        captured = capsys.readouterr()
        assert "Scenario Comparison" in captured.out
        assert "Best" in captured.out
        assert "Worse" in captured.out
        assert "+$7.0" in captured.out

    def test_best_scenario_shows_zero_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        scenarios = [
            KeeperScenario(
                name="Only",
                keepers=[1],
                keeper_set=_make_keeper_set(),
                delta_vs_optimal=0.0,
            ),
        ]
        print_keeper_scenarios(scenarios)
        captured = capsys.readouterr()
        assert "Only" in captured.out


class TestPrintKeeperTradeImpact:
    def test_shows_before_and_after(self, capsys: pytest.CaptureFixture[str]) -> None:
        before = _make_keeper_solution()
        after_player = _make_keeper_decision(
            player_id=4, player_name="Juan Soto", position="of", cost=12.0, projected_value=32.0, surplus=20.0
        )
        after_set = _make_keeper_set(
            players=(
                _make_keeper_decision(),
                after_player,
            )
        )
        after = KeeperSolution(
            optimal=after_set,
            alternatives=[],
            sensitivity=[],
        )
        impact = KeeperTradeImpact(before=before, after=after, score_delta=5.0)
        print_keeper_trade_impact(impact)
        captured = capsys.readouterr()
        assert "Before" in captured.out
        assert "After" in captured.out
        assert "5.0" in captured.out

    def test_shows_negative_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = _make_keeper_solution()
        worse_set = _make_keeper_set(
            players=(
                _make_keeper_decision(surplus=5.0),
                _make_keeper_decision(player_id=2, player_name="Aaron Judge", position="of", surplus=5.0),
            )
        )
        worse = KeeperSolution(optimal=worse_set, alternatives=[], sensitivity=[])
        impact = KeeperTradeImpact(before=solution, after=worse, score_delta=-20.0)
        print_keeper_trade_impact(impact)
        captured = capsys.readouterr()
        assert "-$20.0" in captured.out


# ---------------------------------------------------------------------------
# _residuals.py tests
# ---------------------------------------------------------------------------


def _make_error_decomposition_report(
    player_type: str = "batter",
    *,
    with_features: bool = True,
) -> ErrorDecompositionReport:
    vol_key = "ip" if player_type == "pitcher" else "pa"
    features: dict[str, float] = {"age": 32.0, vol_key: 500.0}
    return ErrorDecompositionReport(
        target="avg",
        player_type=player_type,
        season=2024,
        system="fbm",
        version="1.0",
        top_misses=[
            PlayerResidual(1, "Mike Trout", 0.300, 0.250, 0.050, features),
            PlayerResidual(2, "Mookie Betts", 0.280, 0.310, -0.030, features),
        ],
        over_predictions=[],
        under_predictions=[],
        summary=MissPopulationSummary(
            mean_age=31.5 if with_features else None,
            position_distribution={"OF": 2} if with_features else {},
            mean_volume=550.0,
            distinguishing_features=[
                DistinguishingFeature("sprint_speed", 28.5, 27.0, 1.5),
            ]
            if with_features
            else [],
        ),
    )


class TestPrintErrorDecompositionReport:
    def test_shows_title_and_players(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error_decomposition_report(_make_error_decomposition_report())
        captured = capsys.readouterr()
        assert "Mike Trout" in captured.out
        assert "Mookie Betts" in captured.out
        assert "31.5" in captured.out
        assert "OF: 2" in captured.out
        assert "sprint_speed" in captured.out

    def test_pitcher_uses_ip(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error_decomposition_report(_make_error_decomposition_report("pitcher"))
        captured = capsys.readouterr()
        assert "IP" in captured.out

    def test_no_features_or_positions(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error_decomposition_report(_make_error_decomposition_report(with_features=False))
        captured = capsys.readouterr()
        assert "Distinguishing Features" not in captured.out


class TestPrintFeatureGapReport:
    def test_shows_in_model_and_not_in_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = FeatureGapReport(
            target="avg",
            player_type="batter",
            season=2024,
            system="fbm",
            version="1.0",
            gaps=[
                FeatureGap("sprint_speed", 0.25, 0.01, 0.300, 0.260, True),
                FeatureGap("barrel_pct", 0.18, 0.07, 0.100, 0.080, False),
            ],
        )
        print_feature_gap_report(report)
        captured = capsys.readouterr()
        assert "In-Model Features" in captured.out
        assert "Not-In-Model Features" in captured.out
        assert "sprint_speed" in captured.out
        assert "barrel_pct" in captured.out

    def test_empty_gap_section_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = FeatureGapReport(
            target="avg",
            player_type="batter",
            season=2024,
            system="fbm",
            version="1.0",
            gaps=[FeatureGap("x", 0.1, 0.5, 0.3, 0.2, True)],
        )
        print_feature_gap_report(report)
        captured = capsys.readouterr()
        assert "Not-In-Model Features" not in captured.out


def _make_cohort_bias_report(*, significant: bool = True) -> CohortBiasReport:
    return CohortBiasReport(
        target="avg",
        player_type="batter",
        season=2024,
        system="fbm",
        version="1.0",
        dimension="age",
        cohorts=[
            CohortBias("young", 50, 0.020, 0.025, 0.030, significant),
            CohortBias("old", 40, -0.015, 0.020, 0.028, False),
        ],
    )


class TestPrintCohortBiasReport:
    def test_shows_cohorts(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_cohort_bias_report(_make_cohort_bias_report())
        captured = capsys.readouterr()
        assert "young" in captured.out
        assert "old" in captured.out
        assert "age" in captured.out


class TestPrintCohortBiasSummary:
    def test_with_significant_cohorts(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_cohort_bias_summary([_make_cohort_bias_report(significant=True)])
        captured = capsys.readouterr()
        assert "Most Biased Cohorts" in captured.out
        assert "young" in captured.out

    def test_no_significant_cohorts(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_cohort_bias_summary([_make_cohort_bias_report(significant=False)])
        captured = capsys.readouterr()
        assert "No statistically significant" in captured.out


# ---------------------------------------------------------------------------
# _validate.py tests
# ---------------------------------------------------------------------------


def _make_preflight_result(confidence: str = "high") -> PreflightResult:
    return PreflightResult(
        details=(
            TargetPreflightDetail("avg", 0.80, -0.002, 0.001),
            TargetPreflightDetail("hr", 0.55, 0.005, 0.003),
        ),
        confidence=confidence,
        recommendation="proceed" if confidence == "high" else "skip",
    )


class TestPrintPreflightResult:
    def test_shows_details_and_confidence(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_preflight_result(_make_preflight_result())
        captured = capsys.readouterr()
        assert "Pre-flight Check" in captured.out
        assert "avg" in captured.out
        assert "hr" in captured.out
        assert "high" in captured.out
        assert "proceed" in captured.out

    def test_low_confidence(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_preflight_result(_make_preflight_result("low"))
        captured = capsys.readouterr()
        assert "low" in captured.out
        assert "skip" in captured.out


def _make_gate_validation_result(*, passed: bool = True, low_preflight: bool = False) -> GateValidationResult:
    check = RegressionCheckResult(
        passed=passed,
        rmse_passed=passed,
        rank_correlation_passed=passed,
        explanation="PASS" if passed else "FAIL: regression",
    )
    preflight = _make_preflight_result("low") if low_preflight else None
    return GateValidationResult(
        model_name="statcast-gbm",
        old_version="1.0",
        new_version="2.0",
        segments=[
            ValidationSegmentResult(season=2023, segment="top300", check=check),
            ValidationSegmentResult(season=2024, segment="top300", check=check),
        ],
        preflight=preflight,
    )


class TestPrintValidationResult:
    def test_passing_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_validation_result(_make_gate_validation_result(passed=True))
        captured = capsys.readouterr()
        assert "Validation Gate" in captured.out
        assert "statcast-gbm" in captured.out
        assert "1.0" in captured.out
        assert "2.0" in captured.out
        assert "OVERALL: PASS" in captured.out

    def test_failing_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_validation_result(_make_gate_validation_result(passed=False))
        captured = capsys.readouterr()
        assert "OVERALL: FAIL" in captured.out

    def test_low_preflight_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_validation_result(_make_gate_validation_result(low_preflight=True))
        captured = capsys.readouterr()
        assert "LOW" in captured.out


# ---------------------------------------------------------------------------
# _draft.py tests
# ---------------------------------------------------------------------------


def _make_league_settings() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=(
            CategoryConfig("hr", "Home Runs", StatType.COUNTING, Direction.HIGHER),
            CategoryConfig("sb", "Stolen Bases", StatType.COUNTING, Direction.HIGHER),
        ),
        pitching_categories=(CategoryConfig("w", "Wins", StatType.COUNTING, Direction.HIGHER),),
        positions={"C": 1, "1B": 1, "OF": 3},
        pitcher_positions={"SP": 5, "RP": 2},
    )


class TestPrintDraftReport:
    def test_full_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = DraftReport(
            total_value=250.0,
            optimal_value=280.0,
            value_efficiency=0.893,
            budget=260,
            total_spent=245,
            category_standings=[
                CategoryStanding("hr", 12.5, 3, 12),
                CategoryStanding("sb", -2.0, 8, 12),
            ],
            pick_grades=[
                PickGrade(1, 100, "Player A", "OF", 40.0, 42.0, 0.95),
                PickGrade(2, 200, "Player B", "SP", 30.0, 35.0, 0.65),
            ],
            mean_grade=0.80,
            steals=[StealOrReach(3, 300, "Steal Guy", "1B", 25.0, 15)],
            reaches=[StealOrReach(4, 400, "Reach Guy", "SS", 10.0, -10)],
        )
        print_draft_report(report)
        captured = capsys.readouterr()
        assert "Draft Report" in captured.out
        assert "250.0" in captured.out
        assert "89.3%" in captured.out
        assert "$260" in captured.out
        assert "Category Standings" in captured.out
        assert "hr" in captured.out
        assert "Pick Grades" in captured.out
        assert "Player A" in captured.out
        assert "0.95" in captured.out
        assert "Steals" in captured.out
        assert "Steal Guy" in captured.out
        assert "Reaches" in captured.out
        assert "Reach Guy" in captured.out


class TestPrintDraftBoardWithAge:
    def test_age_and_bats_throws_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Player A",
                rank=1,
                player_type="batter",
                position="OF",
                value=40.0,
                category_z_scores={},
                age=28,
                bats_throws="R/R",
            ),
        ]
        board = DraftBoard(rows=rows, batting_categories=("hr",), pitching_categories=("w",))
        print_draft_board(board)
        captured = capsys.readouterr()
        assert "Age" in captured.out
        assert "28" in captured.out
        assert "B/T" in captured.out
        assert "R/R" in captured.out


class TestPrintDraftTiersWithADP:
    def test_adp_column_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        tiers = [
            PlayerTier(player_id=1, player_name="Player A", position="OF", tier=1, value=40.0, rank=1),
            PlayerTier(player_id=2, player_name="Player B", position="OF", tier=2, value=30.0, rank=2),
        ]
        adp_map = {
            1: ADP(player_id=1, season=2025, provider="espn", overall_pick=5.0, rank=5, positions="OF"),
        }
        print_draft_tiers(tiers, adp_by_player=adp_map)
        captured = capsys.readouterr()
        assert "Player A" in captured.out
        assert "Player B" in captured.out
        assert "ADP" in captured.out
        assert "5.0" in captured.out


class TestPrintCategoryNeedsWithTradeoffs:
    def test_tradeoff_categories_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        needs = [
            CategoryNeed(
                category="sb",
                current_rank=10,
                target_rank=5,
                best_available=(PlayerRecommendation(1, "Fast Guy", 2.5, ("hr", "rbi")),),
            ),
        ]
        print_category_needs(needs, num_teams=12)
        captured = capsys.readouterr()
        assert "SB" in captured.out
        assert "Fast Guy" in captured.out
        assert "hr, rbi" in captured.out

    def test_no_available_players(self, capsys: pytest.CaptureFixture[str]) -> None:
        needs = [CategoryNeed("sb", 10, 5, ())]
        print_category_needs(needs, num_teams=12)
        captured = capsys.readouterr()
        assert "No available players" in captured.out


class TestPrintPickTradeEvaluationEven:
    def test_even_recommendation(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = PickTradeEvaluation(
            trade=PickTrade(gives=[5], receives=[6]),
            gives_value=20.0,
            receives_value=20.0,
            net_value=0.0,
            gives_detail=[DomainPickValue(5, 20.0, "Player X", "high")],
            receives_detail=[DomainPickValue(6, 20.0, "Player Y", "high")],
            recommendation="even",
        )
        print_pick_trade_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "Even" in captured.out


class TestPrintCascadeResultEven:
    def test_even_recommendation(self, capsys: pytest.CaptureFixture[str]) -> None:
        pick = DraftPick(round=1, pick=1, team_idx=0, player_id=1, player_name="P1", position="OF", value=20.0)
        roster = CascadeRoster(picks=[pick], total_value=20.0)
        result = CascadeResult(
            trade=PickTrade(gives=[1], receives=[2]),
            before=roster,
            after=roster,
            value_delta=0.0,
            recommendation="even",
        )
        print_cascade_result(result)
        captured = capsys.readouterr()
        assert "Even" in captured.out


class TestPrintScarcityReport:
    def test_shows_positions_and_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        scarcities = [
            PositionScarcity("C", 25.0, 5.0, 80.0, -2.0, 1),
            PositionScarcity("OF", 40.0, 10.0, 200.0, -0.5, None),
        ]
        print_scarcity_report(scarcities, _make_league_settings())
        captured = capsys.readouterr()
        assert "Positional Scarcity Report" in captured.out
        assert "C" in captured.out
        assert "OF" in captured.out
        assert "$25.0" in captured.out
        assert "$5.0" in captured.out


class TestPrintValueCurve:
    def test_with_cliff(self, capsys: pytest.CaptureFixture[str]) -> None:
        curve = PositionValueCurve(
            position="C",
            values=[(1, "Catcher A", 30.0), (2, "Catcher B", 25.0), (3, "Catcher C", 10.0)],
            cliff_rank=2,
        )
        print_value_curve(curve, _make_league_settings())
        captured = capsys.readouterr()
        assert "C" in captured.out
        assert "Catcher A" in captured.out
        assert "Cliff at rank 2" in captured.out

    def test_no_cliff(self, capsys: pytest.CaptureFixture[str]) -> None:
        curve = PositionValueCurve(
            position="OF",
            values=[(1, "OF A", 30.0), (2, "OF B", 28.0)],
            cliff_rank=None,
        )
        print_value_curve(curve, _make_league_settings())
        captured = capsys.readouterr()
        assert "No significant cliff" in captured.out


class TestPrintScarcityRankings:
    def test_shows_players_with_deltas(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        players = [
            ScarcityAdjustedPlayer(1, "Player A", "C", "batter", 30.0, 35.0, 5, 2, 0.8),
            ScarcityAdjustedPlayer(2, "Player B", "OF", "batter", 28.0, 27.0, 3, 4, 0.2),
            ScarcityAdjustedPlayer(3, "Player C", "1B", "batter", 25.0, 25.0, 4, 4, 0.5),
        ]
        print_scarcity_rankings(players, _make_league_settings())
        captured = capsys.readouterr()
        assert "Scarcity-Adjusted Rankings" in captured.out
        assert "Player A" in captured.out
        assert "Player B" in captured.out
        assert "+3" in captured.out  # delta for Player A: 5-2=3

    def test_empty_rankings(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_scarcity_rankings([], _make_league_settings())
        captured = capsys.readouterr()
        assert "No scarcity ranking data" in captured.out


# ---------------------------------------------------------------------------
# _experiments.py tests
# ---------------------------------------------------------------------------


def _make_target_result(rmse: float = 0.05, baseline: float = 0.06) -> TargetResult:
    delta = rmse - baseline
    return TargetResult(rmse=rmse, baseline_rmse=baseline, delta=delta, delta_pct=delta / baseline * 100)


def _make_experiment(
    *,
    experiment_id: int = 1,
    target_results: dict[str, TargetResult] | None = None,
    parent_id: int | None = None,
) -> Experiment:
    if target_results is None:
        target_results = {"avg": _make_target_result()}
    return Experiment(
        id=experiment_id,
        timestamp="2025-03-01T12:00:00",
        hypothesis="Adding sprint_speed improves batting avg prediction",
        model="statcast-gbm",
        player_type="batter",
        feature_diff={"added": ["sprint_speed"], "removed": []},
        seasons={"train": [2021, 2022], "holdout": [2023]},
        params={"max_depth": 6},
        target_results=target_results,
        conclusion="Improved avg RMSE",
        tags=["sprint", "batting"],
        parent_id=parent_id,
    )


class TestPrintExperimentSearchResults:
    def test_shows_experiments(self, capsys: pytest.CaptureFixture[str]) -> None:
        experiments = [_make_experiment(), _make_experiment(experiment_id=2)]
        print_experiment_search_results(experiments)
        captured = capsys.readouterr()
        assert "Experiment Search Results" in captured.out
        assert "sprint_speed" in captured.out

    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_experiment_search_results([])
        captured = capsys.readouterr()
        assert "No experiments found" in captured.out

    def test_target_filter(self, capsys: pytest.CaptureFixture[str]) -> None:
        experiments = [_make_experiment()]
        print_experiment_search_results(experiments, target="avg")
        captured = capsys.readouterr()
        assert "avg" not in captured.out or "Experiment" in captured.out  # delta shown for avg

    def test_no_target_results_shows_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        exp = _make_experiment(target_results={})
        print_experiment_search_results([exp])
        captured = capsys.readouterr()
        assert "+0.00%" in captured.out


class TestPrintExperimentDetail:
    def test_shows_all_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_experiment_detail(_make_experiment(parent_id=42))
        captured = capsys.readouterr()
        assert "Experiment #1" in captured.out
        assert "statcast-gbm" in captured.out
        assert "sprint_speed" in captured.out
        assert "Improved avg" in captured.out
        assert "42" in captured.out  # parent_id
        assert "avg" in captured.out

    def test_no_target_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_experiment_detail(_make_experiment(target_results={}))
        captured = capsys.readouterr()
        assert "Experiment #1" in captured.out
        assert "Target Results" not in captured.out


class TestPrintExperimentSummary:
    def test_shows_features_and_targets(self, capsys: pytest.CaptureFixture[str]) -> None:
        summary = ExplorationSummary(
            model="statcast-gbm",
            player_type="batter",
            total_experiments=5,
            features_tested=[
                FeatureExplorationResult("sprint_speed", -1.5, 1, 3),
            ],
            targets_explored=[
                TargetExplorationResult("avg", 0.0450, -1.5, 1, 5),
            ],
            best_experiment_id=1,
            best_experiment_delta_pct=-1.5,
        )
        print_experiment_summary(summary)
        captured = capsys.readouterr()
        assert "Exploration Summary" in captured.out
        assert "sprint_speed" in captured.out
        assert "avg" in captured.out
        assert "#1" in captured.out


class TestPrintCheckpointList:
    def test_shows_checkpoints(self, capsys: pytest.CaptureFixture[str]) -> None:
        checkpoints = [
            FeatureCheckpoint(
                name="baseline",
                model="statcast-gbm",
                player_type="batter",
                feature_columns=["hr", "avg"],
                params={"max_depth": 6},
                target_results={},
                experiment_id=1,
                created_at="2025-03-01",
            ),
        ]
        print_checkpoint_list(checkpoints)
        captured = capsys.readouterr()
        assert "baseline" in captured.out
        assert "statcast-gbm" in captured.out

    def test_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_checkpoint_list([])
        captured = capsys.readouterr()
        assert "No checkpoints found" in captured.out


class TestPrintCheckpointDetail:
    def test_with_target_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        cp = FeatureCheckpoint(
            name="v2",
            model="statcast-gbm",
            player_type="batter",
            feature_columns=["hr", "avg", "sprint_speed"],
            params={"max_depth": 6},
            target_results={"avg": _make_target_result()},
            experiment_id=5,
            created_at="2025-03-01",
            notes="Added sprint speed",
        )
        print_checkpoint_detail(cp)
        captured = capsys.readouterr()
        assert "Checkpoint: v2" in captured.out
        assert "Added sprint speed" in captured.out
        assert "Target Results" in captured.out
        assert "avg" in captured.out

    def test_without_target_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        cp = FeatureCheckpoint(
            name="v1",
            model="m",
            player_type="batter",
            feature_columns=["a"],
            params={},
            target_results={},
            experiment_id=1,
            created_at="2025-01-01",
        )
        print_checkpoint_detail(cp)
        captured = capsys.readouterr()
        assert "Checkpoint: v1" in captured.out
        assert "Target Results" not in captured.out


class TestPrintCompareFeaturesResult:
    def test_shows_deltas(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = FeatureSetComparisonResult(
            columns_a=("hr", "avg"),
            columns_b=("hr", "avg", "sprint_speed"),
            deltas=(
                TargetDelta("avg", 0.0500, 0.0480, -0.002, -4.0),
                TargetDelta("hr", 0.1000, 0.1050, 0.005, 5.0),
            ),
            n_improved=1,
            n_total=2,
            avg_delta_pct=0.5,
            n_folds=3,
        )
        print_compare_features_result(result)
        captured = capsys.readouterr()
        assert "avg" in captured.out
        assert "hr" in captured.out
        assert "1/2 targets" in captured.out
        assert "3 folds" in captured.out


# ---------------------------------------------------------------------------
# _evaluation.py tests
# ---------------------------------------------------------------------------


class TestPrintGateResult:
    def test_passing_gate(self, capsys: pytest.CaptureFixture[str]) -> None:
        check = RegressionCheckResult(passed=True, rmse_passed=True, rank_correlation_passed=True, explanation="ok")
        result = GateResult(
            model_name="statcast-gbm",
            baseline="1.0",
            segments=[
                GateSegmentResult(season=2023, segment="top300", check=check),
            ],
        )
        print_gate_result(result)
        captured = capsys.readouterr()
        assert "Regression Gate" in captured.out
        assert "statcast-gbm" in captured.out
        assert "OVERALL: PASS" in captured.out

    def test_failing_gate(self, capsys: pytest.CaptureFixture[str]) -> None:
        check = RegressionCheckResult(passed=False, rmse_passed=False, rank_correlation_passed=True, explanation="bad")
        result = GateResult(
            model_name="statcast-gbm",
            baseline="1.0",
            segments=[
                GateSegmentResult(season=2023, segment="top300", check=check),
                GateSegmentResult(season=2024, segment="top300", check=check),
            ],
        )
        print_gate_result(result)
        captured = capsys.readouterr()
        assert "OVERALL: FAIL" in captured.out
        assert "2 check(s) failed" in captured.out


class TestPrintTailAccuracyWithData:
    def test_tail_section_with_two_systems(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        tail = TailAccuracy(ns=(25, 50), rmse_by_stat={"hr": {25: 5.12, 50: 4.80}})
        sys_a = SystemMetrics(
            system="sys_a",
            version="1.0",
            source_type="ml",
            metrics={"hr": _make_stat_metrics()},
            tail=tail,
        )
        sys_b = SystemMetrics(
            system="sys_b",
            version="2.0",
            source_type="ml",
            metrics={"hr": _make_stat_metrics()},
            tail=tail,
        )
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "top-25" in captured.out
        assert "top-50" in captured.out
        assert "5.1200" in captured.out


# ---------------------------------------------------------------------------
# _keeper.py tests
# ---------------------------------------------------------------------------


class TestPrintKeeperDecisions:
    def test_shows_decisions(self, capsys: pytest.CaptureFixture[str]) -> None:
        decisions = [
            _make_keeper_decision(recommendation="keep"),
            _make_keeper_decision(
                player_id=2,
                player_name="Bad Player",
                position="util",
                cost=20.0,
                projected_value=10.0,
                surplus=-10.0,
                recommendation="release",
            ),
        ]
        print_keeper_decisions(decisions)
        captured = capsys.readouterr()
        assert "Keeper Decisions" in captured.out
        assert "Mike Trout" in captured.out
        assert "Bad Player" in captured.out
        assert "keep" in captured.out
        assert "release" in captured.out


class TestPrintAdjustedRankings:
    def test_various_value_changes(self, capsys: pytest.CaptureFixture[str]) -> None:
        rankings = [
            AdjustedValuation(1, "Big Up", "batter", "OF", 30.0, 35.0, 5.0),
            AdjustedValuation(2, "Small Up", "batter", "1B", 25.0, 27.0, 2.0),
            AdjustedValuation(3, "Small Down", "batter", "SS", 20.0, 18.0, -2.0),
            AdjustedValuation(4, "No Change", "batter", "C", 15.0, 15.0, 0.0),
            AdjustedValuation(5, "Big Down", "pitcher", "SP", 22.0, 17.0, -5.0),
        ]
        print_adjusted_rankings(rankings, top=3)
        captured = capsys.readouterr()
        assert "Keeper-Adjusted Rankings" in captured.out
        assert "Big Up" in captured.out
        assert "Small Up" in captured.out
        assert "Small Down" in captured.out
        # top=3 should exclude "No Change" and "Big Down"
        assert "No Change" not in captured.out


class TestPrintTradeEvaluation:
    def test_team_a_wins(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = TradeEvaluation(
            team_a_gives=[
                TradePlayerDetail(1, "Given Player", "OF", 15.0, 20.0, 5.0, 2),
            ],
            team_b_gives=[
                TradePlayerDetail(2, "Received Player", "SP", 10.0, 25.0, 15.0, 1),
            ],
            team_a_surplus_delta=10.0,
            team_b_surplus_delta=-10.0,
            winner="team_a",
        )
        print_trade_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "You win this trade" in captured.out

    def test_even_trade(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = TradeEvaluation(
            team_a_gives=[TradePlayerDetail(1, "P1", "OF", 10.0, 20.0, 10.0, 1)],
            team_b_gives=[TradePlayerDetail(2, "P2", "SP", 10.0, 20.0, 10.0, 1)],
            team_a_surplus_delta=0.0,
            team_b_surplus_delta=0.0,
            winner="even",
        )
        print_trade_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "Even trade" in captured.out

    def test_team_b_wins(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = TradeEvaluation(
            team_a_gives=[TradePlayerDetail(1, "P1", "OF", 10.0, 30.0, 20.0, 1)],
            team_b_gives=[TradePlayerDetail(2, "P2", "SP", 10.0, 15.0, 5.0, 1)],
            team_a_surplus_delta=-15.0,
            team_b_surplus_delta=15.0,
            winner="team_b",
        )
        print_trade_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "They win this trade" in captured.out


class TestPrintKeeperSolutionEdgeCases:
    def test_no_alternatives_no_sensitivity(self, capsys: pytest.CaptureFixture[str]) -> None:
        solution = KeeperSolution(
            optimal=_make_keeper_set(),
            alternatives=[],
            sensitivity=[],
        )
        print_keeper_solution(solution)
        captured = capsys.readouterr()
        assert "Optimal Keeper Set" in captured.out
        assert "Alternatives" not in captured.out
        assert "Sensitivity" not in captured.out


# ---------------------------------------------------------------------------
# Minor gap fills
# ---------------------------------------------------------------------------


class TestPrintTuneResultNoDivergence:
    """_model.py line 139: per_target_best with no divergence skips section."""

    def test_no_divergence_skips_per_target(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TuneResult(
            model_name="xgb-v1",
            batter_params={"max_depth": 4},
            pitcher_params={"max_depth": 3},
            batter_cv_rmse={"hr": 0.05},
            pitcher_cv_rmse={"era": 0.50},
            pitcher_per_target_best={
                "era": PerTargetBest(
                    target="era",
                    best_params={"max_depth": 3},
                    best_rmse=0.50,
                    joint_rmse=0.50,
                    delta_pct=0.0,
                ),
            },
        )
        print_tune_result(result)
        captured = capsys.readouterr()
        assert "per-target optimal" not in captured.out


class TestPrintCandidateValuesNull:
    """_feature_factory.py lines 22-23: NULL value handling."""

    def test_null_values_counted(self, capsys: pytest.CaptureFixture[str]) -> None:
        values = [
            CandidateValue(player_id=1, season=2024, value=None),
            CandidateValue(player_id=2, season=2024, value=0.350),
        ]
        print_candidate_values(values)
        captured = capsys.readouterr()
        assert "NULL" in captured.out
        assert "1 with NULL values" in captured.out


class TestPrintBinTargetMeansMissingEntry:
    """_feature_factory.py line 96: missing bin-target entry shows dash."""

    def test_missing_entry_shows_dash(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Two bins, two targets, but only 3 of 4 combinations present
        means = [
            BinTargetMean("low", "avg", 0.260, 50),
            BinTargetMean("low", "hr", 15.0, 50),
            BinTargetMean("high", "avg", 0.300, 50),
            # missing: ("high", "hr")
        ]
        print_bin_target_means(means)
        captured = capsys.readouterr()
        assert "-" in captured.out


class TestPrintBatchSimulationTeamIdx:
    """_mock_draft.py line 73: team_idx is not None."""

    def test_team_idx_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = BatchSimulationResult(
            summary=SimulationSummary(
                n_simulations=100,
                team_idx=2,
                avg_roster_value=150.0,
                median_roster_value=148.0,
                p10_roster_value=130.0,
                p25_roster_value=140.0,
                p75_roster_value=160.0,
                p90_roster_value=170.0,
            ),
            player_frequencies=[],
            strategy_comparisons=[],
            user_rosters=[],
            user_roster_values=[],
            all_player_picks={},
        )
        print_batch_simulation_result(result)
        captured = capsys.readouterr()
        assert "Position" in captured.out
        assert "3" in captured.out  # team_idx + 1


class TestPrintTalentDeltaReportEmpty:
    """_reports.py lines 64-65: empty deltas."""

    def test_empty_deltas(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_talent_delta_report("Breakouts", [])
        captured = capsys.readouterr()
        assert "No results found" in captured.out


class TestPrintPlayerValuationsNoCategories:
    """_valuations.py branch 23->16: empty category_scores."""

    def test_empty_category_scores(self, capsys: pytest.CaptureFixture[str]) -> None:
        val = _make_player_valuation(category_scores={})
        print_player_valuations([val])
        captured = capsys.readouterr()
        assert "Juan Soto" in captured.out
        assert "42.5" in captured.out


class TestPrintUpgrades:
    def test_shows_players(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        values = [
            MarginalValue(1, "Player A", "OF", 30.0, 28.0, {"hr": 1.0}, True, None),
            MarginalValue(2, "Player B", "SS", 25.0, 20.0, {"hr": 0.5}, False, "Player C"),
        ]
        print_upgrades(values)
        captured = capsys.readouterr()
        assert "Player A" in captured.out
        assert "Player B" in captured.out
        assert "Marginal Value" in captured.out
        assert "Fills Need" in captured.out
        assert "Yes" in captured.out
        assert "Player C" in captured.out

    def test_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_upgrades([])
        captured = capsys.readouterr()
        assert "No upgrade data" in captured.out

    def test_with_opportunity_costs(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        values = [MarginalValue(1, "Player A", "OF", 30.0, 28.0, {}, True)]
        costs = [OpportunityCost("OF", "Player A", 28.0, 10.0, 18.0, "draft now")]
        print_upgrades(values, opportunity_costs=costs)
        captured = capsys.readouterr()
        assert "Player A" in captured.out
        assert "Recommendation" in captured.out
        assert "draft now" in captured.out


class TestPrintOpportunityCosts:
    def test_shows_recommendations(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        costs = [
            OpportunityCost("OF", "Player A", 28.0, 10.0, 18.0, "draft now"),
            OpportunityCost("SS", "Player B", 20.0, 25.0, -5.0, "wait"),
        ]
        print_opportunity_costs(costs)
        captured = capsys.readouterr()
        assert "Player A" in captured.out
        assert "draft now" in captured.out
        assert "wait" in captured.out
        assert "Opp Cost" in captured.out

    def test_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_opportunity_costs([])
        captured = capsys.readouterr()
        assert "No opportunity cost data" in captured.out


class TestPrintPositionCheck:
    def test_shows_positions(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=300))
        upgrades = [
            PositionUpgrade("OF", None, 0.0, "Player A", 30.0, 30.0, "Player B", 10.0, "high"),
            PositionUpgrade("SS", "Player C", 15.0, "Player D", 20.0, 5.0, None, 20.0, "low"),
        ]
        print_position_check(upgrades)
        captured = capsys.readouterr()
        assert "OF" in captured.out
        assert "SS" in captured.out
        assert "Player A" in captured.out
        assert "high" in captured.out
        assert "low" in captured.out
        assert "Urgency" in captured.out

    def test_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_position_check([])
        captured = capsys.readouterr()
        assert "No position check data" in captured.out


class TestPrintBreakoutCandidates:
    def test_prints_candidates(self, capsys: pytest.CaptureFixture[str]) -> None:
        predictions = [
            BreakoutPrediction(
                player_id=1,
                player_name="Mike Trout",
                player_type="batter",
                position="OF",
                p_breakout=0.65,
                p_bust=0.10,
                p_neutral=0.25,
                top_features=[("exit_velo", 0.3), ("age", 0.2), ("sprint_speed", 0.1)],
            ),
            BreakoutPrediction(
                player_id=2,
                player_name="Aaron Judge",
                player_type="batter",
                position="OF",
                p_breakout=0.45,
                p_bust=0.20,
                p_neutral=0.35,
                top_features=[("barrel_rate", 0.4)],
            ),
        ]
        print_breakout_candidates(predictions)
        captured = capsys.readouterr()
        assert "Breakout Candidates" in captured.out
        assert "Mike Trout" in captured.out
        assert "Aaron Judge" in captured.out
        assert "0.650" in captured.out
        assert "exit_velo" in captured.out

    def test_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_breakout_candidates([])
        captured = capsys.readouterr()
        assert "No candidates found" in captured.out

    def test_custom_title(self, capsys: pytest.CaptureFixture[str]) -> None:
        predictions = [
            BreakoutPrediction(
                player_id=1,
                player_name="Test",
                player_type="batter",
                position="1B",
                p_breakout=0.1,
                p_bust=0.8,
                p_neutral=0.1,
            ),
        ]
        print_breakout_candidates(predictions, title="Bust Risks")
        captured = capsys.readouterr()
        assert "Bust Risks" in captured.out


class TestPrintClassifierEvaluation:
    def test_prints_full_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = ClassifierEvaluation(
            threshold_metrics=[
                ThresholdMetrics(
                    label="breakout",
                    threshold=0.3,
                    precision=0.45,
                    recall=0.60,
                    f1=0.51,
                    flagged=20,
                    true_positives=9,
                ),
            ],
            calibration_bins=[
                ClassifierCalibrationBin(
                    bin_center=0.25,
                    mean_predicted=0.24,
                    mean_actual=0.20,
                    count=50,
                ),
            ],
            lift_results=[
                LiftResult(
                    label="breakout",
                    top_n=20,
                    flagged_rate=0.35,
                    base_rate=0.17,
                    lift=2.06,
                ),
            ],
            log_loss=0.85,
            base_rate_log_loss=1.05,
            n_evaluated=200,
        )
        print_classifier_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "Classifier Evaluation" in captured.out
        assert "200 players" in captured.out
        assert "0.8500" in captured.out
        assert "1.0500" in captured.out
        assert "Threshold Metrics" in captured.out
        assert "breakout" in captured.out
        assert "Lift Results" in captured.out
        assert "2.06" in captured.out
        assert "Calibration Bins" in captured.out

    def test_empty_evaluation(self, capsys: pytest.CaptureFixture[str]) -> None:
        evaluation = ClassifierEvaluation(
            threshold_metrics=[],
            calibration_bins=[],
            lift_results=[],
            log_loss=0.0,
            base_rate_log_loss=0.0,
            n_evaluated=0,
        )
        print_classifier_evaluation(evaluation)
        captured = capsys.readouterr()
        assert "0 players" in captured.out


class TestPrintRunInspect:
    def _make_record(self) -> ModelRunRecord:
        return ModelRunRecord(
            system="statcast-gbm",
            version="2026.1",
            operation="train",
            config_json={
                "seasons": [2022, 2023, 2024],
                "model_params": {"n_estimators": 500, "learning_rate": 0.05},
                "feature_set": "v3",
            },
            metrics_json={"rmse": 0.1234, "r_squared": 0.8765},
            artifact_type="file",
            created_at="2026-03-01T12:00:00",
            git_commit="abc1234",
            tags_json={"experiment": "baseline"},
        )

    def test_output_contains_section_headers(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=200))
        print_run_inspect(self._make_record())
        captured = capsys.readouterr()
        assert "Run Info" in captured.out
        assert "Config" in captured.out
        assert "Metrics" in captured.out
        assert "Tags" in captured.out

    def test_section_filter_limits_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=200))
        print_run_inspect(self._make_record(), section="metrics")
        captured = capsys.readouterr()
        assert "rmse" in captured.out
        assert "Run Info" not in captured.out
        assert "Tags" not in captured.out


class TestDiffRecords:
    def _make_record(self, **overrides: Any) -> ModelRunRecord:
        defaults: dict[str, Any] = {
            "system": "statcast-gbm",
            "version": "2026.1",
            "operation": "train",
            "config_json": {"seasons": [2022, 2023], "model_params": {"n_estimators": 500}},
            "metrics_json": {"rmse": 0.15, "r_squared": 0.85},
            "artifact_type": "file",
            "created_at": "2026-03-01T12:00:00",
        }
        defaults.update(overrides)
        return ModelRunRecord(**defaults)

    def test_added_removed_changed_keys(self) -> None:
        a = self._make_record(config_json={"alpha": 1, "beta": 2})
        b = self._make_record(version="2026.2", config_json={"beta": 3, "gamma": 4})
        diff = diff_records(a, b)
        assert diff["config"]["added"] == {"gamma": 4}
        assert diff["config"]["removed"] == {"alpha": 1}
        assert "beta" in diff["config"]["changed"]
        assert diff["config"]["changed"]["beta"]["old"] == 2
        assert diff["config"]["changed"]["beta"]["new"] == 3

    def test_nested_dict_diffs(self) -> None:
        a = self._make_record(config_json={"model_params": {"n_estimators": 500}})
        b = self._make_record(version="2026.2", config_json={"model_params": {"n_estimators": 1000}})
        diff = diff_records(a, b)
        assert "model_params" in diff["config"]["changed"]

    def test_metric_deltas_computed(self) -> None:
        a = self._make_record(metrics_json={"rmse": 0.15, "r_squared": 0.85})
        b = self._make_record(version="2026.2", metrics_json={"rmse": 0.12, "r_squared": 0.88})
        diff = diff_records(a, b)
        rmse_change = diff["metrics"]["changed"]["rmse"]
        assert abs(rmse_change["delta"] - (-0.03)) < 1e-9
        r2_change = diff["metrics"]["changed"]["r_squared"]
        assert abs(r2_change["delta"] - 0.03) < 1e-9

    def test_print_run_diff_output(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(_output, "console", Console(highlight=False, width=200))
        a = self._make_record(config_json={"alpha": 1}, metrics_json={"rmse": 0.15})
        b = self._make_record(version="2026.2", config_json={"alpha": 2}, metrics_json={"rmse": 0.12})
        print_run_diff(a, b)
        captured = capsys.readouterr()
        assert "Diff:" in captured.out
        assert "changed" in captured.out
