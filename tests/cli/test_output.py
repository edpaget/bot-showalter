from typing import Any

import pytest

from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_adp_accuracy_report,
    print_adp_movers_report,
    print_comparison_result,
    print_dataset_list,
    print_draft_board,
    print_error,
    print_features,
    print_import_result,
    print_ingest_result,
    print_performance_report,
    print_player_projections,
    print_player_valuations,
    print_predict_result,
    print_prepare_result,
    print_projection_confidence,
    print_residual_persistence_report,
    print_run_detail,
    print_run_list,
    print_stratified_comparison_result,
    print_system_metrics,
    print_system_summaries,
    print_talent_quality_report,
    print_train_result,
    print_tune_result,
    print_valuation_eval_result,
    print_valuation_rankings,
    print_value_over_adp,
)
from fantasy_baseball_manager.domain.adp_accuracy import ADPAccuracyReport, ADPAccuracyResult, SystemAccuracyResult
from fantasy_baseball_manager.domain.adp_movers import ADPMover, ADPMoversReport
from fantasy_baseball_manager.domain.adp_report import ValueOverADP, ValueOverADPReport
from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StatMetrics,
    StratifiedComparisonResult,
    SystemMetrics,
)
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
from fantasy_baseball_manager.domain.projection import PlayerProjection, SystemSummary
from fantasy_baseball_manager.domain.projection_confidence import ConfidenceReport, PlayerConfidence, StatSpread
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
from fantasy_baseball_manager.domain.valuation import PlayerValuation, ValuationAccuracy, ValuationEvalResult
from fantasy_baseball_manager.features.types import (
    DeltaFeature,
    DerivedTransformFeature,
    Feature,
    Source,
    TransformFeature,
)
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
    r_squared: float = 0.75,
    n: int = 100,
) -> StatMetrics:
    return StatMetrics(rmse=rmse, mae=mae, correlation=correlation, r_squared=r_squared, n=n)


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
        assert "R²" in captured.out

    def test_r_squared_value_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_system_metrics(_make_system_metrics())
        captured = capsys.readouterr()
        assert "0.7500" in captured.out
        assert "0.6500" in captured.out


class TestPrintComparisonResult:
    def test_comparison_shows_r_squared(self, capsys: pytest.CaptureFixture[str]) -> None:
        sys_a = _make_system_metrics(system="steamer", version="2025", stats={"hr": _make_stat_metrics(r_squared=0.60)})
        sys_b = _make_system_metrics(system="zips", version="2025", stats={"hr": _make_stat_metrics(r_squared=0.45)})
        result = ComparisonResult(season=2024, stats=["hr"], systems=[sys_a, sys_b])
        print_comparison_result(result)
        captured = capsys.readouterr()
        assert "R²" in captured.out
        assert "0.6000" in captured.out
        assert "0.4500" in captured.out


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
