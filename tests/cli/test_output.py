from typing import Any

import pytest

from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_comparison_result,
    print_error,
    print_features,
    print_player_projections,
    print_player_valuations,
    print_stratified_comparison_result,
    print_system_metrics,
    print_valuation_eval_result,
    print_valuation_rankings,
)
from fantasy_baseball_manager.models.protocols import AblationResult
from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StatMetrics,
    StratifiedComparisonResult,
    SystemMetrics,
)
from fantasy_baseball_manager.domain.projection import PlayerProjection
from fantasy_baseball_manager.domain.valuation import PlayerValuation, ValuationAccuracy, ValuationEvalResult
from fantasy_baseball_manager.features.types import (
    DeltaFeature,
    Feature,
    Source,
    TransformFeature,
)


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
