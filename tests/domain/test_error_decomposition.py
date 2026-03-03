import pytest

from fantasy_baseball_manager.domain.error_decomposition import (
    DistinguishingFeature,
    ErrorDecompositionReport,
    MissPopulationSummary,
    PlayerResidual,
    compute_distinguishing_features,
    compute_miss_summary,
    rank_residuals,
    split_direction,
)


def _make_residual(
    *,
    player_id: int = 1,
    player_name: str = "Test Player",
    predicted: float = 0.0,
    actual: float = 0.0,
    residual: float | None = None,
    feature_values: dict[str, float] | None = None,
) -> PlayerResidual:
    if residual is None:
        residual = predicted - actual
    return PlayerResidual(
        player_id=player_id,
        player_name=player_name,
        predicted=predicted,
        actual=actual,
        residual=residual,
        feature_values=feature_values or {},
    )


class TestPlayerResidualConstruction:
    def test_construction(self) -> None:
        pr = _make_residual(player_id=42, predicted=0.300, actual=0.250)
        assert pr.player_id == 42
        assert pr.predicted == 0.300
        assert pr.actual == 0.250
        assert pr.residual == pytest.approx(0.050)

    def test_immutability(self) -> None:
        pr = _make_residual()
        with pytest.raises(AttributeError):
            pr.player_id = 99  # type: ignore[misc]


class TestErrorDecompositionReportConstruction:
    def test_construction(self) -> None:
        report = ErrorDecompositionReport(
            target="slg",
            player_type="batter",
            season=2024,
            system="statcast-gbm",
            version="latest",
            top_misses=[],
            over_predictions=[],
            under_predictions=[],
            summary=MissPopulationSummary(
                mean_age=None,
                position_distribution={},
                mean_volume=0.0,
                distinguishing_features=[],
            ),
        )
        assert report.target == "slg"
        assert report.player_type == "batter"

    def test_immutability(self) -> None:
        report = ErrorDecompositionReport(
            target="slg",
            player_type="batter",
            season=2024,
            system="test",
            version="v1",
            top_misses=[],
            over_predictions=[],
            under_predictions=[],
            summary=MissPopulationSummary(
                mean_age=None,
                position_distribution={},
                mean_volume=0.0,
                distinguishing_features=[],
            ),
        )
        with pytest.raises(AttributeError):
            report.target = "era"  # type: ignore[misc]


class TestRankResiduals:
    def test_ranks_by_absolute_residual_descending(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=0.010),
            _make_residual(player_id=2, residual=-0.050),
            _make_residual(player_id=3, residual=0.030),
            _make_residual(player_id=4, residual=-0.020),
            _make_residual(player_id=5, residual=0.100),
        ]
        top3 = rank_residuals(residuals, top_n=3)
        assert len(top3) == 3
        assert [r.player_id for r in top3] == [5, 2, 3]

    def test_top_n_exceeds_list_length(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=0.010),
            _make_residual(player_id=2, residual=-0.050),
        ]
        result = rank_residuals(residuals, top_n=10)
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert rank_residuals([], top_n=5) == []


class TestSplitDirection:
    def test_splits_over_and_under(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=0.050),
            _make_residual(player_id=2, residual=-0.030),
            _make_residual(player_id=3, residual=0.020),
            _make_residual(player_id=4, residual=-0.010),
            _make_residual(player_id=5, residual=0.0),
        ]
        over, under = split_direction(residuals)
        assert [r.player_id for r in over] == [1, 3]
        assert [r.player_id for r in under] == [2, 4]

    def test_over_sorted_by_residual_desc(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=0.020),
            _make_residual(player_id=2, residual=0.050),
        ]
        over, _ = split_direction(residuals)
        assert [r.player_id for r in over] == [2, 1]

    def test_under_sorted_by_residual_asc(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=-0.020),
            _make_residual(player_id=2, residual=-0.050),
        ]
        _, under = split_direction(residuals)
        assert [r.player_id for r in under] == [2, 1]

    def test_empty_list(self) -> None:
        over, under = split_direction([])
        assert over == []
        assert under == []


class TestComputeDistinguishingFeatures:
    def test_identifies_largest_mean_difference(self) -> None:
        miss_group = [
            _make_residual(player_id=1, feature_values={"age": 35.0, "pa": 500.0}),
            _make_residual(player_id=2, feature_values={"age": 36.0, "pa": 480.0}),
        ]
        rest = [
            _make_residual(player_id=3, feature_values={"age": 27.0, "pa": 520.0}),
            _make_residual(player_id=4, feature_values={"age": 28.0, "pa": 510.0}),
        ]
        features = compute_distinguishing_features(miss_group, rest)
        assert len(features) == 2
        # pa diff = |490 - 515| = 25, age diff = |35.5 - 27.5| = 8
        assert features[0].feature_name == "pa"
        assert features[0].difference == pytest.approx(25.0)
        assert features[1].feature_name == "age"
        assert features[1].difference == pytest.approx(8.0)

    def test_correct_means(self) -> None:
        miss_group = [
            _make_residual(player_id=1, feature_values={"age": 30.0}),
            _make_residual(player_id=2, feature_values={"age": 34.0}),
        ]
        rest = [
            _make_residual(player_id=3, feature_values={"age": 26.0}),
        ]
        features = compute_distinguishing_features(miss_group, rest)
        assert len(features) == 1
        assert features[0].mean_miss_group == pytest.approx(32.0)
        assert features[0].mean_rest == pytest.approx(26.0)
        assert features[0].difference == pytest.approx(6.0)

    def test_empty_miss_group(self) -> None:
        features = compute_distinguishing_features([], [_make_residual(feature_values={"age": 30.0})])
        assert features == []

    def test_empty_rest_group(self) -> None:
        features = compute_distinguishing_features([_make_residual(feature_values={"age": 30.0})], [])
        assert features == []

    def test_skips_features_not_in_both_groups(self) -> None:
        miss_group = [_make_residual(feature_values={"age": 30.0, "speed": 28.0})]
        rest = [_make_residual(feature_values={"age": 25.0})]
        features = compute_distinguishing_features(miss_group, rest)
        assert len(features) == 1
        assert features[0].feature_name == "age"


class TestComputeMissSummary:
    def test_basic_summary(self) -> None:
        misses = [
            _make_residual(player_id=1, feature_values={"age": 32.0, "pa": 500.0}),
            _make_residual(player_id=2, feature_values={"age": 34.0, "pa": 480.0}),
        ]
        rest = [
            _make_residual(player_id=3, feature_values={"age": 26.0, "pa": 550.0}),
        ]
        positions = {1: "1B", 2: "DH"}
        summary = compute_miss_summary(misses, rest, positions)
        assert summary.mean_age == pytest.approx(33.0)
        assert summary.position_distribution == {"1B": 1, "DH": 1}
        assert summary.mean_volume == pytest.approx(490.0)
        assert len(summary.distinguishing_features) > 0

    def test_no_age_in_features(self) -> None:
        misses = [_make_residual(player_id=1, feature_values={"pa": 500.0})]
        rest = [_make_residual(player_id=2, feature_values={"pa": 550.0})]
        summary = compute_miss_summary(misses, rest, {})
        assert summary.mean_age is None

    def test_no_volume_defaults_to_zero(self) -> None:
        misses = [_make_residual(player_id=1, feature_values={"age": 30.0})]
        rest = [_make_residual(player_id=2, feature_values={"age": 25.0})]
        summary = compute_miss_summary(misses, rest, {})
        assert summary.mean_volume == pytest.approx(0.0)

    def test_missing_position_excluded(self) -> None:
        misses = [
            _make_residual(player_id=1, feature_values={}),
            _make_residual(player_id=2, feature_values={}),
        ]
        positions = {1: "SS"}
        summary = compute_miss_summary(misses, [], positions)
        assert summary.position_distribution == {"SS": 1}


class TestDistinguishingFeatureConstruction:
    def test_construction(self) -> None:
        df = DistinguishingFeature(
            feature_name="age",
            mean_miss_group=33.0,
            mean_rest=27.0,
            difference=6.0,
        )
        assert df.feature_name == "age"
        assert df.difference == 6.0

    def test_immutability(self) -> None:
        df = DistinguishingFeature(
            feature_name="age",
            mean_miss_group=33.0,
            mean_rest=27.0,
            difference=6.0,
        )
        with pytest.raises(AttributeError):
            df.feature_name = "pa"  # type: ignore[misc]
