import pytest

from fantasy_baseball_manager.domain.error_decomposition import (
    CohortBias,
    CohortBiasReport,
    DistinguishingFeature,
    ErrorDecompositionReport,
    FeatureGap,
    FeatureGapReport,
    MissPopulationSummary,
    PlayerResidual,
    bucket_by_age,
    bucket_by_experience,
    bucket_by_handedness,
    bucket_by_position,
    compute_cohort_metrics,
    compute_distinguishing_features,
    compute_miss_summary,
    rank_residuals,
    split_direction,
    split_residuals_by_quality,
)
from fantasy_baseball_manager.domain.identity import PlayerType


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
            player_type=PlayerType.BATTER,
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
            player_type=PlayerType.BATTER,
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


class TestFeatureGapConstruction:
    def test_construction(self) -> None:
        gap = FeatureGap(
            feature_name="barrel_pct",
            ks_statistic=0.85,
            p_value=0.001,
            mean_well=0.08,
            mean_poor=0.15,
            in_model=False,
        )
        assert gap.feature_name == "barrel_pct"
        assert gap.ks_statistic == 0.85
        assert gap.in_model is False

    def test_immutability(self) -> None:
        gap = FeatureGap(
            feature_name="age",
            ks_statistic=0.5,
            p_value=0.05,
            mean_well=27.0,
            mean_poor=34.0,
            in_model=True,
        )
        with pytest.raises(AttributeError):
            gap.feature_name = "pa"  # type: ignore[misc]


class TestFeatureGapReportConstruction:
    def test_construction(self) -> None:
        report = FeatureGapReport(
            target="slg",
            player_type=PlayerType.BATTER,
            season=2024,
            system="test",
            version="v1",
            gaps=[],
        )
        assert report.target == "slg"
        assert report.gaps == []


class TestSplitResidualsByQuality:
    def test_splits_into_well_and_poorly_predicted(self) -> None:
        # Create 10 players with varying residual magnitudes
        residuals = [
            _make_residual(player_id=i, residual=r)
            for i, r in enumerate(
                [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.10, -0.15, 0.20, -0.30],
                start=1,
            )
        ]
        well, poor = split_residuals_by_quality(residuals, miss_percentile=80.0)
        # Median abs residual ≈ 0.055
        # Well-predicted: abs(resid) < median → players 1,2,3,4,5
        assert {r.player_id for r in well} == {1, 2, 3, 4, 5}
        # Poorly-predicted: abs(resid) > P80 threshold
        # All poorly-predicted should have large residuals
        for r in poor:
            assert abs(r.residual) >= 0.20
        # Player 10 (|resid|=0.30) should always be poorly predicted
        assert 10 in {r.player_id for r in poor}

    def test_all_same_residual_returns_empty_groups(self) -> None:
        residuals = [_make_residual(player_id=i, residual=0.05) for i in range(1, 6)]
        well, poor = split_residuals_by_quality(residuals)
        # All residuals the same — none strictly below median, none strictly above P80
        assert well == []
        assert poor == []

    def test_single_player_returns_empty_groups(self) -> None:
        residuals = [_make_residual(player_id=1, residual=0.10)]
        well, poor = split_residuals_by_quality(residuals)
        assert well == []
        assert poor == []

    def test_empty_list_returns_empty_groups(self) -> None:
        well, poor = split_residuals_by_quality([])
        assert well == []
        assert poor == []

    def test_custom_percentile(self) -> None:
        residuals = [
            _make_residual(player_id=i, residual=r)
            for i, r in enumerate(
                [0.01, -0.02, 0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 0.40, -0.50],
                start=1,
            )
        ]
        well, poor = split_residuals_by_quality(residuals, miss_percentile=50.0)
        # P50 = median abs residual, so "poor" = abs(resid) > median
        # More players should be in the poorly-predicted group with a lower percentile
        assert len(poor) > 0
        # All poor should have abs(residual) > all well
        if well and poor:
            max_well = max(abs(r.residual) for r in well)
            min_poor = min(abs(r.residual) for r in poor)
            assert min_poor > max_well


class TestCohortBiasConstruction:
    def test_construction(self) -> None:
        cb = CohortBias(
            cohort_label="22-25",
            n=10,
            mean_residual=0.05,
            mean_abs_residual=0.08,
            rmse=0.10,
            significant=True,
        )
        assert cb.cohort_label == "22-25"
        assert cb.n == 10
        assert cb.significant is True

    def test_immutability(self) -> None:
        cb = CohortBias(
            cohort_label="22-25", n=10, mean_residual=0.05, mean_abs_residual=0.08, rmse=0.10, significant=True
        )
        with pytest.raises(AttributeError):
            cb.cohort_label = "26-29"  # type: ignore[misc]


class TestCohortBiasReportConstruction:
    def test_construction(self) -> None:
        report = CohortBiasReport(
            target="slg",
            player_type=PlayerType.BATTER,
            season=2024,
            system="test",
            version="v1",
            dimension="age",
            cohorts=[],
        )
        assert report.target == "slg"
        assert report.dimension == "age"
        assert report.cohorts == []


class TestBucketByAge:
    def test_groups_into_correct_buckets(self) -> None:
        residuals = [
            _make_residual(player_id=1, feature_values={"age": 22.0}),
            _make_residual(player_id=2, feature_values={"age": 25.0}),
            _make_residual(player_id=3, feature_values={"age": 26.0}),
            _make_residual(player_id=4, feature_values={"age": 29.0}),
            _make_residual(player_id=5, feature_values={"age": 30.0}),
            _make_residual(player_id=6, feature_values={"age": 33.0}),
            _make_residual(player_id=7, feature_values={"age": 34.0}),
            _make_residual(player_id=8, feature_values={"age": 40.0}),
        ]
        buckets = bucket_by_age(residuals)
        assert {r.player_id for r in buckets["22-25"]} == {1, 2}
        assert {r.player_id for r in buckets["26-29"]} == {3, 4}
        assert {r.player_id for r in buckets["30-33"]} == {5, 6}
        assert {r.player_id for r in buckets["34+"]} == {7, 8}

    def test_boundary_25_in_22_25(self) -> None:
        residuals = [_make_residual(player_id=1, feature_values={"age": 25.0})]
        buckets = bucket_by_age(residuals)
        assert len(buckets["22-25"]) == 1

    def test_boundary_26_in_26_29(self) -> None:
        residuals = [_make_residual(player_id=1, feature_values={"age": 26.0})]
        buckets = bucket_by_age(residuals)
        assert len(buckets["26-29"]) == 1

    def test_missing_age_excluded(self) -> None:
        residuals = [
            _make_residual(player_id=1, feature_values={"age": 25.0}),
            _make_residual(player_id=2, feature_values={"pa": 500.0}),
        ]
        buckets = bucket_by_age(residuals)
        total = sum(len(v) for v in buckets.values())
        assert total == 1

    def test_empty_input(self) -> None:
        assert bucket_by_age([]) == {}

    def test_age_below_22_excluded(self) -> None:
        residuals = [_make_residual(player_id=1, feature_values={"age": 20.0})]
        buckets = bucket_by_age(residuals)
        total = sum(len(v) for v in buckets.values())
        assert total == 0


class TestBucketByPosition:
    def test_groups_by_position(self) -> None:
        residuals = [
            _make_residual(player_id=1),
            _make_residual(player_id=2),
            _make_residual(player_id=3),
        ]
        positions = {1: "SS", 2: "SS", 3: "1B"}
        buckets = bucket_by_position(residuals, positions)
        assert len(buckets["SS"]) == 2
        assert len(buckets["1B"]) == 1

    def test_missing_position_excluded(self) -> None:
        residuals = [
            _make_residual(player_id=1),
            _make_residual(player_id=2),
        ]
        positions = {1: "CF"}
        buckets = bucket_by_position(residuals, positions)
        total = sum(len(v) for v in buckets.values())
        assert total == 1

    def test_empty_input(self) -> None:
        assert bucket_by_position([], {}) == {}


class TestBucketByHandedness:
    def test_groups_by_hand(self) -> None:
        residuals = [
            _make_residual(player_id=1),
            _make_residual(player_id=2),
            _make_residual(player_id=3),
        ]
        handedness = {1: "L", 2: "R", 3: "S"}
        buckets = bucket_by_handedness(residuals, handedness)
        assert len(buckets["L"]) == 1
        assert len(buckets["R"]) == 1
        assert len(buckets["S"]) == 1

    def test_missing_handedness_excluded(self) -> None:
        residuals = [_make_residual(player_id=1), _make_residual(player_id=2)]
        handedness = {1: "R"}
        buckets = bucket_by_handedness(residuals, handedness)
        total = sum(len(v) for v in buckets.values())
        assert total == 1

    def test_empty_input(self) -> None:
        assert bucket_by_handedness([], {}) == {}


class TestBucketByExperience:
    def test_groups_into_correct_buckets(self) -> None:
        residuals = [
            _make_residual(player_id=1),
            _make_residual(player_id=2),
            _make_residual(player_id=3),
            _make_residual(player_id=4),
        ]
        experience = {1: 1, 2: 3, 3: 7, 4: 12}
        buckets = bucket_by_experience(residuals, experience)
        assert len(buckets["1-2"]) == 1
        assert len(buckets["3-5"]) == 1
        assert len(buckets["6-10"]) == 1
        assert len(buckets["11+"]) == 1

    def test_boundary_2_in_1_2(self) -> None:
        residuals = [_make_residual(player_id=1)]
        experience = {1: 2}
        buckets = bucket_by_experience(residuals, experience)
        assert len(buckets["1-2"]) == 1

    def test_boundary_3_in_3_5(self) -> None:
        residuals = [_make_residual(player_id=1)]
        experience = {1: 3}
        buckets = bucket_by_experience(residuals, experience)
        assert len(buckets["3-5"]) == 1

    def test_missing_experience_excluded(self) -> None:
        residuals = [_make_residual(player_id=1), _make_residual(player_id=2)]
        experience = {1: 5}
        buckets = bucket_by_experience(residuals, experience)
        total = sum(len(v) for v in buckets.values())
        assert total == 1

    def test_empty_input(self) -> None:
        assert bucket_by_experience([], {}) == {}


class TestComputeCohortMetrics:
    def test_known_values(self) -> None:
        residuals = [
            _make_residual(player_id=1, residual=0.10),
            _make_residual(player_id=2, residual=-0.10),
            _make_residual(player_id=3, residual=0.20),
        ]
        mean_r, mean_abs_r, rmse = compute_cohort_metrics(residuals)
        # mean_residual = (0.10 + -0.10 + 0.20) / 3 ≈ 0.0667
        assert mean_r == pytest.approx(0.2 / 3, abs=1e-6)
        # mean_abs_residual = (0.10 + 0.10 + 0.20) / 3 ≈ 0.1333
        assert mean_abs_r == pytest.approx(0.4 / 3, abs=1e-6)
        # rmse = sqrt((0.01 + 0.01 + 0.04) / 3) = sqrt(0.02) ≈ 0.1414
        assert rmse == pytest.approx((0.06 / 3) ** 0.5, abs=1e-6)

    def test_single_player(self) -> None:
        residuals = [_make_residual(player_id=1, residual=0.05)]
        mean_r, mean_abs_r, rmse = compute_cohort_metrics(residuals)
        assert mean_r == pytest.approx(0.05)
        assert mean_abs_r == pytest.approx(0.05)
        assert rmse == pytest.approx(0.05)

    def test_all_zero_residuals(self) -> None:
        residuals = [_make_residual(player_id=i, residual=0.0) for i in range(1, 4)]
        mean_r, mean_abs_r, rmse = compute_cohort_metrics(residuals)
        assert mean_r == pytest.approx(0.0)
        assert mean_abs_r == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)
