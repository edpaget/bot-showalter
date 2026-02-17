import pytest

from fantasy_baseball_manager.domain.talent_quality import (
    StatTalentMetrics,
    TalentQualitySummary,
    TrueTalentQualityReport,
    compute_predictive_validity,
    compute_r_squared_with_buckets,
    compute_residual_yoy_correlation,
    compute_shrinkage,
)


class TestStatTalentMetricsConstruction:
    def test_frozen_dataclass(self) -> None:
        m = StatTalentMetrics(
            stat_name="avg",
            model_next_season_corr=0.7,
            raw_next_season_corr=0.5,
            predictive_validity_pass=True,
            residual_yoy_corr=0.05,
            residual_non_persistence_pass=True,
            shrinkage_ratio=0.8,
            estimate_raw_corr=0.95,
            shrinkage_pass=True,
            r_squared=0.75,
            residual_by_bucket={"<200": 0.02, "200-400": 0.01, "400+": 0.005},
            r_squared_pass=True,
            regression_rate=0.95,
            regression_rate_pass=True,
            n_season_n=100,
            n_returning=80,
        )
        assert m.stat_name == "avg"
        assert m.model_next_season_corr == 0.7
        assert m.predictive_validity_pass is True
        with pytest.raises(AttributeError):
            m.stat_name = "obp"  # type: ignore[misc]


class TestTalentQualitySummaryConstruction:
    def test_frozen_dataclass(self) -> None:
        s = TalentQualitySummary(
            predictive_validity_passes=6,
            predictive_validity_total=13,
            residual_non_persistence_passes=11,
            residual_non_persistence_total=13,
            shrinkage_passes=8,
            shrinkage_total=13,
            r_squared_passes=7,
            r_squared_total=13,
            regression_rate_passes=10,
            regression_rate_total=13,
        )
        assert s.predictive_validity_passes == 6
        assert s.regression_rate_total == 13
        with pytest.raises(AttributeError):
            s.shrinkage_passes = 0  # type: ignore[misc]


class TestTrueTalentQualityReportConstruction:
    def test_frozen_dataclass(self) -> None:
        summary = TalentQualitySummary(
            predictive_validity_passes=1,
            predictive_validity_total=1,
            residual_non_persistence_passes=1,
            residual_non_persistence_total=1,
            shrinkage_passes=1,
            shrinkage_total=1,
            r_squared_passes=1,
            r_squared_total=1,
            regression_rate_passes=1,
            regression_rate_total=1,
        )
        report = TrueTalentQualityReport(
            system="statcast-gbm",
            version="latest",
            season_n=2024,
            season_n1=2025,
            player_type="batter",
            stat_metrics=[],
            summary=summary,
        )
        assert report.system == "statcast-gbm"
        assert report.season_n == 2024
        assert report.season_n1 == 2025
        with pytest.raises(AttributeError):
            report.system = "other"  # type: ignore[misc]


# --- Pure computation helper tests ---


class TestComputePredictiveValidity:
    def test_model_correlates_better_than_raw(self) -> None:
        # Model estimates that track next-season actuals well
        model_n = [0.280, 0.300, 0.260, 0.310, 0.270]
        raw_n = [0.320, 0.250, 0.290, 0.275, 0.310]
        # Next-season actuals close to model estimates
        actuals_n1 = [0.285, 0.295, 0.265, 0.305, 0.275]

        model_corr, raw_corr = compute_predictive_validity(model_n, raw_n, actuals_n1)
        assert model_corr > raw_corr
        assert model_corr > 0.9

    def test_n_less_than_2_returns_zeros(self) -> None:
        model_corr, raw_corr = compute_predictive_validity([0.3], [0.3], [0.3])
        assert model_corr == 0.0
        assert raw_corr == 0.0

    def test_empty_returns_zeros(self) -> None:
        model_corr, raw_corr = compute_predictive_validity([], [], [])
        assert model_corr == 0.0
        assert raw_corr == 0.0


class TestComputeResidualYoyCorrelation:
    def test_uncorrelated_residuals(self) -> None:
        residuals_n = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.015, 0.005]
        residuals_n1 = [0.005, 0.015, 0.01, -0.03, -0.01, 0.02, -0.02, 0.03]
        corr = compute_residual_yoy_correlation(residuals_n, residuals_n1)
        assert abs(corr) < 0.4

    def test_correlated_residuals(self) -> None:
        residuals_n = [0.01, 0.02, 0.03, 0.04, 0.05]
        residuals_n1 = [0.011, 0.019, 0.031, 0.039, 0.051]
        corr = compute_residual_yoy_correlation(residuals_n, residuals_n1)
        assert corr > 0.9

    def test_n_less_than_2_returns_zero(self) -> None:
        assert compute_residual_yoy_correlation([0.01], [0.02]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_residual_yoy_correlation([], []) == 0.0


class TestComputeShrinkage:
    def test_estimates_lower_variance_than_raw(self) -> None:
        raw = [0.200, 0.250, 0.300, 0.350, 0.400]
        estimates = [0.270, 0.280, 0.300, 0.320, 0.330]
        ratio, corr = compute_shrinkage(estimates, raw)
        assert ratio < 1.0
        assert corr > 0.9

    def test_n_less_than_2_returns_zeros(self) -> None:
        ratio, corr = compute_shrinkage([0.3], [0.3])
        assert ratio == 0.0
        assert corr == 0.0

    def test_zero_raw_variance_returns_zero_ratio(self) -> None:
        ratio, corr = compute_shrinkage([0.29, 0.30, 0.31], [0.30, 0.30, 0.30])
        assert ratio == 0.0


class TestComputeRSquaredWithBuckets:
    def test_perfect_fit(self) -> None:
        estimates = [0.280, 0.300, 0.260]
        actuals = [0.280, 0.300, 0.260]
        sample_sizes = [500.0, 300.0, 100.0]
        bucket_edges = (200.0, 400.0)
        bucket_labels = ("<200", "200-400", "400+")

        r2, buckets = compute_r_squared_with_buckets(estimates, actuals, sample_sizes, bucket_edges, bucket_labels)
        assert r2 == pytest.approx(1.0)
        for label in bucket_labels:
            if label in buckets:
                assert buckets[label] == pytest.approx(0.0)

    def test_bucket_assignment(self) -> None:
        estimates = [0.280, 0.300, 0.260, 0.290, 0.310]
        actuals = [0.290, 0.295, 0.270, 0.285, 0.320]
        sample_sizes = [100.0, 300.0, 500.0, 150.0, 450.0]
        bucket_edges = (200.0, 400.0)
        bucket_labels = ("<200", "200-400", "400+")

        r2, buckets = compute_r_squared_with_buckets(estimates, actuals, sample_sizes, bucket_edges, bucket_labels)
        assert r2 > 0.0
        # Players with sample_size 100 and 150 go in "<200"
        assert "<200" in buckets
        # Player with sample_size 300 goes in "200-400"
        assert "200-400" in buckets
        # Players with sample_size 500 and 450 go in "400+"
        assert "400+" in buckets

    def test_n_less_than_2_returns_zeros(self) -> None:
        r2, buckets = compute_r_squared_with_buckets([0.3], [0.3], [500.0], (200.0, 400.0), ("<200", "200-400", "400+"))
        assert r2 == 0.0
        assert buckets == {}

    def test_constant_actuals_returns_zero_r_squared(self) -> None:
        estimates = [0.280, 0.300, 0.260]
        actuals = [0.290, 0.290, 0.290]
        sample_sizes = [500.0, 300.0, 100.0]
        r2, _ = compute_r_squared_with_buckets(
            estimates, actuals, sample_sizes, (200.0, 400.0), ("<200", "200-400", "400+")
        )
        assert r2 == 0.0
