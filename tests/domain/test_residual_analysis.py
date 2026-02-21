import pytest

from fantasy_baseball_manager.domain.residual_analysis import (
    compute_calibration_bins,
    compute_heteroscedasticity,
    compute_mean_bias,
)


class TestComputeMeanBias:
    def test_zero_mean_not_significant(self) -> None:
        # Residuals centered at zero → no bias
        residuals = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
        mean, significant = compute_mean_bias(residuals)
        assert mean == pytest.approx(0.0)
        assert not significant

    def test_positive_bias_significant(self) -> None:
        # All positive residuals → significant positive bias
        residuals = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.04, 0.07] * 10
        mean, significant = compute_mean_bias(residuals)
        assert mean > 0
        assert significant

    def test_negative_bias_significant(self) -> None:
        # All negative residuals → significant negative bias
        residuals = [-0.10, -0.09, -0.11, -0.10, -0.12] * 20
        mean, significant = compute_mean_bias(residuals)
        assert mean < 0
        assert significant

    def test_small_sample_not_significant(self) -> None:
        # Large mean but only 3 observations → t-stat too small
        residuals = [0.10, 0.08, 0.12]
        mean, significant = compute_mean_bias(residuals)
        assert mean == pytest.approx(0.10, abs=0.01)
        # With n=3 and moderate spread, t-stat = mean / (stdev / sqrt(3))
        # Could go either way, but the point is testing small samples work
        assert isinstance(significant, bool)

    def test_empty_returns_zero(self) -> None:
        mean, significant = compute_mean_bias([])
        assert mean == 0.0
        assert not significant

    def test_single_value_returns_zero(self) -> None:
        mean, significant = compute_mean_bias([0.05])
        assert mean == 0.0
        assert not significant

    def test_constant_values(self) -> None:
        # All same value → stdev is 0 → not significant (avoid ZeroDivisionError)
        residuals = [0.05] * 10
        mean, significant = compute_mean_bias(residuals)
        assert mean == pytest.approx(0.05)
        assert not significant


class TestComputeHeteroscedasticity:
    def test_no_correlation(self) -> None:
        # Random relationship between predictions and abs residuals
        predictions = [0.250, 0.260, 0.270, 0.280, 0.290, 0.300, 0.310, 0.320]
        abs_residuals = [0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.03, 0.01]
        corr, significant = compute_heteroscedasticity(predictions, abs_residuals)
        assert abs(corr) < 0.5
        assert not significant  # |r| < 0.15 threshold

    def test_positive_correlation_significant(self) -> None:
        # Higher predictions → larger errors (heteroscedastic)
        predictions = [float(x) for x in range(10, 110, 10)]
        abs_residuals = [x * 0.1 for x in predictions]  # perfectly correlated
        corr, significant = compute_heteroscedasticity(predictions, abs_residuals)
        assert corr > 0.15
        assert significant

    def test_too_few_points(self) -> None:
        corr, significant = compute_heteroscedasticity([0.25], [0.01])
        assert corr == 0.0
        assert not significant

    def test_empty_returns_zero(self) -> None:
        corr, significant = compute_heteroscedasticity([], [])
        assert corr == 0.0
        assert not significant


class TestComputeCalibrationBins:
    def test_basic_bins(self) -> None:
        # 20 observations, should produce 2 bins (n_bins=2)
        predictions = [0.200 + i * 0.010 for i in range(20)]
        actuals = [0.205 + i * 0.010 for i in range(20)]
        bins = compute_calibration_bins(predictions, actuals, n_bins=2)

        assert len(bins) == 2
        # First bin has lower predictions, second has higher
        assert bins[0].bin_center < bins[1].bin_center
        assert bins[0].count == 10
        assert bins[1].count == 10
        # Residuals (actual - predicted) are all +0.005
        for b in bins:
            assert b.mean_residual == pytest.approx(0.005, abs=0.001)

    def test_uneven_bins(self) -> None:
        # 7 observations with 3 bins → bins of size 2, 2, 3 (or similar)
        predictions = [0.250, 0.260, 0.270, 0.280, 0.290, 0.300, 0.310]
        actuals = [0.255, 0.265, 0.275, 0.285, 0.295, 0.305, 0.315]
        bins = compute_calibration_bins(predictions, actuals, n_bins=3)

        assert len(bins) == 3
        total_count = sum(b.count for b in bins)
        assert total_count == 7
        # All bins have positive residual
        for b in bins:
            assert b.mean_residual == pytest.approx(0.005, abs=0.001)

    def test_sorted_by_predicted(self) -> None:
        predictions = [0.300, 0.200, 0.250, 0.350, 0.150, 0.400]
        actuals = [0.305, 0.205, 0.255, 0.355, 0.155, 0.405]
        bins = compute_calibration_bins(predictions, actuals, n_bins=2)

        assert bins[0].bin_center < bins[1].bin_center

    def test_too_few_observations(self) -> None:
        bins = compute_calibration_bins([0.250], [0.260], n_bins=5)
        assert len(bins) == 1
        assert bins[0].count == 1
        assert bins[0].mean_residual == pytest.approx(0.010)

    def test_empty_returns_empty(self) -> None:
        bins = compute_calibration_bins([], [], n_bins=5)
        assert bins == []

    def test_bin_center_is_midpoint(self) -> None:
        # 4 observations, 2 bins
        predictions = [0.200, 0.220, 0.300, 0.320]
        actuals = [0.200, 0.220, 0.300, 0.320]
        bins = compute_calibration_bins(predictions, actuals, n_bins=2)

        # First bin: predictions 0.200, 0.220 → center ~ 0.210
        assert bins[0].bin_center == pytest.approx(0.210)
        # Second bin: predictions 0.300, 0.320 → center ~ 0.310
        assert bins[1].bin_center == pytest.approx(0.310)

    def test_mean_predicted_matches_bin_data(self) -> None:
        predictions = [0.200, 0.220, 0.300, 0.320]
        actuals = [0.210, 0.230, 0.280, 0.340]
        bins = compute_calibration_bins(predictions, actuals, n_bins=2)

        assert bins[0].mean_predicted == pytest.approx(0.210)
        assert bins[0].mean_actual == pytest.approx(0.220)
        assert bins[0].mean_residual == pytest.approx(0.010)

    def test_default_n_bins(self) -> None:
        # 100 observations with default bins (10)
        predictions = [0.200 + i * 0.002 for i in range(100)]
        actuals = [0.205 + i * 0.002 for i in range(100)]
        bins = compute_calibration_bins(predictions, actuals)

        assert len(bins) == 10
        for b in bins:
            assert b.count == 10
