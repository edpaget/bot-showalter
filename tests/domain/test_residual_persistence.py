import pytest

from fantasy_baseball_manager.domain.residual_persistence import (
    ChronicPerformer,
    compute_residual_correlation_by_bucket,
    compute_rmse_ceiling,
    identify_chronic_performers,
)


class TestComputeResidualCorrelationByBucket:
    def test_three_buckets_with_known_correlations(self) -> None:
        # Bucket <200: players 0,1 — perfectly correlated residuals
        # Bucket 200-400: players 2,3 — perfectly anti-correlated
        # Bucket 400+: players 4,5,6 — moderate positive
        residuals_n = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.03]
        residuals_n1 = [0.02, 0.04, -0.01, -0.02, 0.005, 0.015, 0.02]
        sample_sizes = [100.0, 150.0, 250.0, 350.0, 450.0, 500.0, 550.0]
        bucket_edges = (200.0, 400.0)
        bucket_labels = ("<200", "200-400", "400+")

        corr_by_bucket, n_by_bucket = compute_residual_correlation_by_bucket(
            residuals_n, residuals_n1, sample_sizes, bucket_edges, bucket_labels
        )

        assert n_by_bucket == {"<200": 2, "200-400": 2, "400+": 3}
        assert corr_by_bucket["<200"] == pytest.approx(1.0)
        assert corr_by_bucket["200-400"] == pytest.approx(-1.0)
        assert corr_by_bucket["400+"] > 0.5

    def test_empty_bucket_returns_zero(self) -> None:
        # All players in 400+ bucket; <200 and 200-400 are empty
        residuals_n = [0.01, 0.02, 0.03]
        residuals_n1 = [0.02, 0.04, 0.06]
        sample_sizes = [500.0, 600.0, 700.0]
        bucket_edges = (200.0, 400.0)
        bucket_labels = ("<200", "200-400", "400+")

        corr_by_bucket, n_by_bucket = compute_residual_correlation_by_bucket(
            residuals_n, residuals_n1, sample_sizes, bucket_edges, bucket_labels
        )

        assert "<200" not in corr_by_bucket
        assert "200-400" not in corr_by_bucket
        assert corr_by_bucket["400+"] == pytest.approx(1.0)
        assert n_by_bucket["400+"] == 3

    def test_single_player_bucket_returns_zero(self) -> None:
        # Only 1 player in the <200 bucket — can't compute correlation
        residuals_n = [0.01, 0.02, 0.03]
        residuals_n1 = [0.02, 0.04, 0.06]
        sample_sizes = [100.0, 500.0, 600.0]
        bucket_edges = (200.0, 400.0)
        bucket_labels = ("<200", "200-400", "400+")

        corr_by_bucket, n_by_bucket = compute_residual_correlation_by_bucket(
            residuals_n, residuals_n1, sample_sizes, bucket_edges, bucket_labels
        )

        assert corr_by_bucket["<200"] == 0.0
        assert n_by_bucket["<200"] == 1


class TestIdentifyChronicPerformers:
    def test_players_above_one_sigma_both_seasons(self) -> None:
        # 6 players: players 0,1 are overperformers (>1σ both seasons)
        # Player 5 is underperformer (<-1σ both seasons)
        # Others are in the middle
        residuals_n = [0.05, 0.04, 0.001, -0.001, 0.002, -0.05]
        residuals_n1 = [0.06, 0.045, -0.002, 0.003, -0.001, -0.04]
        player_ids = [10, 20, 30, 40, 50, 60]
        player_names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
        pa_n = [400.0, 500.0, 300.0, 350.0, 450.0, 400.0]
        pa_n1 = [420.0, 510.0, 310.0, 360.0, 460.0, 410.0]

        overperformers, underperformers = identify_chronic_performers(
            residuals_n, residuals_n1, player_ids, player_names, pa_n, pa_n1
        )

        assert len(overperformers) == 2
        assert len(underperformers) == 1

        # Sorted by abs(mean_residual) descending
        assert overperformers[0].player_name == "Alpha"
        assert overperformers[1].player_name == "Bravo"
        assert underperformers[0].player_name == "Foxtrot"

    def test_no_chronic_performers_returns_empty(self) -> None:
        # All residuals near zero — no one exceeds 1σ in both seasons
        residuals_n = [0.001, -0.001, 0.002, -0.002]
        residuals_n1 = [-0.001, 0.001, -0.002, 0.002]
        player_ids = [1, 2, 3, 4]
        player_names = ["A", "B", "C", "D"]
        pa_n = [400.0, 400.0, 400.0, 400.0]
        pa_n1 = [400.0, 400.0, 400.0, 400.0]

        overperformers, underperformers = identify_chronic_performers(
            residuals_n, residuals_n1, player_ids, player_names, pa_n, pa_n1
        )

        assert overperformers == []
        assert underperformers == []

    def test_sorted_by_magnitude(self) -> None:
        # Two overperformers with different magnitudes
        residuals_n = [0.03, 0.06, -0.001, 0.001]
        residuals_n1 = [0.04, 0.05, 0.001, -0.001]
        player_ids = [1, 2, 3, 4]
        player_names = ["Small", "Big", "Normal1", "Normal2"]
        pa_n = [400.0, 400.0, 400.0, 400.0]
        pa_n1 = [400.0, 400.0, 400.0, 400.0]

        overperformers, _ = identify_chronic_performers(
            residuals_n, residuals_n1, player_ids, player_names, pa_n, pa_n1
        )

        assert len(overperformers) == 2
        # Big has larger mean residual, should be first
        assert overperformers[0].player_name == "Big"
        assert overperformers[1].player_name == "Small"

    def test_chronic_performer_fields(self) -> None:
        residuals_n = [0.05, -0.001, 0.001]
        residuals_n1 = [0.06, 0.001, -0.001]
        player_ids = [10, 20, 30]
        player_names = ["Star", "Normal1", "Normal2"]
        pa_n = [400.0, 300.0, 350.0]
        pa_n1 = [420.0, 310.0, 360.0]

        overperformers, _ = identify_chronic_performers(
            residuals_n, residuals_n1, player_ids, player_names, pa_n, pa_n1
        )

        assert len(overperformers) == 1
        p = overperformers[0]
        assert isinstance(p, ChronicPerformer)
        assert p.player_id == 10
        assert p.player_name == "Star"
        assert p.residual_n == pytest.approx(0.05)
        assert p.residual_n1 == pytest.approx(0.06)
        assert p.mean_residual == pytest.approx(0.055)
        assert p.pa_n == pytest.approx(400.0)
        assert p.pa_n1 == pytest.approx(420.0)


class TestComputeRmseCeiling:
    def test_correction_lowers_rmse(self) -> None:
        # Actuals systematically above estimates — prior residuals capture this
        actuals_n1 = [0.310, 0.320, 0.330]
        model_estimates_n1 = [0.290, 0.300, 0.310]
        mean_prior_residuals = [0.020, 0.020, 0.020]

        rmse_baseline, rmse_corrected, improvement_pct = compute_rmse_ceiling(
            actuals_n1, model_estimates_n1, mean_prior_residuals
        )

        assert rmse_baseline > 0
        assert rmse_corrected < rmse_baseline
        assert improvement_pct > 0

    def test_zero_residuals_returns_zero_improvement(self) -> None:
        actuals_n1 = [0.300, 0.310, 0.320]
        model_estimates_n1 = [0.290, 0.300, 0.310]
        mean_prior_residuals = [0.0, 0.0, 0.0]

        rmse_baseline, rmse_corrected, improvement_pct = compute_rmse_ceiling(
            actuals_n1, model_estimates_n1, mean_prior_residuals
        )

        assert rmse_baseline == rmse_corrected
        assert improvement_pct == pytest.approx(0.0)

    def test_percentage_math(self) -> None:
        # Exact case: baseline RMSE=0.020, corrected should be 0.0
        actuals_n1 = [0.310, 0.320]
        model_estimates_n1 = [0.290, 0.300]
        mean_prior_residuals = [0.020, 0.020]  # perfect correction

        rmse_baseline, rmse_corrected, improvement_pct = compute_rmse_ceiling(
            actuals_n1, model_estimates_n1, mean_prior_residuals
        )

        assert rmse_baseline == pytest.approx(0.020)
        assert rmse_corrected == pytest.approx(0.0, abs=1e-10)
        assert improvement_pct == pytest.approx(100.0)

    def test_empty_lists_returns_zeros(self) -> None:
        rmse_baseline, rmse_corrected, improvement_pct = compute_rmse_ceiling([], [], [])

        assert rmse_baseline == 0.0
        assert rmse_corrected == 0.0
        assert improvement_pct == 0.0
