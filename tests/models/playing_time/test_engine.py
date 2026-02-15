import random

from fantasy_baseball_manager.models.playing_time.engine import (
    PlayingTimeCoefficients,
    ResidualBuckets,
    ResidualPercentiles,
    _bucket_key,
    compute_residual_buckets,
    fit_playing_time,
    predict_playing_time,
    predict_playing_time_distribution,
    select_alpha,
)


class TestFitPlayingTime:
    def test_fit_recovers_known_coefficients(self) -> None:
        """Synthetic data where y = 2*x1 + 3*x2 + 10."""
        rows = [
            {"x1": 1.0, "x2": 2.0, "target": 2 * 1.0 + 3 * 2.0 + 10},
            {"x1": 3.0, "x2": 1.0, "target": 2 * 3.0 + 3 * 1.0 + 10},
            {"x1": 5.0, "x2": 4.0, "target": 2 * 5.0 + 3 * 4.0 + 10},
            {"x1": 2.0, "x2": 6.0, "target": 2 * 2.0 + 3 * 6.0 + 10},
            {"x1": 4.0, "x2": 3.0, "target": 2 * 4.0 + 3 * 3.0 + 10},
        ]
        result = fit_playing_time(rows, ["x1", "x2"], "target", "batter")
        assert result.feature_names == ("x1", "x2")
        assert result.player_type == "batter"
        assert abs(result.coefficients[0] - 2.0) < 1e-6
        assert abs(result.coefficients[1] - 3.0) < 1e-6
        assert abs(result.intercept - 10.0) < 1e-6
        assert abs(result.r_squared - 1.0) < 1e-6

    def test_fit_skips_rows_with_none_target(self) -> None:
        rows = [
            {"x1": 1.0, "target": 5.0},
            {"x1": 2.0, "target": None},
            {"x1": 3.0, "target": 10.0},
            {"x1": 5.0, "target": 20.0},
        ]
        result = fit_playing_time(rows, ["x1"], "target", "batter")
        # Only 3 rows used; should still produce valid coefficients
        assert result.player_type == "batter"
        assert len(result.coefficients) == 1

    def test_fit_treats_none_features_as_zero(self) -> None:
        """None feature values should be treated as 0.0."""
        rows = [
            {"x1": 1.0, "x2": None, "target": 2 * 1.0 + 3 * 0.0 + 10},
            {"x1": 3.0, "x2": 1.0, "target": 2 * 3.0 + 3 * 1.0 + 10},
            {"x1": 5.0, "x2": 4.0, "target": 2 * 5.0 + 3 * 4.0 + 10},
            {"x1": 2.0, "x2": 6.0, "target": 2 * 2.0 + 3 * 6.0 + 10},
            {"x1": 4.0, "x2": 3.0, "target": 2 * 4.0 + 3 * 3.0 + 10},
        ]
        result = fit_playing_time(rows, ["x1", "x2"], "target", "batter")
        assert abs(result.coefficients[0] - 2.0) < 1e-6
        assert abs(result.coefficients[1] - 3.0) < 1e-6
        assert abs(result.intercept - 10.0) < 1e-6


class TestFitPlayingTimeRidge:
    """Tests for ridge regression support in fit_playing_time."""

    def _ols_rows(self) -> list[dict[str, float]]:
        """Synthetic data: y = 2*x1 + 3*x2 + 10."""
        return [
            {"x1": 1.0, "x2": 2.0, "target": 2 * 1.0 + 3 * 2.0 + 10},
            {"x1": 3.0, "x2": 1.0, "target": 2 * 3.0 + 3 * 1.0 + 10},
            {"x1": 5.0, "x2": 4.0, "target": 2 * 5.0 + 3 * 4.0 + 10},
            {"x1": 2.0, "x2": 6.0, "target": 2 * 2.0 + 3 * 6.0 + 10},
            {"x1": 4.0, "x2": 3.0, "target": 2 * 4.0 + 3 * 3.0 + 10},
        ]

    def test_alpha_zero_matches_ols(self) -> None:
        rows = self._ols_rows()
        ols = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=0.0)
        assert abs(ols.coefficients[0] - 2.0) < 1e-6
        assert abs(ols.coefficients[1] - 3.0) < 1e-6
        assert abs(ols.intercept - 10.0) < 1e-6

    def test_large_alpha_shrinks_coefficients(self) -> None:
        rows = self._ols_rows()
        ols = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=0.0)
        ridge = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=1000.0)
        for ols_c, ridge_c in zip(ols.coefficients, ridge.coefficients):
            assert abs(ridge_c) < abs(ols_c)

    def test_ridge_stores_alpha_in_result(self) -> None:
        rows = self._ols_rows()
        result = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=42.0)
        assert result.alpha == 42.0

    def test_ridge_still_recovers_intercept(self) -> None:
        # With perfect data and moderate alpha, intercept should still be reasonable
        # (not regularized). For perfect data, intercept absorbs some bias but should
        # not be pushed toward zero the way coefficients are.
        rows = self._ols_rows()
        ridge = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=1.0)
        # Intercept should be non-trivial (OLS intercept is 10.0)
        assert ridge.intercept > 1.0

    def test_ridge_r_squared_on_training_data(self) -> None:
        rows = self._ols_rows()
        result = fit_playing_time(rows, ["x1", "x2"], "target", "batter", alpha=1.0)
        assert 0.0 <= result.r_squared <= 1.0


class TestSelectAlpha:
    """Tests for select_alpha cross-validation."""

    def _multi_season_rows(self) -> list[dict[str, float]]:
        """Rows across 3 seasons with y = 2*x1 + 10 + noise."""
        rng = random.Random(99)
        rows: list[dict[str, float]] = []
        for season in [2021, 2022, 2023]:
            for i in range(20):
                x1 = float(i * 5 + rng.gauss(0, 1))
                rows.append({"x1": x1, "season": float(season), "target": 2.0 * x1 + 10.0 + rng.gauss(0, 5)})
        return rows

    def test_returns_alpha_from_candidate_list(self) -> None:
        rows = self._multi_season_rows()
        alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
        result = select_alpha(rows, ["x1"], "target", "batter", alphas=alphas)
        assert result in alphas

    def test_splits_by_season_not_random(self) -> None:
        # With 3 distinct seasons, each fold should hold out exactly one season
        rows = [
            {"x1": float(i), "season": float(s), "target": 2.0 * i + 10.0}
            for s in [2021, 2022, 2023]
            for i in range(10)
        ]
        # If splitting is by season, result should still be valid
        result = select_alpha(rows, ["x1"], "target", "batter")
        assert result in (0.01, 0.1, 1.0, 10.0, 100.0)

    def test_single_season_returns_first_alpha(self) -> None:
        rows = [{"x1": float(i), "season": 2023.0, "target": 2.0 * i} for i in range(20)]
        alphas = (5.0, 10.0, 50.0)
        result = select_alpha(rows, ["x1"], "target", "batter", alphas=alphas)
        assert result == 5.0

    def test_custom_alphas(self) -> None:
        rows = self._multi_season_rows()
        alphas = (0.5, 5.0, 50.0)
        result = select_alpha(rows, ["x1"], "target", "batter", alphas=alphas)
        assert result in alphas

    def test_n_folds_capped_by_season_count(self) -> None:
        # 3 seasons, n_folds=5 → should use 3 folds (one per season)
        rows = self._multi_season_rows()
        result = select_alpha(rows, ["x1"], "target", "batter", n_folds=5)
        assert result in (0.01, 0.1, 1.0, 10.0, 100.0)

    def test_perfect_data_prefers_low_alpha(self) -> None:
        # Noise-free data: least regularization should win
        rows = [
            {"x1": float(i), "season": float(s), "target": 2.0 * i + 10.0}
            for s in [2021, 2022, 2023]
            for i in range(20)
        ]
        alphas = (0.01, 1.0, 100.0)
        result = select_alpha(rows, ["x1"], "target", "batter", alphas=alphas)
        assert result == 0.01


class TestPredictPlayingTime:
    def test_predict_applies_coefficients(self) -> None:
        coefficients = PlayingTimeCoefficients(
            feature_names=("x1", "x2"),
            coefficients=(2.0, 3.0),
            intercept=10.0,
            r_squared=1.0,
            player_type="batter",
        )
        result = predict_playing_time({"x1": 5.0, "x2": 4.0}, coefficients)
        assert abs(result - (10.0 + 2.0 * 5.0 + 3.0 * 4.0)) < 1e-6

    def test_predict_clamps_to_min(self) -> None:
        coefficients = PlayingTimeCoefficients(
            feature_names=("x1",),
            coefficients=(-100.0,),
            intercept=0.0,
            r_squared=1.0,
            player_type="batter",
        )
        result = predict_playing_time({"x1": 10.0}, coefficients)
        assert result == 0.0

    def test_predict_clamps_to_max(self) -> None:
        coefficients = PlayingTimeCoefficients(
            feature_names=("x1",),
            coefficients=(100.0,),
            intercept=0.0,
            r_squared=1.0,
            player_type="batter",
        )
        result = predict_playing_time({"x1": 10.0}, coefficients, clamp_max=750.0)
        assert result == 750.0

    def test_predict_treats_none_as_zero(self) -> None:
        coefficients = PlayingTimeCoefficients(
            feature_names=("x1", "x2"),
            coefficients=(2.0, 3.0),
            intercept=10.0,
            r_squared=1.0,
            player_type="batter",
        )
        result = predict_playing_time({"x1": 5.0, "x2": None}, coefficients)
        assert abs(result - (10.0 + 2.0 * 5.0 + 3.0 * 0.0)) < 1e-6


class TestBucketKey:
    def test_bucket_key_young_healthy(self) -> None:
        assert _bucket_key(25, 0) == "young_healthy"

    def test_bucket_key_young_injured(self) -> None:
        assert _bucket_key(25, 15) == "young_injured"

    def test_bucket_key_old_healthy(self) -> None:
        assert _bucket_key(32, 0) == "old_healthy"

    def test_bucket_key_old_injured(self) -> None:
        assert _bucket_key(32, 30) == "old_injured"

    def test_bucket_key_none_age_is_young(self) -> None:
        assert _bucket_key(None, 0) == "young_healthy"

    def test_bucket_key_none_il_is_healthy(self) -> None:
        assert _bucket_key(25, None) == "young_healthy"

    def test_bucket_key_boundary_age_30_is_old(self) -> None:
        assert _bucket_key(30, 0) == "old_healthy"


def _make_residual_rows(
    n: int,
    *,
    age: float = 25.0,
    il_days_1: float = 0.0,
    noise_std: float = 0.0,
) -> tuple[list[dict[str, float | None]], PlayingTimeCoefficients]:
    """Build synthetic rows where target = 2*x1 + 100 + noise."""
    rng = random.Random(42)
    coeff = PlayingTimeCoefficients(
        feature_names=("x1",),
        coefficients=(2.0,),
        intercept=100.0,
        r_squared=1.0,
        player_type="batter",
    )
    rows: list[dict[str, float | None]] = []
    for i in range(n):
        x1 = float(i * 10)
        noise = rng.gauss(0, noise_std) if noise_std > 0 else 0.0
        rows.append(
            {
                "x1": x1,
                "age": age,
                "il_days_1": il_days_1,
                "target": 100.0 + 2.0 * x1 + noise,
            }
        )
    return rows, coeff


class TestComputeResidualBuckets:
    def test_all_bucket_always_present(self) -> None:
        rows, coeff = _make_residual_rows(30)
        result = compute_residual_buckets(rows, coeff, "target")
        assert "all" in result.buckets

    def test_residual_percentiles_ordered(self) -> None:
        rows, coeff = _make_residual_rows(30, noise_std=20.0)
        result = compute_residual_buckets(rows, coeff, "target")
        for percs in result.buckets.values():
            assert percs.p10 <= percs.p25 <= percs.p50 <= percs.p75 <= percs.p90

    def test_perfect_fit_has_zero_residuals(self) -> None:
        rows, coeff = _make_residual_rows(30, noise_std=0.0)
        result = compute_residual_buckets(rows, coeff, "target")
        percs = result.buckets["all"]
        assert abs(percs.p10) < 1e-6
        assert abs(percs.p50) < 1e-6
        assert abs(percs.p90) < 1e-6

    def test_noisy_data_captures_spread(self) -> None:
        rows, coeff = _make_residual_rows(100, noise_std=50.0)
        result = compute_residual_buckets(rows, coeff, "target")
        percs = result.buckets["all"]
        assert percs.p90 - percs.p10 > 0

    def test_buckets_reflect_feature_based_variance(self) -> None:
        low_rows, coeff = _make_residual_rows(30, age=25.0, il_days_1=0.0, noise_std=5.0)
        high_rows, _ = _make_residual_rows(30, age=35.0, il_days_1=20.0, noise_std=50.0)
        all_rows = low_rows + high_rows
        result = compute_residual_buckets(all_rows, coeff, "target")
        yh = result.buckets["young_healthy"]
        oi = result.buckets["old_injured"]
        assert (oi.p90 - oi.p10) > (yh.p90 - yh.p10)

    def test_small_bucket_excluded(self) -> None:
        rows, coeff = _make_residual_rows(30, age=25.0, il_days_1=0.0)
        # Add just 2 old_injured rows — below min_bucket_size
        for i in range(2):
            rows.append({"x1": float(i), "age": 35.0, "il_days_1": 20.0, "target": 100.0 + 2.0 * i})
        result = compute_residual_buckets(rows, coeff, "target", min_bucket_size=20)
        assert "old_injured" not in result.buckets
        assert "all" in result.buckets


def _zero_residual_buckets(player_type: str = "batter") -> ResidualBuckets:
    """Build a ResidualBuckets with all offsets at zero."""
    percs = ResidualPercentiles(p10=0.0, p25=0.0, p50=0.0, p75=0.0, p90=0.0, count=100, std=0.0, mean_offset=0.0)
    return ResidualBuckets(buckets={"all": percs, "young_healthy": percs}, player_type=player_type)


def _spread_residual_buckets(player_type: str = "batter") -> ResidualBuckets:
    """Build a ResidualBuckets with known nonzero offsets."""
    percs = ResidualPercentiles(
        p10=-50.0,
        p25=-20.0,
        p50=5.0,
        p75=30.0,
        p90=60.0,
        count=100,
        std=35.0,
        mean_offset=3.0,
    )
    return ResidualBuckets(buckets={"all": percs, "young_healthy": percs}, player_type=player_type)


class TestPredictPlayingTimeDistribution:
    def test_distribution_percentiles_ordered(self) -> None:
        buckets = _spread_residual_buckets()
        dist = predict_playing_time_distribution(400.0, {"age": 25.0, "il_days_1": 0.0}, buckets)
        assert dist.p10 <= dist.p25 <= dist.p50 <= dist.p75 <= dist.p90

    def test_zero_residuals_give_point_at_all_percentiles(self) -> None:
        buckets = _zero_residual_buckets()
        dist = predict_playing_time_distribution(400.0, {"age": 25.0, "il_days_1": 0.0}, buckets)
        assert dist.p10 == dist.p25 == dist.p50 == dist.p75 == dist.p90 == 400.0

    def test_negative_offset_clamps_to_zero(self) -> None:
        # point estimate = 20, p10 offset = -50 → should clamp to 0
        buckets = _spread_residual_buckets()
        dist = predict_playing_time_distribution(20.0, {"age": 25.0, "il_days_1": 0.0}, buckets)
        assert dist.p10 == 0.0

    def test_positive_offset_clamps_to_max(self) -> None:
        buckets = _spread_residual_buckets()
        dist = predict_playing_time_distribution(720.0, {"age": 25.0, "il_days_1": 0.0}, buckets, clamp_max=750.0)
        assert dist.p90 == 750.0

    def test_fallback_to_all_when_bucket_missing(self) -> None:
        # Only "all" bucket exists; old_injured should fall back to "all"
        percs = ResidualPercentiles(
            p10=-10.0,
            p25=-5.0,
            p50=0.0,
            p75=5.0,
            p90=10.0,
            count=50,
            std=8.0,
            mean_offset=0.0,
        )
        buckets = ResidualBuckets(buckets={"all": percs}, player_type="batter")
        dist = predict_playing_time_distribution(400.0, {"age": 35.0, "il_days_1": 20.0}, buckets)
        assert dist.p10 == 390.0
        assert dist.p90 == 410.0

    def test_stat_name_matches_player_type(self) -> None:
        batter_buckets = _zero_residual_buckets("batter")
        pitcher_buckets = _zero_residual_buckets("pitcher")
        bat_dist = predict_playing_time_distribution(400.0, {"age": 25.0, "il_days_1": 0.0}, batter_buckets)
        pitch_dist = predict_playing_time_distribution(180.0, {"age": 25.0, "il_days_1": 0.0}, pitcher_buckets)
        assert bat_dist.stat == "pa"
        assert pitch_dist.stat == "ip"
