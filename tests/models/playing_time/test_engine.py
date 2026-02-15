from fantasy_baseball_manager.models.playing_time.engine import (
    PlayingTimeCoefficients,
    fit_playing_time,
    predict_playing_time,
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
