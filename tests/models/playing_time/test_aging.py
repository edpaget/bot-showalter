import pytest

from fantasy_baseball_manager.models.playing_time.aging import (
    AgingCurve,
    compute_age_pt_factor,
    enrich_rows_with_age_pt_factor,
    fit_playing_time_aging_curve,
)


class TestComputeAgePtFactor:
    def test_factor_at_peak_is_one(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        assert compute_age_pt_factor(27, curve) == 1.0

    def test_factor_below_peak_greater_than_one(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        result = compute_age_pt_factor(24, curve)
        assert result == pytest.approx(1.03)

    def test_factor_above_peak_less_than_one(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        result = compute_age_pt_factor(32, curve)
        assert result == pytest.approx(0.975)

    def test_factor_none_age_returns_one(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        assert compute_age_pt_factor(None, curve) == 1.0

    def test_factor_monotonically_decreasing_after_peak(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        factors = [compute_age_pt_factor(age, curve) for age in range(27, 40)]
        for i in range(len(factors) - 1):
            assert factors[i] > factors[i + 1]


def _make_fit_rows(
    ages: range,
    deltas: dict[int, float],
    prior: float = 500.0,
    n_per_age: int = 40,
) -> list[dict[str, object]]:
    """Build synthetic rows for fit_playing_time_aging_curve tests."""
    rows: list[dict[str, object]] = []
    for age in ages:
        delta = deltas.get(age, 0.0)
        current = prior * (1.0 + delta)
        for _ in range(n_per_age):
            rows.append({"age": age, "prior": prior, "current": current})
    return rows


class TestFitPlayingTimeAgingCurve:
    def test_fit_recovers_known_curve(self) -> None:
        deltas = {age: 0.03 for age in range(22, 27)}
        deltas[27] = 0.0
        deltas.update({age: -0.02 for age in range(28, 37)})
        rows = _make_fit_rows(range(22, 37), deltas)
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=1,
        )
        assert curve.peak_age == pytest.approx(27.0)
        assert curve.improvement_rate == pytest.approx(0.03, abs=0.005)
        assert curve.decline_rate == pytest.approx(0.02, abs=0.005)

    def test_fit_skips_zero_prior(self) -> None:
        rows = [{"age": 25, "prior": 0.0, "current": 500.0}] * 50
        rows += [{"age": 26, "prior": 500.0, "current": 500.0}] * 50
        rows += [{"age": 27, "prior": 500.0, "current": 500.0}] * 50
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=1,
        )
        # Zero-prior rows excluded; should still fit from remaining data
        assert curve.player_type == "batter"

    def test_fit_skips_none_age(self) -> None:
        rows = [{"age": None, "prior": 500.0, "current": 520.0}] * 50
        rows += [{"age": 26, "prior": 500.0, "current": 500.0}] * 50
        rows += [{"age": 27, "prior": 500.0, "current": 500.0}] * 50
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=1,
        )
        assert curve.player_type == "batter"

    def test_fit_skips_below_min_pt(self) -> None:
        rows = [{"age": 25, "prior": 10.0, "current": 500.0}] * 50
        rows += [{"age": 26, "prior": 500.0, "current": 500.0}] * 50
        rows += [{"age": 27, "prior": 500.0, "current": 500.0}] * 50
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=1,
        )
        assert curve.player_type == "batter"

    def test_fit_falls_back_with_insufficient_data(self) -> None:
        rows = [{"age": 25, "prior": 500.0, "current": 520.0}]
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=30,
        )
        # Default batter curve
        assert curve.peak_age == 27.0
        assert curve.improvement_rate == 0.01
        assert curve.decline_rate == 0.005

    def test_fit_clamps_negative_rates(self) -> None:
        # Adversarial: older players improving (negative decline)
        deltas = {age: 0.01 for age in range(22, 37)}
        rows = _make_fit_rows(range(22, 37), deltas)
        curve = fit_playing_time_aging_curve(
            rows,
            "batter",
            "current",
            "prior",
            min_pt=50.0,
            min_samples=1,
        )
        assert curve.decline_rate >= 0.0
        assert curve.improvement_rate >= 0.0


class TestEnrichRowsWithAgePtFactor:
    def test_enrich_adds_age_pt_factor(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        rows = [{"age": 25, "pa_1": 500.0}]
        result = enrich_rows_with_age_pt_factor(rows, curve)
        assert "age_pt_factor" in result[0]
        assert result[0]["age_pt_factor"] == pytest.approx(1.02)

    def test_enrich_does_not_mutate_input(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        rows = [{"age": 25, "pa_1": 500.0}]
        enrich_rows_with_age_pt_factor(rows, curve)
        assert "age_pt_factor" not in rows[0]

    def test_enrich_handles_none_age(self) -> None:
        curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        rows = [{"age": None, "pa_1": 500.0}]
        result = enrich_rows_with_age_pt_factor(rows, curve)
        assert result[0]["age_pt_factor"] == 1.0
