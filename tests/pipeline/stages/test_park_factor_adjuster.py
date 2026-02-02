import pytest

from fantasy_baseball_manager.pipeline.stages.park_factor_adjuster import (
    ParkFactorAdjuster,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


class FakeParkFactorProvider:
    def __init__(self, factors: dict[str, dict[str, float]]) -> None:
        self._factors = factors

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        return self._factors


def _make_player(
    team: str = "COL",
    rates: dict[str, float] | None = None,
) -> PlayerRates:
    return PlayerRates(
        player_id="p1",
        name="Test Player",
        year=2025,
        age=28,
        rates=rates or {"hr": 0.05, "bb": 0.08, "so": 0.20},
        metadata={"team": team},
    )


class TestParkFactorAdjuster:
    def test_coors_inflation_divided_out(self) -> None:
        """HR rate at Coors (factor 1.2) should be divided by 1.2."""
        provider = FakeParkFactorProvider(
            {
                "COL": {"hr": 1.2, "bb": 1.0, "so": 0.95},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        result = adjuster.adjust([_make_player(team="COL")])
        assert result[0].rates["hr"] == pytest.approx(0.05 / 1.2)
        assert result[0].rates["bb"] == pytest.approx(0.08)
        assert result[0].rates["so"] == pytest.approx(0.20 / 0.95)

    def test_neutral_park_unchanged(self) -> None:
        """A park with factor 1.0 for all stats leaves rates unchanged."""
        provider = FakeParkFactorProvider(
            {
                "NYY": {"hr": 1.0, "bb": 1.0, "so": 1.0},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        result = adjuster.adjust([_make_player(team="NYY")])
        assert result[0].rates["hr"] == pytest.approx(0.05)
        assert result[0].rates["bb"] == pytest.approx(0.08)
        assert result[0].rates["so"] == pytest.approx(0.20)

    def test_missing_team_passthrough(self) -> None:
        """Players with no team metadata pass through unchanged."""
        provider = FakeParkFactorProvider(
            {
                "COL": {"hr": 1.2},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        player = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            rates={"hr": 0.05},
            metadata={},
        )
        result = adjuster.adjust([player])
        assert result[0].rates["hr"] == pytest.approx(0.05)

    def test_unknown_team_passthrough(self) -> None:
        """Players with a team not in the provider pass through."""
        provider = FakeParkFactorProvider(
            {
                "COL": {"hr": 1.2},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        result = adjuster.adjust([_make_player(team="UNKNOWN")])
        assert result[0].rates["hr"] == pytest.approx(0.05)

    def test_stat_without_factor_uses_one(self) -> None:
        """Stats not in the park factor dict default to factor 1.0."""
        provider = FakeParkFactorProvider(
            {
                "COL": {"hr": 1.2},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        result = adjuster.adjust([_make_player(team="COL")])
        assert result[0].rates["bb"] == pytest.approx(0.08)

    def test_preserves_metadata(self) -> None:
        provider = FakeParkFactorProvider(
            {
                "COL": {"hr": 1.2},
            }
        )
        adjuster = ParkFactorAdjuster(provider)
        result = adjuster.adjust([_make_player(team="COL")])
        assert result[0].metadata["team"] == "COL"
