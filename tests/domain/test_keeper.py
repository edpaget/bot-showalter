import pytest

from fantasy_baseball_manager.domain.keeper import KeeperCost


class TestKeeperCost:
    def test_create_with_required_fields(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=25.0, source="auction")
        assert cost.player_id == 1
        assert cost.season == 2026
        assert cost.league == "dynasty"
        assert cost.cost == 25.0
        assert cost.source == "auction"

    def test_defaults(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=10.0, source="auction")
        assert cost.years_remaining == 1
        assert cost.id is None
        assert cost.loaded_at is None

    def test_custom_optional_fields(self) -> None:
        cost = KeeperCost(
            player_id=1,
            season=2026,
            league="dynasty",
            cost=15.0,
            source="contract",
            years_remaining=3,
            id=42,
            loaded_at="2026-02-28T12:00:00",
        )
        assert cost.years_remaining == 3
        assert cost.id == 42
        assert cost.loaded_at == "2026-02-28T12:00:00"

    def test_immutability(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=25.0, source="auction")
        with pytest.raises(AttributeError):
            cost.cost = 30.0  # type: ignore[misc]
