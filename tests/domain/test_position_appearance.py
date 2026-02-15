import pytest

from fantasy_baseball_manager.domain.position_appearance import PositionAppearance


class TestPositionAppearance:
    def test_construct_with_required_fields(self) -> None:
        pa = PositionAppearance(player_id=1, season=2024, position="CF", games=120)
        assert pa.player_id == 1
        assert pa.season == 2024
        assert pa.position == "CF"
        assert pa.games == 120

    def test_optional_fields_default_to_none(self) -> None:
        pa = PositionAppearance(player_id=1, season=2024, position="CF", games=120)
        assert pa.id is None
        assert pa.loaded_at is None

    def test_construct_with_all_fields(self) -> None:
        pa = PositionAppearance(
            player_id=1,
            season=2024,
            position="CF",
            games=120,
            id=42,
            loaded_at="2024-06-15T12:00:00+00:00",
        )
        assert pa.id == 42
        assert pa.loaded_at == "2024-06-15T12:00:00+00:00"

    def test_frozen(self) -> None:
        pa = PositionAppearance(player_id=1, season=2024, position="CF", games=120)
        with pytest.raises(AttributeError):
            pa.player_id = 2  # type: ignore[misc]
