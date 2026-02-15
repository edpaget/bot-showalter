import pytest

from fantasy_baseball_manager.domain.il_stint import ILStint


class TestILStint:
    def test_construct_with_required_fields(self) -> None:
        stint = ILStint(player_id=1, season=2024, start_date="2024-05-15", il_type="15")
        assert stint.player_id == 1
        assert stint.season == 2024
        assert stint.start_date == "2024-05-15"
        assert stint.il_type == "15"

    def test_optional_fields_default_to_none(self) -> None:
        stint = ILStint(player_id=1, season=2024, start_date="2024-05-15", il_type="15")
        assert stint.id is None
        assert stint.end_date is None
        assert stint.days is None
        assert stint.injury_location is None
        assert stint.transaction_type is None
        assert stint.loaded_at is None

    def test_construct_with_all_fields(self) -> None:
        stint = ILStint(
            player_id=1,
            season=2024,
            start_date="2024-05-15",
            il_type="15",
            id=42,
            end_date="2024-06-01",
            days=17,
            injury_location="Right elbow inflammation",
            transaction_type="placement",
            loaded_at="2024-06-15T12:00:00+00:00",
        )
        assert stint.id == 42
        assert stint.end_date == "2024-06-01"
        assert stint.days == 17
        assert stint.injury_location == "Right elbow inflammation"
        assert stint.transaction_type == "placement"

    def test_frozen(self) -> None:
        stint = ILStint(player_id=1, season=2024, start_date="2024-05-15", il_type="15")
        with pytest.raises(AttributeError):
            stint.player_id = 2  # type: ignore[misc]
