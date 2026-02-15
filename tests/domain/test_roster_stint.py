import pytest

from fantasy_baseball_manager.domain.roster_stint import RosterStint


class TestRosterStint:
    def test_construct_with_required_fields(self) -> None:
        stint = RosterStint(player_id=1, team_id=10, season=2024, start_date="2024-03-28")
        assert stint.player_id == 1
        assert stint.team_id == 10
        assert stint.season == 2024
        assert stint.start_date == "2024-03-28"

    def test_optional_fields_default_to_none(self) -> None:
        stint = RosterStint(player_id=1, team_id=10, season=2024, start_date="2024-03-28")
        assert stint.id is None
        assert stint.end_date is None
        assert stint.loaded_at is None

    def test_construct_with_all_fields(self) -> None:
        stint = RosterStint(
            player_id=1,
            team_id=10,
            season=2024,
            start_date="2024-03-28",
            id=42,
            end_date="2024-07-30",
            loaded_at="2024-08-01T12:00:00+00:00",
        )
        assert stint.id == 42
        assert stint.end_date == "2024-07-30"
        assert stint.loaded_at == "2024-08-01T12:00:00+00:00"

    def test_frozen(self) -> None:
        stint = RosterStint(player_id=1, team_id=10, season=2024, start_date="2024-03-28")
        with pytest.raises(AttributeError):
            stint.player_id = 2  # type: ignore[misc]
