from fantasy_baseball_manager.domain.sprint_speed import SprintSpeed


class TestSprintSpeed:
    def test_instantiation_required_fields(self) -> None:
        ss = SprintSpeed(mlbam_id=123456, season=2024)
        assert ss.mlbam_id == 123456
        assert ss.season == 2024

    def test_optional_fields_default_none(self) -> None:
        ss = SprintSpeed(mlbam_id=123456, season=2024)
        assert ss.sprint_speed is None
        assert ss.hp_to_1b is None
        assert ss.bolts is None
        assert ss.competitive_runs is None
        assert ss.id is None
        assert ss.loaded_at is None

    def test_all_fields(self) -> None:
        ss = SprintSpeed(
            mlbam_id=123456,
            season=2024,
            sprint_speed=28.5,
            hp_to_1b=4.2,
            bolts=3,
            competitive_runs=50,
            id=1,
            loaded_at="2024-01-01T00:00:00",
        )
        assert ss.sprint_speed == 28.5
        assert ss.hp_to_1b == 4.2
        assert ss.bolts == 3
        assert ss.competitive_runs == 50
        assert ss.id == 1
        assert ss.loaded_at == "2024-01-01T00:00:00"

    def test_frozen(self) -> None:
        ss = SprintSpeed(mlbam_id=123456, season=2024)
        try:
            ss.mlbam_id = 999  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass
