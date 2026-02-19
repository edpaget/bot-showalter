from typing import Any

from fantasy_baseball_manager.domain.sprint_speed import SprintSpeed
from fantasy_baseball_manager.ingest.column_maps import make_sprint_speed_mapper


class TestSprintSpeedMapper:
    def test_maps_valid_row(self) -> None:
        mapper = make_sprint_speed_mapper(season=2024)
        row: dict[str, Any] = {
            "player_id": 123456,
            "sprint_speed": 28.5,
            "hp_to_1b": 4.2,
            "bolts": 3,
            "competitive_runs": 50,
        }
        result = mapper(row)
        assert isinstance(result, SprintSpeed)
        assert result.mlbam_id == 123456
        assert result.season == 2024
        assert result.sprint_speed == 28.5
        assert result.hp_to_1b == 4.2
        assert result.bolts == 3
        assert result.competitive_runs == 50

    def test_missing_player_id_returns_none(self) -> None:
        mapper = make_sprint_speed_mapper(season=2024)
        row: dict[str, Any] = {
            "player_id": float("nan"),
            "sprint_speed": 28.5,
        }
        result = mapper(row)
        assert result is None

    def test_nan_sprint_speed(self) -> None:
        mapper = make_sprint_speed_mapper(season=2024)
        row: dict[str, Any] = {
            "player_id": 123456,
            "sprint_speed": float("nan"),
            "hp_to_1b": float("nan"),
            "bolts": float("nan"),
            "competitive_runs": float("nan"),
        }
        result = mapper(row)
        assert result is not None
        assert result.sprint_speed is None
        assert result.hp_to_1b is None
        assert result.bolts is None
        assert result.competitive_runs is None

    def test_season_from_closure(self) -> None:
        mapper = make_sprint_speed_mapper(season=2023)
        row: dict[str, Any] = {
            "player_id": 123456,
            "sprint_speed": 27.0,
        }
        result = mapper(row)
        assert result is not None
        assert result.season == 2023

    def test_string_values_from_csv(self) -> None:
        mapper = make_sprint_speed_mapper(season=2024)
        row: dict[str, Any] = {
            "player_id": "123456",
            "sprint_speed": "28.5",
            "hp_to_1b": "4.2",
            "bolts": "3",
            "competitive_runs": "50",
        }
        result = mapper(row)
        assert isinstance(result, SprintSpeed)
        assert result.mlbam_id == 123456
        assert result.sprint_speed == 28.5
        assert result.hp_to_1b == 4.2
        assert result.bolts == 3
        assert result.competitive_runs == 50

    def test_none_values_for_missing_fields(self) -> None:
        mapper = make_sprint_speed_mapper(season=2024)
        row: dict[str, Any] = {
            "player_id": "123456",
            "sprint_speed": None,
            "hp_to_1b": None,
            "bolts": None,
            "competitive_runs": None,
        }
        result = mapper(row)
        assert result is not None
        assert result.mlbam_id == 123456
        assert result.sprint_speed is None
        assert result.hp_to_1b is None
        assert result.bolts is None
        assert result.competitive_runs is None
