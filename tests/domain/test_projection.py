import pytest

from fantasy_baseball_manager.domain.projection import Projection


class TestProjection:
    def test_construct_with_required_fields(self) -> None:
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={"hr": 30, "avg": 0.280},
        )
        assert proj.player_id == 1
        assert proj.season == 2025
        assert proj.system == "steamer"
        assert proj.version == "2025.1"
        assert proj.player_type == "batter"
        assert proj.stat_json == {"hr": 30, "avg": 0.280}

    def test_optional_fields_default_to_none(self) -> None:
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={},
        )
        assert proj.id is None
        assert proj.loaded_at is None

    def test_frozen(self) -> None:
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={},
        )
        with pytest.raises(AttributeError):
            proj.system = "zips"  # type: ignore[misc]
