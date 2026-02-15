import pytest

from fantasy_baseball_manager.domain.level_factor import LevelFactor


class TestLevelFactor:
    def test_construct_with_required_fields(self) -> None:
        factor = LevelFactor(
            level="AAA",
            season=2024,
            factor=0.80,
            k_factor=1.15,
            bb_factor=0.92,
            iso_factor=0.85,
            babip_factor=0.95,
        )
        assert factor.level == "AAA"
        assert factor.season == 2024
        assert factor.factor == 0.80
        assert factor.k_factor == 1.15
        assert factor.bb_factor == 0.92
        assert factor.iso_factor == 0.85
        assert factor.babip_factor == 0.95

    def test_optional_fields_default_to_none(self) -> None:
        factor = LevelFactor(
            level="AAA",
            season=2024,
            factor=0.80,
            k_factor=1.15,
            bb_factor=0.92,
            iso_factor=0.85,
            babip_factor=0.95,
        )
        assert factor.id is None
        assert factor.loaded_at is None

    def test_frozen(self) -> None:
        factor = LevelFactor(
            level="AAA",
            season=2024,
            factor=0.80,
            k_factor=1.15,
            bb_factor=0.92,
            iso_factor=0.85,
            babip_factor=0.95,
        )
        with pytest.raises(AttributeError):
            factor.level = "AA"  # type: ignore[misc]
