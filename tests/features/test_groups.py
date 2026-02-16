import pytest

from fantasy_baseball_manager.features.groups import (
    FeatureGroup,
    _clear,
    compose_feature_set,
    get_group,
    list_groups,
    register_group,
)
from fantasy_baseball_manager.features.types import Feature, Source, SpineFilter


def _make_feature(name: str) -> Feature:
    return Feature(name=name, source=Source.BATTING, column=name)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestFeatureGroup:
    def test_frozen(self) -> None:
        group = FeatureGroup(
            name="test",
            description="A test group",
            player_type="batter",
            features=(_make_feature("hr"),),
        )
        with pytest.raises(AttributeError):
            group.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        f = _make_feature("hr")
        group = FeatureGroup(
            name="test",
            description="desc",
            player_type="pitcher",
            features=(f,),
        )
        assert group.name == "test"
        assert group.description == "desc"
        assert group.player_type == "pitcher"
        assert group.features == (f,)


class TestRegistry:
    def test_register_and_get(self) -> None:
        group = FeatureGroup(
            name="my_group",
            description="desc",
            player_type="batter",
            features=(_make_feature("hr"),),
        )
        result = register_group(group)
        assert result is group
        assert get_group("my_group") is group

    def test_duplicate_raises_value_error(self) -> None:
        group = FeatureGroup(
            name="dup",
            description="desc",
            player_type="batter",
            features=(_make_feature("hr"),),
        )
        register_group(group)
        with pytest.raises(ValueError, match="dup"):
            register_group(group)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="unknown"):
            get_group("unknown")

    def test_list_groups_sorted(self) -> None:
        for name in ("zebra", "alpha", "mid"):
            register_group(
                FeatureGroup(
                    name=name,
                    description="desc",
                    player_type="both",
                    features=(_make_feature("hr"),),
                )
            )
        assert list_groups() == ["alpha", "mid", "zebra"]

    def test_clear_empties_registry(self) -> None:
        register_group(
            FeatureGroup(
                name="temp",
                description="desc",
                player_type="batter",
                features=(_make_feature("hr"),),
            )
        )
        _clear()
        assert list_groups() == []


class TestComposeFeatureSet:
    def test_two_groups_union(self) -> None:
        g1 = FeatureGroup(
            name="g1",
            description="first",
            player_type="batter",
            features=(_make_feature("hr"), _make_feature("rbi")),
        )
        g2 = FeatureGroup(
            name="g2",
            description="second",
            player_type="batter",
            features=(_make_feature("sb"),),
        )
        fs = compose_feature_set(
            name="composed",
            groups=[g1, g2],
            seasons=(2023, 2024),
        )
        assert [f.name for f in fs.features] == ["hr", "rbi", "sb"]

    def test_deduplicates_first_wins(self) -> None:
        f_hr_1 = Feature(name="hr", source=Source.BATTING, column="hr", lag=1)
        f_hr_2 = Feature(name="hr", source=Source.BATTING, column="hr", lag=2)
        g1 = FeatureGroup(
            name="g1",
            description="first",
            player_type="batter",
            features=(f_hr_1,),
        )
        g2 = FeatureGroup(
            name="g2",
            description="second",
            player_type="batter",
            features=(f_hr_2, _make_feature("sb")),
        )
        fs = compose_feature_set(
            name="composed",
            groups=[g1, g2],
            seasons=(2023,),
        )
        assert len(fs.features) == 2
        assert fs.features[0] is f_hr_1  # first wins
        assert fs.features[1].name == "sb"

    def test_empty_groups_produce_empty_feature_set(self) -> None:
        fs = compose_feature_set(
            name="empty",
            groups=[],
            seasons=(2023,),
        )
        assert fs.features == ()

    def test_passthrough_fields(self) -> None:
        g = FeatureGroup(
            name="g",
            description="desc",
            player_type="batter",
            features=(_make_feature("hr"),),
        )
        spine = SpineFilter(min_pa=100, player_type="batter")
        fs = compose_feature_set(
            name="my_set",
            groups=[g],
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=spine,
        )
        assert fs.name == "my_set"
        assert fs.seasons == (2022, 2023)
        assert fs.source_filter == "fangraphs"
        assert fs.spine_filter == spine
