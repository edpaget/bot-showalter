import importlib

import pytest

import fantasy_baseball_manager.features.group_library
from fantasy_baseball_manager.features.group_library import (
    make_batting_counting_lags,
    make_batting_rate_lags,
    make_pitching_counting_lags,
)
from fantasy_baseball_manager.features.groups import _clear, get_group, list_groups
from fantasy_baseball_manager.features.types import Feature, Source, TransformFeature


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()
    importlib.reload(fantasy_baseball_manager.features.group_library)


class TestStaticGroups:
    def test_all_static_groups_registered(self) -> None:
        names = list_groups()
        expected = [
            "age",
            "positions",
            "projected_batting_pt",
            "projected_pitching_pt",
            "statcast_batted_ball",
            "statcast_expected_stats",
            "statcast_pitch_mix",
            "statcast_plate_discipline",
            "statcast_spin_profile",
        ]
        for name in expected:
            assert name in names, f"'{name}' not found in registered groups"

    def test_player_types(self) -> None:
        assert get_group("age").player_type == "both"
        assert get_group("positions").player_type == "both"
        assert get_group("statcast_batted_ball").player_type == "batter"
        assert get_group("statcast_plate_discipline").player_type == "both"
        assert get_group("statcast_expected_stats").player_type == "batter"
        assert get_group("statcast_pitch_mix").player_type == "pitcher"
        assert get_group("statcast_spin_profile").player_type == "pitcher"
        assert get_group("projected_batting_pt").player_type == "batter"
        assert get_group("projected_pitching_pt").player_type == "pitcher"

    def test_all_groups_have_features(self) -> None:
        for name in list_groups():
            group = get_group(name)
            assert len(group.features) > 0, f"'{name}' has no features"

    def test_statcast_groups_contain_transform_features(self) -> None:
        statcast_names = [
            "statcast_batted_ball",
            "statcast_plate_discipline",
            "statcast_expected_stats",
            "statcast_pitch_mix",
            "statcast_spin_profile",
        ]
        for name in statcast_names:
            group = get_group(name)
            assert any(isinstance(f, TransformFeature) for f in group.features), f"'{name}' has no TransformFeature"

    def test_age_group_contains_feature(self) -> None:
        group = get_group("age")
        assert all(isinstance(f, Feature) for f in group.features)

    def test_projected_pt_groups_contain_feature(self) -> None:
        batting_pt = get_group("projected_batting_pt")
        assert len(batting_pt.features) == 1
        assert isinstance(batting_pt.features[0], Feature)
        assert batting_pt.features[0].name == "proj_pa"

        pitching_pt = get_group("projected_pitching_pt")
        assert len(pitching_pt.features) == 1
        assert isinstance(pitching_pt.features[0], Feature)
        assert pitching_pt.features[0].name == "proj_ip"


class TestMakeBattingCountingLags:
    def test_default_produces_expected_features(self) -> None:
        group = make_batting_counting_lags(("hr", "rbi"), (1, 2))
        # pa + each category, for each lag
        # lag 1: pa_1, hr_1, rbi_1
        # lag 2: pa_2, hr_2, rbi_2
        assert len(group.features) == 6
        names = [f.name for f in group.features]
        assert names == ["pa_1", "hr_1", "rbi_1", "pa_2", "hr_2", "rbi_2"]

    def test_custom_categories_and_lags(self) -> None:
        group = make_batting_counting_lags(("h",), (1, 2, 3))
        assert len(group.features) == 6  # (pa + h) * 3 lags
        names = [f.name for f in group.features]
        assert "pa_3" in names
        assert "h_3" in names

    def test_features_are_batting_source(self) -> None:
        group = make_batting_counting_lags(("hr",), (1,))
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.source == Source.BATTING


class TestMakePitchingCountingLags:
    def test_produces_expected_features(self) -> None:
        group = make_pitching_counting_lags(("so", "bb"), (1, 2))
        # ip, g, gs + each category, for each lag
        # lag 1: ip_1, g_1, gs_1, so_1, bb_1
        # lag 2: ip_2, g_2, gs_2, so_2, bb_2
        assert len(group.features) == 10
        names = [f.name for f in group.features]
        assert names == [
            "ip_1",
            "g_1",
            "gs_1",
            "so_1",
            "bb_1",
            "ip_2",
            "g_2",
            "gs_2",
            "so_2",
            "bb_2",
        ]

    def test_features_are_pitching_source(self) -> None:
        group = make_pitching_counting_lags(("era",), (1,))
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.source == Source.PITCHING


class TestMakeBattingRateLags:
    def test_produces_lag1_rate_features(self) -> None:
        group = make_batting_rate_lags(("avg", "obp", "slg"), (1,))
        assert len(group.features) == 3
        names = [f.name for f in group.features]
        assert names == ["avg_1", "obp_1", "slg_1"]

    def test_multiple_lags(self) -> None:
        group = make_batting_rate_lags(("avg", "obp"), (1, 2))
        assert len(group.features) == 4
        names = [f.name for f in group.features]
        assert names == ["avg_1", "obp_1", "avg_2", "obp_2"]
