from fantasy_baseball_manager.features.types import FeatureSet, SpineFilter, TransformFeature
from fantasy_baseball_manager.models.statcast_gbm.features import (
    build_batter_feature_set,
    build_pitcher_feature_set,
)


class TestBatterFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_batter_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_includes_age(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_batter_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names

    def test_includes_batting_lags(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "pa_2" in names
        assert "hr_1" in names
        assert "hr_2" in names


class TestPitcherFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_includes_age(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_pitcher_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names

    def test_includes_pitching_lags(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "ip_1" in names
        assert "ip_2" in names
        assert "so_1" in names
        assert "so_2" in names
