from fantasy_baseball_manager.features import batting, player
from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, FeatureSet, Source, SpineFilter
from fantasy_baseball_manager.models.composite.features import (
    append_training_targets,
    batter_target_features,
    build_composite_batting_features,
    build_composite_pitching_features,
    pitcher_target_features,
)


class TestCompositeBattingFeatures:
    def test_includes_projection_pa(self) -> None:
        features = build_composite_batting_features(categories=("hr", "bb"), weights=(5.0, 4.0))
        proj_features = [f for f in features if isinstance(f, Feature) and f.source == Source.PROJECTION]
        assert len(proj_features) == 1
        assert proj_features[0].name == "proj_pa"
        assert proj_features[0].system == "playing_time"
        assert proj_features[0].column == "pa"

    def test_includes_age(self) -> None:
        features = build_composite_batting_features(categories=("hr",), weights=(5.0,))
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_stat_lags(self) -> None:
        features = build_composite_batting_features(categories=("hr", "bb"), weights=(5.0, 4.0))
        hr_features = [f for f in features if isinstance(f, Feature) and f.column == "hr"]
        assert len(hr_features) == 2
        bb_features = [f for f in features if isinstance(f, Feature) and f.column == "bb"]
        assert len(bb_features) == 2

    def test_includes_pa_lags(self) -> None:
        features = build_composite_batting_features(categories=("hr",), weights=(5.0, 4.0))
        pa_features = [
            f for f in features if isinstance(f, Feature) and f.column == "pa" and f.source == Source.BATTING
        ]
        assert len(pa_features) == 2

    def test_includes_weighted_rates_transform(self) -> None:
        features = build_composite_batting_features(categories=("hr",), weights=(5.0, 4.0))
        transforms = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [t.name for t in transforms]
        assert "batting_weighted_rates" in names

    def test_includes_league_averages_transform(self) -> None:
        features = build_composite_batting_features(categories=("hr",), weights=(5.0, 4.0))
        transforms = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [t.name for t in transforms]
        assert "batting_league_averages" in names


class TestCompositePitchingFeatures:
    def test_includes_projection_ip(self) -> None:
        features = build_composite_pitching_features(categories=("so", "bb"), weights=(3.0, 2.0))
        proj_features = [f for f in features if isinstance(f, Feature) and f.source == Source.PROJECTION]
        assert len(proj_features) == 1
        assert proj_features[0].name == "proj_ip"
        assert proj_features[0].system == "playing_time"
        assert proj_features[0].column == "ip"

    def test_includes_age(self) -> None:
        features = build_composite_pitching_features(categories=("so",), weights=(3.0,))
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_stat_lags(self) -> None:
        features = build_composite_pitching_features(categories=("so", "bb"), weights=(3.0, 2.0))
        so_features = [f for f in features if isinstance(f, Feature) and f.column == "so"]
        assert len(so_features) == 2

    def test_includes_ip_g_gs_lags(self) -> None:
        features = build_composite_pitching_features(categories=("so",), weights=(3.0, 2.0))
        ip_features = [
            f for f in features if isinstance(f, Feature) and f.column == "ip" and f.source == Source.PITCHING
        ]
        g_features = [f for f in features if isinstance(f, Feature) and f.column == "g"]
        gs_features = [f for f in features if isinstance(f, Feature) and f.column == "gs"]
        assert len(ip_features) == 2
        assert len(g_features) == 2
        assert len(gs_features) == 2

    def test_includes_weighted_rates_transform(self) -> None:
        features = build_composite_pitching_features(categories=("so",), weights=(3.0, 2.0))
        transforms = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [t.name for t in transforms]
        assert "pitching_weighted_rates" in names

    def test_includes_league_averages_transform(self) -> None:
        features = build_composite_pitching_features(categories=("so",), weights=(3.0, 2.0))
        transforms = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [t.name for t in transforms]
        assert "pitching_league_averages" in names


class TestBatterTargetFeatures:
    def test_returns_9_features(self) -> None:
        targets = batter_target_features()
        assert len(targets) == 9

    def test_all_target_prefix(self) -> None:
        targets = batter_target_features()
        for f in targets:
            assert f.name.startswith("target_")

    def test_all_lag_0(self) -> None:
        targets = batter_target_features()
        for f in targets:
            assert f.lag == 0

    def test_all_batting_source(self) -> None:
        targets = batter_target_features()
        for f in targets:
            assert f.source == Source.BATTING

    def test_expected_names(self) -> None:
        targets = batter_target_features()
        names = [f.name for f in targets]
        assert names == [
            "target_avg",
            "target_obp",
            "target_slg",
            "target_woba",
            "target_h",
            "target_hr",
            "target_ab",
            "target_so",
            "target_sf",
        ]


class TestPitcherTargetFeatures:
    def test_returns_9_features(self) -> None:
        targets = pitcher_target_features()
        assert len(targets) == 9

    def test_all_target_prefix(self) -> None:
        targets = pitcher_target_features()
        for f in targets:
            assert f.name.startswith("target_")

    def test_all_lag_0(self) -> None:
        targets = pitcher_target_features()
        for f in targets:
            assert f.lag == 0

    def test_all_pitching_source(self) -> None:
        targets = pitcher_target_features()
        for f in targets:
            assert f.source == Source.PITCHING

    def test_expected_names(self) -> None:
        targets = pitcher_target_features()
        names = [f.name for f in targets]
        assert names == [
            "target_era",
            "target_fip",
            "target_k_per_9",
            "target_bb_per_9",
            "target_whip",
            "target_h",
            "target_hr",
            "target_ip",
            "target_so",
        ]


class TestAppendTrainingTargets:
    def _make_prediction_fs(self) -> FeatureSet:
        return FeatureSet(
            name="composite_batting",
            features=(player.age(), batting.col("pa").lag(1).alias("pa_1")),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )

    def test_name_has_train_suffix(self) -> None:
        fs = self._make_prediction_fs()
        train_fs = append_training_targets(fs, batter_target_features())
        assert train_fs.name == "composite_batting_train"

    def test_features_include_originals_plus_targets(self) -> None:
        fs = self._make_prediction_fs()
        targets = batter_target_features()
        train_fs = append_training_targets(fs, targets)
        assert len(train_fs.features) == len(fs.features) + len(targets)
        # Original features come first
        assert train_fs.features[: len(fs.features)] == fs.features
        # Targets come after
        assert train_fs.features[len(fs.features) :] == tuple(targets)

    def test_preserves_seasons(self) -> None:
        fs = self._make_prediction_fs()
        train_fs = append_training_targets(fs, batter_target_features())
        assert train_fs.seasons == (2022, 2023)

    def test_preserves_source_filter(self) -> None:
        fs = self._make_prediction_fs()
        train_fs = append_training_targets(fs, batter_target_features())
        assert train_fs.source_filter == "fangraphs"

    def test_preserves_spine_filter(self) -> None:
        fs = self._make_prediction_fs()
        train_fs = append_training_targets(fs, batter_target_features())
        assert train_fs.spine_filter == SpineFilter(player_type="batter")
