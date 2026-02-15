from fantasy_baseball_manager.features import RowTransform, batting, pitching, player
from fantasy_baseball_manager.features.types import (
    Feature,
    FeatureSet,
    RowTransform as RowTransformDirect,
    Source,
    SourceRef,
    SpineFilter,
)


class TestPublicImports:
    def test_batting_is_source_ref(self) -> None:
        assert isinstance(batting, SourceRef)
        assert batting.source == Source.BATTING

    def test_pitching_is_source_ref(self) -> None:
        assert isinstance(pitching, SourceRef)
        assert pitching.source == Source.PITCHING

    def test_player_is_source_ref(self) -> None:
        assert isinstance(player, SourceRef)
        assert player.source == Source.PLAYER

    def test_row_transform_exported(self) -> None:
        assert RowTransform is RowTransformDirect


class TestEndToEnd:
    def test_marcel_example(self) -> None:
        features = [
            batting.col("pa").lag(1).alias("pa_1"),
            batting.col("pa").lag(2).alias("pa_2"),
            batting.col("pa").lag(3).alias("pa_3"),
            batting.col("hr").lag(1).alias("hr_1"),
            batting.col("hr").lag(2).alias("hr_2"),
            batting.col("hr").lag(3).alias("hr_3"),
            batting.col("bb").lag(1).alias("bb_1"),
            batting.col("hr").per("pa").rolling_mean(3).alias("hr_rate_3yr"),
            player.age(),
            player.col("bats").alias("bats"),
            player.col("position").alias("position"),
            batting.col("hr").lag(0).alias("hr_next"),
        ]

        for f in features:
            assert isinstance(f, Feature)

        feature_set = FeatureSet(
            name="marcel_batting",
            features=tuple(features),
            seasons=(2022, 2023, 2024),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_pa=50, player_type="batter"),
        )

        assert feature_set.name == "marcel_batting"
        assert len(feature_set.features) == 12
        assert feature_set.version
        assert len(feature_set.version) == 12

    def test_feature_set_version_computed_from_features(self) -> None:
        features_a = (
            batting.col("hr").lag(1).alias("hr_1"),
            batting.col("pa").lag(1).alias("pa_1"),
        )
        features_b = (
            batting.col("hr").lag(1).alias("hr_1"),
            batting.col("pa").lag(1).alias("pa_1"),
        )
        fs_a = FeatureSet(name="test", features=features_a, seasons=(2022,))
        fs_b = FeatureSet(name="test", features=features_b, seasons=(2022,))
        assert fs_a.version == fs_b.version
