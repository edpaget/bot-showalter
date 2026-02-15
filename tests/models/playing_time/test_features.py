from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, Source
from fantasy_baseball_manager.models.playing_time.features import (
    batting_pt_feature_columns,
    build_batting_pt_derived_transforms,
    build_batting_pt_features,
    build_batting_pt_training_features,
    build_pitching_pt_derived_transforms,
    build_pitching_pt_features,
    build_pitching_pt_training_features,
    pitching_pt_feature_columns,
)


class TestBattingPtFeatures:
    def test_includes_age(self) -> None:
        features = build_batting_pt_features()
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_lagged_pa(self) -> None:
        features = build_batting_pt_features()
        pa_features = [f for f in features if isinstance(f, Feature) and f.column == "pa"]
        assert len(pa_features) == 3
        lags = sorted(f.lag for f in pa_features)
        assert lags == [1, 2, 3]

    def test_includes_war(self) -> None:
        features = build_batting_pt_features()
        war_features = [f for f in features if isinstance(f, Feature) and f.column == "war"]
        assert len(war_features) == 2
        lags = sorted(f.lag for f in war_features)
        assert lags == [1, 2]

    def test_includes_il_days(self) -> None:
        features = build_batting_pt_features()
        il_days = [f for f in features if isinstance(f, Feature) and f.column == "days"]
        assert len(il_days) == 3
        lags = sorted(f.lag for f in il_days)
        assert lags == [1, 2, 3]

    def test_includes_il_stints(self) -> None:
        features = build_batting_pt_features()
        il_stints = [f for f in features if isinstance(f, Feature) and f.column == "stint_count"]
        assert len(il_stints) == 2
        lags = sorted(f.lag for f in il_stints)
        assert lags == [1, 2]

    def test_il_features_have_il_stint_source(self) -> None:
        features = build_batting_pt_features()
        il_features = [f for f in features if isinstance(f, Feature) and f.column in ("days", "stint_count")]
        assert all(f.source == Source.IL_STINT for f in il_features)

    def test_custom_lags(self) -> None:
        features = build_batting_pt_features(lags=4)
        pa_features = [f for f in features if isinstance(f, Feature) and f.column == "pa"]
        assert len(pa_features) == 4

    def test_custom_lags_affects_il_days(self) -> None:
        features = build_batting_pt_features(lags=4)
        il_days = [f for f in features if isinstance(f, Feature) and f.column == "days"]
        assert len(il_days) == 4

    def test_feature_source_is_batting(self) -> None:
        features = build_batting_pt_features()
        pa_features = [f for f in features if isinstance(f, Feature) and f.column == "pa"]
        assert all(f.source == Source.BATTING for f in pa_features)


class TestPitchingPtFeatures:
    def test_includes_age(self) -> None:
        features = build_pitching_pt_features()
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_lagged_ip(self) -> None:
        features = build_pitching_pt_features()
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        assert len(ip_features) == 3

    def test_includes_lagged_g(self) -> None:
        features = build_pitching_pt_features()
        g_features = [f for f in features if isinstance(f, Feature) and f.column == "g"]
        assert len(g_features) == 3

    def test_includes_gs_lag1_only(self) -> None:
        features = build_pitching_pt_features()
        gs_features = [f for f in features if isinstance(f, Feature) and f.column == "gs"]
        assert len(gs_features) == 1
        assert gs_features[0].lag == 1

    def test_includes_war(self) -> None:
        features = build_pitching_pt_features()
        war_features = [f for f in features if isinstance(f, Feature) and f.column == "war"]
        assert len(war_features) == 2
        lags = sorted(f.lag for f in war_features)
        assert lags == [1, 2]

    def test_includes_il_days(self) -> None:
        features = build_pitching_pt_features()
        il_days = [f for f in features if isinstance(f, Feature) and f.column == "days"]
        assert len(il_days) == 3

    def test_includes_il_stints(self) -> None:
        features = build_pitching_pt_features()
        il_stints = [f for f in features if isinstance(f, Feature) and f.column == "stint_count"]
        assert len(il_stints) == 2

    def test_il_features_have_il_stint_source(self) -> None:
        features = build_pitching_pt_features()
        il_features = [f for f in features if isinstance(f, Feature) and f.column in ("days", "stint_count")]
        assert all(f.source == Source.IL_STINT for f in il_features)

    def test_custom_lags(self) -> None:
        features = build_pitching_pt_features(lags=4)
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        assert len(ip_features) == 4

    def test_feature_source_is_pitching(self) -> None:
        features = build_pitching_pt_features()
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        assert all(f.source == Source.PITCHING for f in ip_features)


class TestBattingDerivedTransforms:
    def test_produces_two_transforms(self) -> None:
        transforms = build_batting_pt_derived_transforms()
        assert len(transforms) == 2

    def test_il_summary_outputs(self) -> None:
        transforms = build_batting_pt_derived_transforms()
        il_summary = next(t for t in transforms if t.name == "il_summary")
        assert "il_days_3yr" in il_summary.outputs
        assert "il_recurrence" in il_summary.outputs

    def test_pt_trend_outputs(self) -> None:
        transforms = build_batting_pt_derived_transforms()
        pt_trend = next(t for t in transforms if t.name == "batting_pt_trend")
        assert "pt_trend" in pt_trend.outputs

    def test_all_are_derived_transform_features(self) -> None:
        transforms = build_batting_pt_derived_transforms()
        assert all(isinstance(t, DerivedTransformFeature) for t in transforms)


class TestPitchingDerivedTransforms:
    def test_produces_two_transforms(self) -> None:
        transforms = build_pitching_pt_derived_transforms()
        assert len(transforms) == 2

    def test_il_summary_outputs(self) -> None:
        transforms = build_pitching_pt_derived_transforms()
        il_summary = next(t for t in transforms if t.name == "il_summary")
        assert "il_days_3yr" in il_summary.outputs
        assert "il_recurrence" in il_summary.outputs

    def test_pt_trend_outputs(self) -> None:
        transforms = build_pitching_pt_derived_transforms()
        pt_trend = next(t for t in transforms if t.name == "pitching_pt_trend")
        assert "pt_trend" in pt_trend.outputs

    def test_all_are_derived_transform_features(self) -> None:
        transforms = build_pitching_pt_derived_transforms()
        assert all(isinstance(t, DerivedTransformFeature) for t in transforms)


class TestBattingDerivedTransformsLags1:
    def test_il_summary_inputs_with_lags_1(self) -> None:
        transforms = build_batting_pt_derived_transforms(lags=1)
        il_summary = next(t for t in transforms if t.name == "il_summary")
        assert "il_stints_2" not in il_summary.inputs

    def test_pt_trend_inputs_with_lags_1(self) -> None:
        transforms = build_batting_pt_derived_transforms(lags=1)
        pt_trend = next(t for t in transforms if t.name == "batting_pt_trend")
        assert "pa_2" not in pt_trend.inputs


class TestPitchingDerivedTransformsLags1:
    def test_il_summary_inputs_with_lags_1(self) -> None:
        transforms = build_pitching_pt_derived_transforms(lags=1)
        il_summary = next(t for t in transforms if t.name == "il_summary")
        assert "il_stints_2" not in il_summary.inputs

    def test_pt_trend_inputs_with_lags_1(self) -> None:
        transforms = build_pitching_pt_derived_transforms(lags=1)
        pt_trend = next(t for t in transforms if t.name == "pitching_pt_trend")
        assert "ip_2" not in pt_trend.inputs


class TestBattingTrainingFeatures:
    def test_training_features_include_target_pa(self) -> None:
        features = build_batting_pt_training_features()
        names = [f.name for f in features if isinstance(f, Feature)]
        assert "target_pa" in names

    def test_training_features_target_has_lag_zero(self) -> None:
        features = build_batting_pt_training_features()
        target = [f for f in features if isinstance(f, Feature) and f.name == "target_pa"]
        assert len(target) == 1
        assert target[0].lag == 0

    def test_training_features_include_derived_transforms(self) -> None:
        features = build_batting_pt_training_features()
        derived = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [d.name for d in derived]
        assert "il_summary" in names
        assert "batting_pt_trend" in names


class TestPitchingTrainingFeatures:
    def test_training_features_include_target_ip(self) -> None:
        features = build_pitching_pt_training_features()
        names = [f.name for f in features if isinstance(f, Feature)]
        assert "target_ip" in names

    def test_training_features_target_has_lag_zero(self) -> None:
        features = build_pitching_pt_training_features()
        target = [f for f in features if isinstance(f, Feature) and f.name == "target_ip"]
        assert len(target) == 1
        assert target[0].lag == 0

    def test_training_features_include_derived_transforms(self) -> None:
        features = build_pitching_pt_training_features()
        derived = [f for f in features if isinstance(f, DerivedTransformFeature)]
        names = [d.name for d in derived]
        assert "il_summary" in names
        assert "pitching_pt_trend" in names


class TestBattingFeatureColumns:
    def test_feature_columns_excludes_target(self) -> None:
        columns = batting_pt_feature_columns()
        assert "target_pa" not in columns

    def test_feature_columns_includes_derived_outputs(self) -> None:
        columns = batting_pt_feature_columns()
        assert "il_days_3yr" in columns
        assert "il_recurrence" in columns
        assert "pt_trend" in columns

    def test_feature_columns_includes_base_features(self) -> None:
        columns = batting_pt_feature_columns()
        assert "age" in columns
        assert "pa_1" in columns

    def test_feature_columns_excludes_metadata(self) -> None:
        columns = batting_pt_feature_columns()
        assert "player_id" not in columns
        assert "season" not in columns


class TestPitchingFeatureColumns:
    def test_feature_columns_excludes_target(self) -> None:
        columns = pitching_pt_feature_columns()
        assert "target_ip" not in columns

    def test_feature_columns_includes_derived_outputs(self) -> None:
        columns = pitching_pt_feature_columns()
        assert "il_days_3yr" in columns
        assert "il_recurrence" in columns
        assert "pt_trend" in columns

    def test_feature_columns_includes_base_features(self) -> None:
        columns = pitching_pt_feature_columns()
        assert "age" in columns
        assert "ip_1" in columns

    def test_feature_columns_excludes_metadata(self) -> None:
        columns = pitching_pt_feature_columns()
        assert "player_id" not in columns
        assert "season" not in columns
