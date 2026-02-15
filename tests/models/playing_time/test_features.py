from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, Source
from fantasy_baseball_manager.models.playing_time.features import (
    build_batting_pt_derived_transforms,
    build_batting_pt_features,
    build_pitching_pt_derived_transforms,
    build_pitching_pt_features,
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
