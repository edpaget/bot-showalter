from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, Source
from fantasy_baseball_manager.models.composite.features import (
    build_composite_batting_features,
    build_composite_pitching_features,
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
