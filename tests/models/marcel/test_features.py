from fantasy_baseball_manager.features.types import Feature, Source
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_features,
    build_pitching_features,
)


class TestBuildBattingFeatures:
    def test_includes_age(self) -> None:
        features = build_batting_features(["hr"])
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_pa_lags(self) -> None:
        features = build_batting_features(["hr"])
        pa_features = [f for f in features if isinstance(f, Feature) and f.column == "pa"]
        assert len(pa_features) == 3
        assert {f.name for f in pa_features} == {"pa_1", "pa_2", "pa_3"}

    def test_includes_category_lags(self) -> None:
        features = build_batting_features(["hr", "h"])
        hr_features = [f for f in features if isinstance(f, Feature) and f.column == "hr"]
        assert len(hr_features) == 3
        assert {f.lag for f in hr_features} == {1, 2, 3}
        h_features = [f for f in features if isinstance(f, Feature) and f.column == "h"]
        assert len(h_features) == 3

    def test_feature_source_is_batting(self) -> None:
        features = build_batting_features(["hr"])
        batting_features = [f for f in features if isinstance(f, Feature) and f.source == Source.BATTING]
        # pa_1..3 + hr_1..3 = 6
        assert len(batting_features) == 6

    def test_custom_lags(self) -> None:
        features = build_batting_features(["hr"], lags=2)
        hr_features = [f for f in features if isinstance(f, Feature) and f.column == "hr"]
        assert len(hr_features) == 2
        assert {f.lag for f in hr_features} == {1, 2}

    def test_total_feature_count(self) -> None:
        cats = ["hr", "h", "bb"]
        features = build_batting_features(cats, lags=3)
        # age + 3 lags * (pa + 3 cats) = 1 + 3 * 4 = 13
        assert len(features) == 13


class TestBuildPitchingFeatures:
    def test_includes_age(self) -> None:
        features = build_pitching_features(["so"])
        age_features = [f for f in features if isinstance(f, Feature) and f.computed == "age"]
        assert len(age_features) == 1

    def test_includes_ip_g_gs_lags(self) -> None:
        features = build_pitching_features(["so"])
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        g_features = [f for f in features if isinstance(f, Feature) and f.column == "g"]
        gs_features = [f for f in features if isinstance(f, Feature) and f.column == "gs"]
        assert len(ip_features) == 3
        assert len(g_features) == 3
        assert len(gs_features) == 3

    def test_includes_category_lags(self) -> None:
        features = build_pitching_features(["so", "er"])
        so_features = [f for f in features if isinstance(f, Feature) and f.column == "so"]
        assert len(so_features) == 3
        er_features = [f for f in features if isinstance(f, Feature) and f.column == "er"]
        assert len(er_features) == 3

    def test_feature_source_is_pitching(self) -> None:
        features = build_pitching_features(["so"])
        pitching_features = [f for f in features if isinstance(f, Feature) and f.source == Source.PITCHING]
        # ip_1..3 + g_1..3 + gs_1..3 + so_1..3 = 12
        assert len(pitching_features) == 12

    def test_total_feature_count(self) -> None:
        cats = ["so", "er"]
        features = build_pitching_features(cats, lags=3)
        # age + 3 lags * (ip + g + gs + 2 cats) = 1 + 3 * 5 = 16
        assert len(features) == 16
