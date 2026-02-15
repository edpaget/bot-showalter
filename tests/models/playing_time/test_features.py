from fantasy_baseball_manager.features.types import Feature, Source
from fantasy_baseball_manager.models.playing_time.features import (
    build_batting_pt_features,
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
        assert len(pa_features) == 2
        lags = sorted(f.lag for f in pa_features)
        assert lags == [1, 2]

    def test_no_stat_categories(self) -> None:
        features = build_batting_pt_features()
        non_pa_non_age = [
            f for f in features if isinstance(f, Feature) and f.computed != "age" and f.column not in ("pa", "position")
        ]
        assert len(non_pa_non_age) == 0

    def test_custom_lags(self) -> None:
        features = build_batting_pt_features(lags=3)
        pa_features = [f for f in features if isinstance(f, Feature) and f.column == "pa"]
        assert len(pa_features) == 3

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
        assert len(ip_features) == 2

    def test_includes_lagged_g_and_gs(self) -> None:
        features = build_pitching_pt_features()
        g_features = [f for f in features if isinstance(f, Feature) and f.column == "g"]
        gs_features = [f for f in features if isinstance(f, Feature) and f.column == "gs"]
        assert len(g_features) == 2
        assert len(gs_features) == 2

    def test_no_stat_categories(self) -> None:
        features = build_pitching_pt_features()
        pt_columns = {"ip", "g", "gs"}
        non_pt_non_age = [
            f
            for f in features
            if isinstance(f, Feature) and f.computed != "age" and f.column not in pt_columns and f.column != "position"
        ]
        assert len(non_pt_non_age) == 0

    def test_custom_lags(self) -> None:
        features = build_pitching_pt_features(lags=3)
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        assert len(ip_features) == 3

    def test_feature_source_is_pitching(self) -> None:
        features = build_pitching_pt_features()
        ip_features = [f for f in features if isinstance(f, Feature) and f.column == "ip"]
        assert all(f.source == Source.PITCHING for f in ip_features)
