from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, Source
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_features,
    build_batting_league_averages,
    build_batting_weighted_rates,
    build_pitching_features,
    build_pitching_league_averages,
    build_pitching_weighted_rates,
)


class TestBuildBattingFeatures:
    def test_includes_position(self) -> None:
        features = build_batting_features(["hr"])
        pos_features = [f for f in features if isinstance(f, Feature) and f.name == "position"]
        assert len(pos_features) == 1
        assert pos_features[0].source == Source.PLAYER
        assert pos_features[0].column == "position"

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
        # age + position + 3 lags * (pa + 3 cats) = 2 + 3 * 4 = 14
        assert len(features) == 14


class TestBuildPitchingFeatures:
    def test_includes_position(self) -> None:
        features = build_pitching_features(["so"])
        pos_features = [f for f in features if isinstance(f, Feature) and f.name == "position"]
        assert len(pos_features) == 1
        assert pos_features[0].source == Source.PLAYER
        assert pos_features[0].column == "position"

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
        # age + position + 3 lags * (ip + g + gs + 2 cats) = 2 + 3 * 5 = 17
        assert len(features) == 17


class TestBuildBattingWeightedRates:
    def test_returns_derived_transform_feature(self) -> None:
        dtf = build_batting_weighted_rates(("hr", "h"), weights=(5.0, 4.0, 3.0))
        assert isinstance(dtf, DerivedTransformFeature)

    def test_group_by_player_season(self) -> None:
        dtf = build_batting_weighted_rates(("hr",), weights=(5.0, 4.0, 3.0))
        assert dtf.group_by == ("player_id", "season")

    def test_inputs_include_category_and_pa_lags(self) -> None:
        dtf = build_batting_weighted_rates(("hr", "h"), weights=(5.0, 4.0, 3.0))
        expected_inputs = {"hr_1", "hr_2", "hr_3", "h_1", "h_2", "h_3", "pa_1", "pa_2", "pa_3"}
        assert set(dtf.inputs) == expected_inputs

    def test_outputs_include_wavg_and_weighted_pt(self) -> None:
        dtf = build_batting_weighted_rates(("hr", "h"), weights=(5.0, 4.0, 3.0))
        assert "hr_wavg" in dtf.outputs
        assert "h_wavg" in dtf.outputs
        assert "weighted_pt" in dtf.outputs

    def test_output_count(self) -> None:
        dtf = build_batting_weighted_rates(("hr", "h", "bb"), weights=(5.0, 4.0, 3.0))
        # 3 categories + weighted_pt = 4
        assert len(dtf.outputs) == 4

    def test_version_derived_from_weights(self) -> None:
        dtf1 = build_batting_weighted_rates(("hr",), weights=(5.0, 4.0, 3.0))
        dtf2 = build_batting_weighted_rates(("hr",), weights=(3.0, 2.0, 1.0))
        assert dtf1.version is not None
        assert dtf2.version is not None
        assert dtf1.version != dtf2.version

    def test_same_weights_same_version(self) -> None:
        dtf1 = build_batting_weighted_rates(("hr",), weights=(5.0, 4.0, 3.0))
        dtf2 = build_batting_weighted_rates(("hr",), weights=(5.0, 4.0, 3.0))
        assert dtf1.version == dtf2.version

    def test_custom_lags(self) -> None:
        dtf = build_batting_weighted_rates(("hr",), weights=(5.0, 4.0))
        # 2 weights → 2 lags: hr_1, hr_2, pa_1, pa_2
        assert set(dtf.inputs) == {"hr_1", "hr_2", "pa_1", "pa_2"}


class TestBuildPitchingWeightedRates:
    def test_returns_derived_transform_feature(self) -> None:
        dtf = build_pitching_weighted_rates(("so",), weights=(3.0, 2.0, 1.0))
        assert isinstance(dtf, DerivedTransformFeature)

    def test_inputs_include_ip_lags(self) -> None:
        dtf = build_pitching_weighted_rates(("so",), weights=(3.0, 2.0, 1.0))
        assert "ip_1" in dtf.inputs
        assert "ip_2" in dtf.inputs
        assert "ip_3" in dtf.inputs

    def test_group_by_player_season(self) -> None:
        dtf = build_pitching_weighted_rates(("so",), weights=(3.0, 2.0, 1.0))
        assert dtf.group_by == ("player_id", "season")

    def test_outputs_include_wavg_and_weighted_pt(self) -> None:
        dtf = build_pitching_weighted_rates(("so", "er"), weights=(3.0, 2.0, 1.0))
        assert "so_wavg" in dtf.outputs
        assert "er_wavg" in dtf.outputs
        assert "weighted_pt" in dtf.outputs


class TestBuildBattingLeagueAverages:
    def test_returns_derived_transform_feature(self) -> None:
        dtf = build_batting_league_averages(("hr", "h"))
        assert isinstance(dtf, DerivedTransformFeature)

    def test_group_by_season_only(self) -> None:
        dtf = build_batting_league_averages(("hr",))
        assert dtf.group_by == ("season",)

    def test_inputs_are_lag1_columns(self) -> None:
        dtf = build_batting_league_averages(("hr", "h", "bb"))
        expected = {"hr_1", "h_1", "bb_1", "pa_1"}
        assert set(dtf.inputs) == expected

    def test_outputs_are_league_rates(self) -> None:
        dtf = build_batting_league_averages(("hr", "h"))
        assert "league_hr_rate" in dtf.outputs
        assert "league_h_rate" in dtf.outputs

    def test_output_count(self) -> None:
        dtf = build_batting_league_averages(("hr", "h", "bb"))
        assert len(dtf.outputs) == 3


class TestBuildPitchingLeagueAverages:
    def test_returns_derived_transform_feature(self) -> None:
        dtf = build_pitching_league_averages(("so", "er"))
        assert isinstance(dtf, DerivedTransformFeature)

    def test_group_by_season_only(self) -> None:
        dtf = build_pitching_league_averages(("so",))
        assert dtf.group_by == ("season",)

    def test_inputs_include_ip_lag1(self) -> None:
        dtf = build_pitching_league_averages(("so",))
        assert "ip_1" in dtf.inputs
        assert "so_1" in dtf.inputs

    def test_outputs_are_league_rates(self) -> None:
        dtf = build_pitching_league_averages(("so", "er"))
        assert "league_so_rate" in dtf.outputs
        assert "league_er_rate" in dtf.outputs
