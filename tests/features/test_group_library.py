import importlib
import sqlite3

import pytest

import fantasy_baseball_manager.features.group_library
from tests.features.conftest import (
    seed_batting_data,
    seed_marcel_projection,
    seed_mle_projection,
    seed_pitching_data,
    seed_statcast_gbm_projection,
)

from fantasy_baseball_manager.features.group_library import (
    make_batting_counting_lags,
    make_batting_rate_lags,
    make_pitching_counting_lags,
)
from fantasy_baseball_manager.features.groups import _clear, get_group, list_groups
from fantasy_baseball_manager.features.sql import generate_sql
from fantasy_baseball_manager.features.types import Feature, FeatureSet, Source, SpineFilter, TransformFeature


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()
    importlib.reload(fantasy_baseball_manager.features.group_library)


class TestStaticGroups:
    def test_all_static_groups_registered(self) -> None:
        names = list_groups()
        expected = [
            "age",
            "marcel_batter_rates",
            "marcel_pitcher_rates",
            "mle_batter_rates",
            "positions",
            "projected_batting_pt",
            "projected_pitching_pt",
            "statcast_batted_ball",
            "statcast_expected_stats",
            "statcast_gbm_batter_rates",
            "statcast_gbm_pitcher_rates",
            "statcast_gbm_preseason_batter_rates",
            "statcast_gbm_preseason_pitcher_rates",
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
        assert get_group("mle_batter_rates").player_type == "batter"
        assert get_group("statcast_gbm_batter_rates").player_type == "batter"
        assert get_group("statcast_gbm_pitcher_rates").player_type == "pitcher"
        assert get_group("statcast_gbm_preseason_batter_rates").player_type == "batter"
        assert get_group("statcast_gbm_preseason_pitcher_rates").player_type == "pitcher"
        assert get_group("marcel_batter_rates").player_type == "batter"
        assert get_group("marcel_pitcher_rates").player_type == "pitcher"

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

    def test_projection_groups_contain_projection_features(self) -> None:
        projection_group_names = [
            "mle_batter_rates",
            "statcast_gbm_batter_rates",
            "statcast_gbm_pitcher_rates",
            "statcast_gbm_preseason_batter_rates",
            "statcast_gbm_preseason_pitcher_rates",
            "marcel_batter_rates",
            "marcel_pitcher_rates",
        ]
        for name in projection_group_names:
            group = get_group(name)
            for f in group.features:
                assert isinstance(f, Feature), f"'{name}' feature '{f.name}' is not a Feature"
                assert f.source == Source.PROJECTION, f"'{name}' feature '{f.name}' is not PROJECTION"

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


class TestProjectionGroupRoundTrip:
    """Integration tests: projection feature groups materialize correctly."""

    def _execute_dicts(self, conn: sqlite3.Connection, fs: FeatureSet) -> list[dict[str, object]]:
        sql, params = generate_sql(fs)
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def test_mle_materializes_for_player_with_projection(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_mle_projection(conn)
        group = get_group("mle_batter_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(conn, fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        assert trout["mle_avg"] == pytest.approx(0.270)
        assert trout["mle_obp"] == pytest.approx(0.340)
        assert trout["mle_slg"] == pytest.approx(0.450)
        assert trout["mle_iso"] == pytest.approx(0.180)
        assert trout["mle_k_pct"] == pytest.approx(0.220)
        assert trout["mle_bb_pct"] == pytest.approx(0.095)
        assert trout["mle_babip"] == pytest.approx(0.310)
        assert trout["mle_pa"] == 450

    def test_mle_null_for_player_without_projection(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_mle_projection(conn)  # Only player 1 has MLE data
        group = get_group("mle_batter_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(conn, fs)
        betts = next(r for r in rows if r["player_id"] == 2)
        assert betts["mle_avg"] is None
        assert betts["mle_k_pct"] is None

    def test_statcast_gbm_batter_materializes_with_lag1(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_pitching_data(conn)
        seed_statcast_gbm_projection(conn)
        group = get_group("statcast_gbm_batter_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(conn, fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        # lag=1 from season 2023 → reads season 2022 statcast-gbm projection
        assert trout["sc_gbm_avg"] == pytest.approx(0.275)
        assert trout["sc_gbm_obp"] == pytest.approx(0.350)
        assert trout["sc_gbm_slg"] == pytest.approx(0.470)
        assert trout["sc_gbm_iso"] == pytest.approx(0.185)
        assert trout["sc_gbm_babip"] == pytest.approx(0.300)

    def test_statcast_gbm_pitcher_materializes(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)  # Needed for spine
        seed_pitching_data(conn)
        seed_statcast_gbm_projection(conn)
        group = get_group("statcast_gbm_pitcher_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        rows = self._execute_dicts(conn, fs)
        cole = next(r for r in rows if r["player_id"] == 3)
        # lag=1 from season 2023 → reads season 2022 statcast-gbm pitcher projection
        assert cole["sc_gbm_era"] == pytest.approx(3.20)
        assert cole["sc_gbm_fip"] == pytest.approx(3.10)
        assert cole["sc_gbm_k_per_9"] == pytest.approx(9.5)
        assert cole["sc_gbm_bb_per_9"] == pytest.approx(2.8)
        assert cole["sc_gbm_hr_per_9"] == pytest.approx(1.0)
        assert cole["sc_gbm_babip"] == pytest.approx(0.290)
        assert cole["sc_gbm_whip"] == pytest.approx(1.15)

    def test_marcel_batter_materializes(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_pitching_data(conn)
        seed_marcel_projection(conn)
        group = get_group("marcel_batter_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(conn, fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        assert trout["marcel_avg"] == pytest.approx(0.280)
        assert trout["marcel_obp"] == pytest.approx(0.355)
        assert trout["marcel_slg"] == pytest.approx(0.460)
        assert trout["marcel_ops"] == pytest.approx(0.840)
        assert trout["marcel_pa"] == 580

    def test_marcel_pitcher_materializes(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)  # Needed for spine
        seed_pitching_data(conn)
        seed_marcel_projection(conn)
        group = get_group("marcel_pitcher_rates")
        fs = FeatureSet(
            name="test",
            features=group.features,
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        rows = self._execute_dicts(conn, fs)
        cole = next(r for r in rows if r["player_id"] == 3)
        assert cole["marcel_era"] == pytest.approx(3.40)
        assert cole["marcel_whip"] == pytest.approx(1.18)
        assert cole["marcel_k_per_9"] == pytest.approx(9.0)
        assert cole["marcel_bb_per_9"] == pytest.approx(2.9)
        assert cole["marcel_ip"] == pytest.approx(190.0)


class TestPreseasonStatcastGbmGroups:
    def test_batter_group_registered_with_6_features(self) -> None:
        group = get_group("statcast_gbm_preseason_batter_rates")
        assert len(group.features) == 6

    def test_batter_group_all_projection_source(self) -> None:
        group = get_group("statcast_gbm_preseason_batter_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.source == Source.PROJECTION

    def test_batter_group_all_preseason_system(self) -> None:
        group = get_group("statcast_gbm_preseason_batter_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.system == "statcast-gbm-preseason"

    def test_batter_group_all_lag_0(self) -> None:
        group = get_group("statcast_gbm_preseason_batter_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.lag == 0

    def test_batter_group_aliases(self) -> None:
        group = get_group("statcast_gbm_preseason_batter_rates")
        names = [f.name for f in group.features]
        assert names == ["sc_pre_avg", "sc_pre_obp", "sc_pre_slg", "sc_pre_woba", "sc_pre_iso", "sc_pre_babip"]

    def test_pitcher_group_registered_with_7_features(self) -> None:
        group = get_group("statcast_gbm_preseason_pitcher_rates")
        assert len(group.features) == 7

    def test_pitcher_group_all_projection_source(self) -> None:
        group = get_group("statcast_gbm_preseason_pitcher_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.source == Source.PROJECTION

    def test_pitcher_group_all_preseason_system(self) -> None:
        group = get_group("statcast_gbm_preseason_pitcher_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.system == "statcast-gbm-preseason"

    def test_pitcher_group_all_lag_0(self) -> None:
        group = get_group("statcast_gbm_preseason_pitcher_rates")
        for f in group.features:
            assert isinstance(f, Feature)
            assert f.lag == 0

    def test_pitcher_group_aliases(self) -> None:
        group = get_group("statcast_gbm_preseason_pitcher_rates")
        names = [f.name for f in group.features]
        assert names == [
            "sc_pre_era",
            "sc_pre_fip",
            "sc_pre_k_per_9",
            "sc_pre_bb_per_9",
            "sc_pre_hr_per_9",
            "sc_pre_babip",
            "sc_pre_whip",
        ]
