from __future__ import annotations

import sqlite3

import pytest

from tests.features.conftest import seed_batting_data

from fantasy_baseball_manager.features.sql import (
    JoinSpec,
    _join_alias,
    _plan_joins,
    _select_expr,
    _source_table,
    _join_clause,
    _spine_cte,
    generate_sql,
)
from fantasy_baseball_manager.features.types import Feature, FeatureSet, Source, SpineFilter


class TestSourceTable:
    def test_batting(self) -> None:
        assert _source_table(Source.BATTING) == "batting_stats"

    def test_pitching(self) -> None:
        assert _source_table(Source.PITCHING) == "pitching_stats"

    def test_player(self) -> None:
        assert _source_table(Source.PLAYER) == "player"

    def test_projection(self) -> None:
        assert _source_table(Source.PROJECTION) == "projection"


class TestJoinAlias:
    def test_batting_lag_0(self) -> None:
        assert _join_alias(Source.BATTING, 0) == "b0"

    def test_batting_lag_1(self) -> None:
        assert _join_alias(Source.BATTING, 1) == "b1"

    def test_pitching_lag_2(self) -> None:
        assert _join_alias(Source.PITCHING, 2) == "pi2"

    def test_player_always_p(self) -> None:
        assert _join_alias(Source.PLAYER, 0) == "p"

    def test_projection_lag_0(self) -> None:
        assert _join_alias(Source.PROJECTION, 0) == "pr0"


class TestPlanJoins:
    def test_single_feature_one_join(self) -> None:
        features = (Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),)
        joins = _plan_joins(features)
        assert len(joins) == 1
        assert joins[0] == JoinSpec(source=Source.BATTING, lag=1, alias="b1", table="batting_stats")

    def test_same_source_lag_deduplicated(self) -> None:
        features = (
            Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
            Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1),
        )
        joins = _plan_joins(features)
        assert len(joins) == 1

    def test_different_lags_separate_joins(self) -> None:
        features = (
            Feature(name="hr_0", source=Source.BATTING, column="hr", lag=0),
            Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
        )
        joins = _plan_joins(features)
        assert len(joins) == 2
        assert joins[0].lag == 0
        assert joins[1].lag == 1

    def test_player_always_lag_0(self) -> None:
        features = (Feature(name="bats", source=Source.PLAYER, column="bats"),)
        joins = _plan_joins(features)
        assert len(joins) == 1
        assert joins[0] == JoinSpec(source=Source.PLAYER, lag=0, alias="p", table="player")

    def test_mixed_sources(self) -> None:
        features = (
            Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
            Feature(name="so_0", source=Source.PITCHING, column="so", lag=0),
            Feature(name="bats", source=Source.PLAYER, column="bats"),
        )
        joins = _plan_joins(features)
        assert len(joins) == 3

    def test_computed_age_creates_player_join(self) -> None:
        features = (Feature(name="age", source=Source.PLAYER, column="", computed="age"),)
        joins = _plan_joins(features)
        assert len(joins) == 1
        assert joins[0].source == Source.PLAYER

    def test_rolling_aggregate_no_join(self) -> None:
        features = (
            Feature(
                name="hr_3yr",
                source=Source.BATTING,
                column="hr",
                lag=1,
                window=3,
                aggregate="mean",
            ),
        )
        joins = _plan_joins(features)
        assert len(joins) == 0

    def test_stable_ordering(self) -> None:
        features = (
            Feature(name="so_0", source=Source.PITCHING, column="so", lag=0),
            Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
            Feature(name="bats", source=Source.PLAYER, column="bats"),
        )
        joins = _plan_joins(features)
        # batting < pitching < player (alphabetical by source.value)
        assert joins[0].source == Source.BATTING
        assert joins[1].source == Source.PITCHING
        assert joins[2].source == Source.PLAYER


class TestSelectExprDirect:
    def _joins_dict(self) -> dict[tuple[Source, int], JoinSpec]:
        return {
            (Source.BATTING, 1): JoinSpec(source=Source.BATTING, lag=1, alias="b1", table="batting_stats"),
            (Source.PITCHING, 0): JoinSpec(source=Source.PITCHING, lag=0, alias="pi0", table="pitching_stats"),
            (Source.PLAYER, 0): JoinSpec(source=Source.PLAYER, lag=0, alias="p", table="player"),
        }

    def test_batting_direct(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        sql, params = _select_expr(f, self._joins_dict(), None)
        assert sql == "b1.hr AS hr_1"
        assert params == []

    def test_pitching_direct(self) -> None:
        f = Feature(name="so_0", source=Source.PITCHING, column="so", lag=0)
        sql, params = _select_expr(f, self._joins_dict(), None)
        assert sql == "pi0.so AS so_0"
        assert params == []

    def test_player_direct(self) -> None:
        f = Feature(name="bats", source=Source.PLAYER, column="bats")
        sql, params = _select_expr(f, self._joins_dict(), None)
        assert sql == "p.bats AS bats"
        assert params == []


class TestSelectExprAge:
    def test_computed_age(self) -> None:
        joins_dict: dict[tuple[Source, int], JoinSpec] = {
            (Source.PLAYER, 0): JoinSpec(source=Source.PLAYER, lag=0, alias="p", table="player"),
        }
        f = Feature(name="age", source=Source.PLAYER, column="", computed="age")
        sql, params = _select_expr(f, joins_dict, None)
        assert sql == "spine.season - CAST(SUBSTR(p.birth_date, 1, 4) AS INTEGER) AS age"
        assert params == []


class TestSelectExprRate:
    def test_rate_stat(self) -> None:
        joins_dict: dict[tuple[Source, int], JoinSpec] = {
            (Source.BATTING, 1): JoinSpec(source=Source.BATTING, lag=1, alias="b1", table="batting_stats"),
        }
        f = Feature(name="hr_rate", source=Source.BATTING, column="hr", lag=1, denominator="pa")
        sql, params = _select_expr(f, joins_dict, None)
        assert sql == "CAST(b1.hr AS REAL) / NULLIF(b1.pa, 0) AS hr_rate"
        assert params == []


class TestSelectExprRolling:
    def test_rolling_mean(self) -> None:
        f = Feature(
            name="hr_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
        )
        sql, params = _select_expr(f, {}, None)
        assert sql == (
            "(SELECT AVG(hr) FROM batting_stats"
            " WHERE player_id = spine.player_id"
            " AND season BETWEEN spine.season - 3 AND spine.season - 1) AS hr_3yr"
        )
        assert params == []

    def test_rolling_sum(self) -> None:
        f = Feature(
            name="hr_5yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=5,
            aggregate="sum",
        )
        sql, params = _select_expr(f, {}, None)
        assert sql == (
            "(SELECT SUM(hr) FROM batting_stats"
            " WHERE player_id = spine.player_id"
            " AND season BETWEEN spine.season - 5 AND spine.season - 1) AS hr_5yr"
        )
        assert params == []

    def test_rolling_with_source_filter(self) -> None:
        f = Feature(
            name="hr_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
        )
        sql, params = _select_expr(f, {}, "fangraphs")
        assert "AND source = ?" in sql
        assert params == ["fangraphs"]

    def test_rolling_without_source_filter(self) -> None:
        f = Feature(
            name="hr_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
        )
        sql, params = _select_expr(f, {}, None)
        assert "source" not in sql
        assert params == []

    def test_window_bounds_lag1_window3(self) -> None:
        f = Feature(
            name="hr_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
        )
        sql, _ = _select_expr(f, {}, None)
        assert "BETWEEN spine.season - 3 AND spine.season - 1" in sql

    def test_window_bounds_lag2_window3(self) -> None:
        f = Feature(
            name="hr_3yr",
            source=Source.BATTING,
            column="hr",
            lag=2,
            window=3,
            aggregate="mean",
        )
        sql, _ = _select_expr(f, {}, None)
        assert "BETWEEN spine.season - 4 AND spine.season - 2" in sql


class TestSelectExprRollingRate:
    def test_rolling_rate(self) -> None:
        f = Feature(
            name="hr_rate_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
            denominator="pa",
        )
        sql, params = _select_expr(f, {}, None)
        assert "CAST(" in sql
        assert "SUM(hr)" in sql
        assert "SUM(pa)" in sql
        assert "NULLIF(" in sql
        assert "BETWEEN spine.season - 3 AND spine.season - 1" in sql
        assert sql.endswith("AS hr_rate_3yr")
        assert params == []

    def test_rolling_rate_with_source_filter(self) -> None:
        f = Feature(
            name="hr_rate_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
            denominator="pa",
        )
        sql, params = _select_expr(f, {}, "fangraphs")
        # source filter appears twice — once for numerator, once for denominator
        assert sql.count("AND source = ?") == 2
        assert params == ["fangraphs", "fangraphs"]


class TestSpineCte:
    def test_basic_batter(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022, 2023),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, params = _spine_cte(fs)
        assert "SELECT DISTINCT player_id, season" in sql
        assert "FROM batting_stats" in sql
        assert "season IN (?, ?)" in sql
        assert params == [2022, 2023]

    def test_with_source_filter(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, params = _spine_cte(fs)
        assert "AND source = ?" in sql
        assert "fangraphs" in params

    def test_with_min_pa(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter", min_pa=50),
        )
        sql, params = _spine_cte(fs)
        assert "AND pa >= ?" in sql
        assert 50 in params

    def test_pitcher_with_min_ip(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="so_0", source=Source.PITCHING, column="so"),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="pitcher", min_ip=30.0),
        )
        sql, params = _spine_cte(fs)
        assert "FROM pitching_stats" in sql
        assert "AND ip >= ?" in sql
        assert 30.0 in params

    def test_player_type_batter(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, _ = _spine_cte(fs)
        assert "FROM batting_stats" in sql

    def test_player_type_pitcher(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="so_0", source=Source.PITCHING, column="so"),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        sql, _ = _spine_cte(fs)
        assert "FROM pitching_stats" in sql

    def test_all_filters_combined(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter", min_pa=50),
        )
        sql, params = _spine_cte(fs)
        assert "FROM batting_stats" in sql
        assert "season IN (?, ?)" in sql
        assert "AND source = ?" in sql
        assert "AND pa >= ?" in sql
        assert params == [2022, 2023, "fangraphs", 50]

    def test_infer_table_from_first_stat_feature(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="so_0", source=Source.PITCHING, column="so"),),
            seasons=(2022,),
        )
        sql, _ = _spine_cte(fs)
        assert "FROM pitching_stats" in sql

    def test_fallback_to_batting(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="age", source=Source.PLAYER, column="", computed="age"),),
            seasons=(2022,),
        )
        sql, _ = _spine_cte(fs)
        assert "FROM batting_stats" in sql


class TestJoinClause:
    def test_batting_lag1_with_source_filter(self) -> None:
        join = JoinSpec(source=Source.BATTING, lag=1, alias="b1", table="batting_stats")
        sql, params = _join_clause(join, "fangraphs")
        assert "LEFT JOIN batting_stats b1" in sql
        assert "b1.player_id = spine.player_id" in sql
        assert "b1.season = spine.season - 1" in sql
        assert "b1.source = ?" in sql
        assert params == ["fangraphs"]

    def test_batting_lag0(self) -> None:
        join = JoinSpec(source=Source.BATTING, lag=0, alias="b0", table="batting_stats")
        sql, params = _join_clause(join, None)
        assert "b0.season = spine.season" in sql
        # Should NOT have "- 0" in the output
        assert "- 0" not in sql
        assert params == []

    def test_no_source_filter(self) -> None:
        join = JoinSpec(source=Source.BATTING, lag=1, alias="b1", table="batting_stats")
        sql, params = _join_clause(join, None)
        assert "source" not in sql
        assert params == []

    def test_player_join(self) -> None:
        join = JoinSpec(source=Source.PLAYER, lag=0, alias="p", table="player")
        sql, params = _join_clause(join, "fangraphs")
        assert "LEFT JOIN player p" in sql
        assert "p.id = spine.player_id" in sql
        assert "season" not in sql
        assert "source" not in sql
        assert params == []

    def test_pitching_lag2(self) -> None:
        join = JoinSpec(source=Source.PITCHING, lag=2, alias="pi2", table="pitching_stats")
        sql, params = _join_clause(join, None)
        assert "pi2.season = spine.season - 2" in sql
        assert params == []


class TestGenerateSql:
    def test_single_direct_feature(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, params = generate_sql(fs)
        assert "WITH spine AS" in sql
        assert "spine.player_id" in sql
        assert "spine.season" in sql
        assert "b1.hr AS hr_1" in sql
        assert "LEFT JOIN batting_stats b1" in sql
        # Params: spine seasons + spine source_filter + join source_filter
        assert params == [2022, "fangraphs", "fangraphs"]

    def test_multiple_features_same_join(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(
                Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
                Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1),
            ),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, _ = generate_sql(fs)
        # Should be only one LEFT JOIN for batting lag 1
        assert sql.count("LEFT JOIN batting_stats b1") == 1
        assert "b1.hr AS hr_1" in sql
        assert "b1.pa AS pa_1" in sql

    def test_mixed_sources_and_types(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(
                Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
                Feature(name="age", source=Source.PLAYER, column="", computed="age"),
                Feature(
                    name="hr_3yr",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    window=3,
                    aggregate="mean",
                ),
            ),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, _ = generate_sql(fs)
        assert "b1.hr AS hr_1" in sql
        assert "AS age" in sql
        assert "AVG(hr)" in sql
        assert "LEFT JOIN batting_stats b1" in sql
        assert "LEFT JOIN player p" in sql

    def test_no_source_filter(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, params = generate_sql(fs)
        assert "source = ?" not in sql
        assert params == [2022]

    def test_always_has_player_id_and_season(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022,),
            spine_filter=SpineFilter(player_type="batter"),
        )
        sql, _ = generate_sql(fs)
        assert "spine.player_id" in sql
        assert "spine.season" in sql


class TestRoundTrip:
    """Integration tests that execute generated SQL against seeded in-memory SQLite."""

    @pytest.fixture(autouse=True)
    def _seed(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        self._conn = conn

    def _execute(self, fs: FeatureSet) -> list[tuple[object, ...]]:
        sql, params = generate_sql(fs)
        return self._conn.execute(sql, params).fetchall()

    def _execute_dicts(self, fs: FeatureSet) -> list[dict[str, object]]:
        sql, params = generate_sql(fs)
        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def test_direct_column_lag1(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        assert len(rows) == 4  # 2 players x 2 seasons
        # For 2022, lag=1 means 2021 data: Trout had 30 HR, Betts had 22
        trout_2022 = next(r for r in rows if r["player_id"] == 1 and r["season"] == 2022)
        assert trout_2022["hr_1"] == 30
        betts_2022 = next(r for r in rows if r["player_id"] == 2 and r["season"] == 2022)
        assert betts_2022["hr_1"] == 22

    def test_age_computation(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="age", source=Source.PLAYER, column="", computed="age"),),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        # 2023 - 1991 = 32
        assert trout["age"] == 32
        betts = next(r for r in rows if r["player_id"] == 2)
        # 2023 - 1992 = 31
        assert betts["age"] == 31

    def test_rate_stat(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(
                Feature(
                    name="hr_rate",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    denominator="pa",
                ),
            ),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        # 2022 data: 40 HR / 550 PA
        assert trout["hr_rate"] == pytest.approx(40 / 550)

    def test_rolling_mean(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(
                Feature(
                    name="hr_avg_3yr",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    window=3,
                    aggregate="mean",
                ),
            ),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        # lag=1, window=3 → seasons 2020, 2021, 2022 → HR: 17, 30, 40
        assert trout["hr_avg_3yr"] == pytest.approx((17 + 30 + 40) / 3)

    def test_rolling_rate(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(
                Feature(
                    name="hr_rate_3yr",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    window=3,
                    aggregate="mean",
                    denominator="pa",
                ),
            ),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        trout = next(r for r in rows if r["player_id"] == 1)
        # lag=1, window=3 → seasons 2020-2022
        # SUM(hr) = 17+30+40 = 87, SUM(pa) = 250+500+550 = 1300
        assert trout["hr_rate_3yr"] == pytest.approx(87 / 1300)

    def test_multiple_seasons_row_count(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2021, 2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute(fs)
        assert len(rows) == 6  # 2 players x 3 seasons

    def test_marcel_like_full_query(self) -> None:
        """All feature types together: direct, age, rate, rolling."""
        fs = FeatureSet(
            name="marcel",
            features=(
                Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
                Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1),
                Feature(name="age", source=Source.PLAYER, column="", computed="age"),
                Feature(
                    name="hr_rate",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    denominator="pa",
                ),
                Feature(
                    name="hr_avg_3yr",
                    source=Source.BATTING,
                    column="hr",
                    lag=1,
                    window=3,
                    aggregate="mean",
                ),
            ),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        assert len(rows) == 2
        trout = next(r for r in rows if r["player_id"] == 1)
        assert trout["hr_1"] == 40
        assert trout["pa_1"] == 550
        assert trout["age"] == 32
        assert trout["hr_rate"] == pytest.approx(40 / 550)
        assert trout["hr_avg_3yr"] == pytest.approx((17 + 30 + 40) / 3)

    def test_min_pa_filters(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_0", source=Source.BATTING, column="hr"),),
            seasons=(2020,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter", min_pa=220),
        )
        rows = self._execute_dicts(fs)
        # Trout had 250 PA in 2020, Betts had 200 → only Trout passes
        assert len(rows) == 1
        assert rows[0]["player_id"] == 1

    def test_missing_lag_data_returns_null(self) -> None:
        fs = FeatureSet(
            name="test",
            features=(Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),),
            seasons=(2020,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        rows = self._execute_dicts(fs)
        # Lag 1 from 2020 → need 2019 data, which doesn't exist
        for row in rows:
            assert row["hr_1"] is None
