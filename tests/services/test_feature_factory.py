from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import Err, Ok
from fantasy_baseball_manager.domain.feature_candidate import CandidateValue, FeatureCandidate
from fantasy_baseball_manager.repos.feature_candidate_repo import SqliteFeatureCandidateRepo
from fantasy_baseball_manager.services.feature_factory import (
    aggregate_candidate,
    candidate_values_to_dict,
    interact_candidates,
    resolve_feature,
    validate_expression,
)

if TYPE_CHECKING:
    import sqlite3


class TestValidateExpression:
    def test_allows_avg(self) -> None:
        result = validate_expression("AVG(launch_speed)")
        assert isinstance(result, Ok)

    def test_allows_filter_where(self) -> None:
        result = validate_expression("AVG(launch_speed) FILTER (WHERE barrel = 1)")
        assert isinstance(result, Ok)

    def test_allows_count_ratio(self) -> None:
        expr = "COUNT(*) FILTER (WHERE description = 'swinging_strike') * 1.0 / COUNT(*)"
        result = validate_expression(expr)
        assert isinstance(result, Ok)

    def test_allows_case(self) -> None:
        expr = "AVG(CASE WHEN barrel = 1 THEN launch_speed END)"
        result = validate_expression(expr)
        assert isinstance(result, Ok)

    def test_allows_sum(self) -> None:
        result = validate_expression("SUM(hit_distance_sc)")
        assert isinstance(result, Ok)

    def test_allows_coalesce(self) -> None:
        result = validate_expression("COALESCE(AVG(launch_speed), 0)")
        assert isinstance(result, Ok)

    def test_allows_iif(self) -> None:
        result = validate_expression("IIF(COUNT(*) > 0, SUM(barrel) * 1.0 / COUNT(*), 0)")
        assert isinstance(result, Ok)

    def test_allows_round(self) -> None:
        result = validate_expression("ROUND(AVG(launch_speed), 2)")
        assert isinstance(result, Ok)

    def test_rejects_drop(self) -> None:
        result = validate_expression("DROP TABLE statcast_pitch")
        assert isinstance(result, Err)
        assert "DROP" in result.error

    def test_rejects_insert(self) -> None:
        result = validate_expression("INSERT INTO foo VALUES (1)")
        assert isinstance(result, Err)

    def test_rejects_union(self) -> None:
        result = validate_expression("AVG(launch_speed) UNION SELECT * FROM player")
        assert isinstance(result, Err)

    def test_rejects_semicolons(self) -> None:
        result = validate_expression("AVG(launch_speed); DROP TABLE statcast_pitch")
        assert isinstance(result, Err)
        assert "semicolon" in result.error.lower()

    def test_rejects_empty(self) -> None:
        result = validate_expression("")
        assert isinstance(result, Err)

    def test_rejects_whitespace_only(self) -> None:
        result = validate_expression("   ")
        assert isinstance(result, Err)

    def test_rejects_select(self) -> None:
        result = validate_expression("SELECT AVG(launch_speed) FROM statcast_pitch")
        assert isinstance(result, Err)

    def test_rejects_delete(self) -> None:
        result = validate_expression("DELETE FROM statcast_pitch")
        assert isinstance(result, Err)

    def test_rejects_update(self) -> None:
        result = validate_expression("UPDATE statcast_pitch SET launch_speed = 0")
        assert isinstance(result, Err)

    def test_rejects_alter(self) -> None:
        result = validate_expression("ALTER TABLE statcast_pitch ADD COLUMN foo")
        assert isinstance(result, Err)

    def test_rejects_attach(self) -> None:
        result = validate_expression("ATTACH DATABASE '/tmp/evil.db' AS evil")
        assert isinstance(result, Err)

    def test_rejects_pragma(self) -> None:
        result = validate_expression("PRAGMA table_info(statcast_pitch)")
        assert isinstance(result, Err)

    def test_case_insensitive_rejection(self) -> None:
        result = validate_expression("drop TABLE foo")
        assert isinstance(result, Err)

    def test_allows_abs(self) -> None:
        result = validate_expression("ABS(AVG(launch_speed) - 90)")
        assert isinstance(result, Ok)

    def test_allows_cast(self) -> None:
        result = validate_expression("CAST(SUM(barrel) AS REAL) / COUNT(*)")
        assert isinstance(result, Ok)

    def test_allows_nullif(self) -> None:
        result = validate_expression("AVG(launch_speed) / NULLIF(COUNT(*), 0)")
        assert isinstance(result, Ok)

    def test_allows_total(self) -> None:
        result = validate_expression("TOTAL(hit_distance_sc)")
        assert isinstance(result, Ok)

    def test_allows_group_concat(self) -> None:
        result = validate_expression("GROUP_CONCAT(events, ',')")
        assert isinstance(result, Ok)

    def test_allows_length(self) -> None:
        result = validate_expression("AVG(LENGTH(events))")
        assert isinstance(result, Ok)

    def test_allows_substr(self) -> None:
        result = validate_expression("COUNT(*) FILTER (WHERE SUBSTR(events, 1, 4) = 'home')")
        assert isinstance(result, Ok)


class TestAggregateCandidate:
    @pytest.fixture
    def statcast_conn(self) -> sqlite3.Connection:
        conn = create_statcast_connection(":memory:")
        # Insert test data: 2 batters, 2 seasons
        rows = [
            # Batter 1, 2023
            (1, "2023-06-01", 100, 200, 1, 1, 90.0, 1, "single", "hit_into_play"),
            (2, "2023-06-02", 100, 201, 1, 1, 95.0, 0, "field_out", "hit_into_play"),
            (3, "2023-06-03", 100, 202, 1, 1, 88.0, 0, None, "swinging_strike"),
            # Batter 1, 2024
            (4, "2024-04-01", 100, 200, 1, 1, 92.0, 1, "double", "hit_into_play"),
            (5, "2024-04-02", 100, 201, 1, 1, 97.0, 1, "home_run", "hit_into_play"),
            # Batter 2, 2023
            (6, "2023-07-01", 300, 200, 1, 1, 85.0, 0, "field_out", "hit_into_play"),
            (7, "2023-07-02", 300, 201, 1, 1, 82.0, 0, "strikeout", "swinging_strike"),
            # Pitcher 200, 2023 (same data, pitcher perspective)
        ]
        for row in rows:
            conn.execute(
                """INSERT INTO statcast_pitch
                   (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
                    launch_speed, barrel, events, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
        conn.commit()
        return conn

    def test_avg_launch_speed(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(statcast_conn, "AVG(launch_speed)", [2023], "batter")
        # Batter 100: avg(90, 95, 88) = 91.0
        batter_100 = [r for r in results if r.player_id == 100 and r.season == 2023]
        assert len(batter_100) == 1
        assert batter_100[0].value is not None
        assert abs(batter_100[0].value - 91.0) < 0.01

    def test_filter_where_barrel(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(
            statcast_conn,
            "AVG(launch_speed) FILTER (WHERE barrel = 1)",
            [2023],
            "batter",
        )
        # Batter 100: only 1 barrel pitch with launch_speed=90
        batter_100 = [r for r in results if r.player_id == 100 and r.season == 2023]
        assert len(batter_100) == 1
        assert batter_100[0].value is not None
        assert abs(batter_100[0].value - 90.0) < 0.01

    def test_count_ratio(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(
            statcast_conn,
            "COUNT(*) FILTER (WHERE description = 'swinging_strike') * 1.0 / COUNT(*)",
            [2023],
            "batter",
        )
        # Batter 100: 1 swinging_strike / 3 total = 0.333
        batter_100 = [r for r in results if r.player_id == 100 and r.season == 2023]
        assert len(batter_100) == 1
        assert batter_100[0].value is not None
        assert abs(batter_100[0].value - 1.0 / 3.0) < 0.01

    def test_null_handling(self, statcast_conn: sqlite3.Connection) -> None:
        """Players with NULL aggregation results should get value=None, not 0."""
        results = aggregate_candidate(
            statcast_conn,
            "AVG(launch_speed) FILTER (WHERE barrel = 1)",
            [2023],
            "batter",
        )
        # Batter 300: no barrels, so FILTER result is NULL
        batter_300 = [r for r in results if r.player_id == 300 and r.season == 2023]
        assert len(batter_300) == 1
        assert batter_300[0].value is None

    def test_min_pa_filtering(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(
            statcast_conn,
            "AVG(launch_speed)",
            [2023],
            "batter",
            min_pa=3,
        )
        # Batter 100 has 2 events (single + field_out), batter 300 has 2 events
        # min_pa=3 should exclude both since events count < 3
        # But let's be specific: batter 100 has events on 2 rows (single, field_out)
        # With min_pa=3, neither qualifies
        player_ids = {r.player_id for r in results}
        # Batter 300 only has 2 events (field_out, strikeout), so excluded
        assert 300 not in player_ids

    def test_min_ip_filtering(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(
            statcast_conn,
            "AVG(launch_speed)",
            [2023],
            "pitcher",
            min_ip=100.0,
        )
        # All pitchers have very few batters faced, so none should qualify
        assert results == []

    def test_invalid_expression_raises(self, statcast_conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="DROP"):
            aggregate_candidate(statcast_conn, "DROP TABLE statcast_pitch", [2023], "batter")

    def test_invalid_player_type_raises(self, statcast_conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="player_type"):
            aggregate_candidate(statcast_conn, "AVG(launch_speed)", [2023], "catcher")

    def test_pitcher_groups_by_pitcher_id(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(statcast_conn, "AVG(launch_speed)", [2023], "pitcher")
        # Should have pitcher IDs, not batter IDs
        player_ids = {r.player_id for r in results}
        assert 200 in player_ids or 201 in player_ids
        # Should NOT have batter IDs as player_id
        # Actually batter_id 100 and 300 could coincidentally match pitcher IDs
        # but pitcher 200 should definitely be present
        assert 200 in player_ids

    def test_multiple_seasons(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(statcast_conn, "AVG(launch_speed)", [2023, 2024], "batter")
        # Batter 100 should appear in both seasons
        batter_100_seasons = {r.season for r in results if r.player_id == 100}
        assert 2023 in batter_100_seasons
        assert 2024 in batter_100_seasons

    def test_returns_candidate_value_instances(self, statcast_conn: sqlite3.Connection) -> None:
        results = aggregate_candidate(statcast_conn, "AVG(launch_speed)", [2023], "batter")
        assert all(isinstance(r, CandidateValue) for r in results)


class TestInteractCandidates:
    def test_product(self) -> None:
        a = [CandidateValue(1, 2023, 2.0), CandidateValue(2, 2023, 3.0)]
        b = [CandidateValue(1, 2023, 4.0), CandidateValue(2, 2023, 5.0)]
        result = interact_candidates(a, b, "product")
        by_pid = {r.player_id: r for r in result}
        assert by_pid[1].value == pytest.approx(8.0)
        assert by_pid[2].value == pytest.approx(15.0)

    def test_sum(self) -> None:
        a = [CandidateValue(1, 2023, 2.0), CandidateValue(2, 2023, 3.0)]
        b = [CandidateValue(1, 2023, 4.0), CandidateValue(2, 2023, 5.0)]
        result = interact_candidates(a, b, "sum")
        by_pid = {r.player_id: r for r in result}
        assert by_pid[1].value == pytest.approx(6.0)
        assert by_pid[2].value == pytest.approx(8.0)

    def test_difference(self) -> None:
        a = [CandidateValue(1, 2023, 10.0), CandidateValue(2, 2023, 3.0)]
        b = [CandidateValue(1, 2023, 4.0), CandidateValue(2, 2023, 5.0)]
        result = interact_candidates(a, b, "difference")
        by_pid = {r.player_id: r for r in result}
        assert by_pid[1].value == pytest.approx(6.0)
        assert by_pid[2].value == pytest.approx(-2.0)

    def test_ratio(self) -> None:
        a = [CandidateValue(1, 2023, 10.0), CandidateValue(2, 2023, 6.0)]
        b = [CandidateValue(1, 2023, 2.0), CandidateValue(2, 2023, 3.0)]
        result = interact_candidates(a, b, "ratio")
        by_pid = {r.player_id: r for r in result}
        assert by_pid[1].value == pytest.approx(5.0)
        assert by_pid[2].value == pytest.approx(2.0)

    def test_null_a_propagates(self) -> None:
        a = [CandidateValue(1, 2023, None)]
        b = [CandidateValue(1, 2023, 4.0)]
        result = interact_candidates(a, b, "product")
        assert result[0].value is None

    def test_null_b_propagates(self) -> None:
        a = [CandidateValue(1, 2023, 2.0)]
        b = [CandidateValue(1, 2023, None)]
        result = interact_candidates(a, b, "sum")
        assert result[0].value is None

    def test_both_null(self) -> None:
        a = [CandidateValue(1, 2023, None)]
        b = [CandidateValue(1, 2023, None)]
        result = interact_candidates(a, b, "difference")
        assert result[0].value is None

    def test_ratio_divide_by_zero_returns_none(self) -> None:
        a = [CandidateValue(1, 2023, 10.0)]
        b = [CandidateValue(1, 2023, 0.0)]
        result = interact_candidates(a, b, "ratio")
        assert result[0].value is None

    def test_ratio_null_denominator_returns_none(self) -> None:
        a = [CandidateValue(1, 2023, 10.0)]
        b = [CandidateValue(1, 2023, None)]
        result = interact_candidates(a, b, "ratio")
        assert result[0].value is None

    def test_only_matching_keys_included(self) -> None:
        a = [CandidateValue(1, 2023, 2.0), CandidateValue(3, 2023, 5.0)]
        b = [CandidateValue(1, 2023, 4.0), CandidateValue(2, 2023, 6.0)]
        result = interact_candidates(a, b, "product")
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_multi_season_join(self) -> None:
        a = [CandidateValue(1, 2023, 2.0), CandidateValue(1, 2024, 3.0)]
        b = [CandidateValue(1, 2023, 4.0), CandidateValue(1, 2024, 5.0)]
        result = interact_candidates(a, b, "product")
        by_season = {r.season: r for r in result}
        assert by_season[2023].value == pytest.approx(8.0)
        assert by_season[2024].value == pytest.approx(15.0)

    def test_invalid_operation_raises(self) -> None:
        a = [CandidateValue(1, 2023, 2.0)]
        b = [CandidateValue(1, 2023, 4.0)]
        with pytest.raises(ValueError, match="operation"):
            interact_candidates(a, b, "modulo")


class TestResolveFeature:
    @pytest.fixture
    def statcast_conn(self) -> sqlite3.Connection:
        conn = create_statcast_connection(":memory:")
        rows = [
            (1, "2023-06-01", 100, 200, 1, 1, 90.0, 1, "single", "hit_into_play"),
            (2, "2023-06-02", 100, 200, 1, 1, 95.0, 0, "field_out", "hit_into_play"),
            (3, "2023-07-01", 300, 200, 1, 1, 85.0, 0, "field_out", "hit_into_play"),
        ]
        for row in rows:
            conn.execute(
                """INSERT INTO statcast_pitch
                   (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
                    launch_speed, barrel, events, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
        conn.commit()
        return conn

    @pytest.fixture
    def fbm_conn(self) -> sqlite3.Connection:
        return create_connection(":memory:")

    def test_resolves_named_candidate(self, statcast_conn: sqlite3.Connection, fbm_conn: sqlite3.Connection) -> None:
        repo = SqliteFeatureCandidateRepo(fbm_conn)
        repo.save(
            FeatureCandidate(
                name="avg_ev",
                expression="AVG(launch_speed)",
                player_type="batter",
                min_pa=None,
                min_ip=None,
                created_at="2026-03-03",
            )
        )
        result = resolve_feature("avg_ev", statcast_conn, repo, [2023], "batter")
        assert len(result) > 0
        # Batter 100 should have avg(90, 95) or avg(90, 95, ...) depending on data
        batter_100 = [r for r in result if r.player_id == 100]
        assert len(batter_100) == 1
        assert batter_100[0].value is not None

    def test_falls_back_to_expression(self, statcast_conn: sqlite3.Connection, fbm_conn: sqlite3.Connection) -> None:
        repo = SqliteFeatureCandidateRepo(fbm_conn)
        result = resolve_feature("AVG(launch_speed)", statcast_conn, repo, [2023], "batter")
        assert len(result) > 0

    def test_named_candidate_uses_stored_min_pa(
        self, statcast_conn: sqlite3.Connection, fbm_conn: sqlite3.Connection
    ) -> None:
        repo = SqliteFeatureCandidateRepo(fbm_conn)
        repo.save(
            FeatureCandidate(
                name="strict_ev",
                expression="AVG(launch_speed)",
                player_type="batter",
                min_pa=100,
                min_ip=None,
                created_at="2026-03-03",
            )
        )
        result = resolve_feature("strict_ev", statcast_conn, repo, [2023], "batter")
        # min_pa=100 should filter out all players (they have only 2-3 pitches)
        assert len(result) == 0


class TestCandidateValuesToDict:
    def test_basic_conversion(self) -> None:
        values = [CandidateValue(1, 2023, 2.0), CandidateValue(2, 2023, 3.0)]
        result = candidate_values_to_dict(values)
        assert result == {(1, 2023): 2.0, (2, 2023): 3.0}

    def test_drops_nulls(self) -> None:
        values = [CandidateValue(1, 2023, 2.0), CandidateValue(2, 2023, None)]
        result = candidate_values_to_dict(values)
        assert result == {(1, 2023): 2.0}
        assert (2, 2023) not in result
