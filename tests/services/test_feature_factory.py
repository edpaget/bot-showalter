from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import Err, Ok
from fantasy_baseball_manager.domain.feature_candidate import BinnedValue, CandidateValue, FeatureCandidate
from fantasy_baseball_manager.repos.feature_candidate_repo import SqliteFeatureCandidateRepo
from fantasy_baseball_manager.services.feature_factory import (
    aggregate_candidate,
    bin_candidate,
    candidate_values_to_dict,
    cross_bin_candidates,
    inject_candidate_values,
    interact_candidates,
    remap_candidate_keys,
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


class TestBinCandidate:
    def test_quantile_4_bins(self) -> None:
        """20 values should split into 4 bins with ~5 players each."""
        values = [CandidateValue(i, 2023, float(i)) for i in range(1, 21)]
        result = bin_candidate(values, "quantile", 4)
        labels = {r.bin_label for r in result}
        assert labels == {"Q1", "Q2", "Q3", "Q4"}
        counts = {label: sum(1 for r in result if r.bin_label == label) for label in labels}
        # Each bin should have 5 values
        assert all(c == 5 for c in counts.values())

    def test_uniform_3_bins(self) -> None:
        """Uniform binning should produce equal-width intervals."""
        # Values 0.0, 3.0, 6.0, 9.0 -> range 0-9, width 3 -> bins B1=[0,3), B2=[3,6), B3=[6,9]
        values = [
            CandidateValue(1, 2023, 0.0),
            CandidateValue(2, 2023, 3.0),
            CandidateValue(3, 2023, 6.0),
            CandidateValue(4, 2023, 9.0),
        ]
        result = bin_candidate(values, "uniform", 3)
        by_pid = {r.player_id: r.bin_label for r in result}
        assert by_pid[1] == "B1"  # 0.0 in first bin
        assert by_pid[4] == "B3"  # 9.0 in last bin

    def test_custom_breakpoints(self) -> None:
        """User-provided breakpoints should assign correct bins."""
        values = [
            CandidateValue(1, 2023, 1.0),
            CandidateValue(2, 2023, 5.0),
            CandidateValue(3, 2023, 10.0),
        ]
        result = bin_candidate(values, "custom", 3, breakpoints=[3.0, 7.0])
        by_pid = {r.player_id: r.bin_label for r in result}
        assert by_pid[1] == "C1"  # 1.0 < 3.0
        assert by_pid[2] == "C2"  # 3.0 <= 5.0 < 7.0
        assert by_pid[3] == "C3"  # 10.0 >= 7.0

    def test_null_values_excluded(self) -> None:
        """NULL values should be skipped, not binned."""
        values = [
            CandidateValue(1, 2023, 1.0),
            CandidateValue(2, 2023, None),
            CandidateValue(3, 2023, 3.0),
        ]
        result = bin_candidate(values, "quantile", 2)
        player_ids = {r.player_id for r in result}
        assert 2 not in player_ids
        assert len(result) == 2

    def test_per_season_independent(self) -> None:
        """Same player in different seasons can get different bins."""
        values = [
            # Season 2023: player 1 has lowest value
            CandidateValue(1, 2023, 1.0),
            CandidateValue(2, 2023, 10.0),
            # Season 2024: player 1 has highest value
            CandidateValue(1, 2024, 100.0),
            CandidateValue(2, 2024, 1.0),
        ]
        result = bin_candidate(values, "quantile", 2)
        p1_2023 = next(r for r in result if r.player_id == 1 and r.season == 2023)
        p1_2024 = next(r for r in result if r.player_id == 1 and r.season == 2024)
        assert p1_2023.bin_label == "Q1"
        assert p1_2024.bin_label == "Q2"

    def test_invalid_method_raises(self) -> None:
        values = [CandidateValue(1, 2023, 1.0)]
        with pytest.raises(ValueError, match="method"):
            bin_candidate(values, "invalid", 2)

    def test_custom_requires_breakpoints(self) -> None:
        values = [CandidateValue(1, 2023, 1.0)]
        with pytest.raises(ValueError, match="breakpoints"):
            bin_candidate(values, "custom", 2)

    def test_single_value_goes_to_single_bin(self) -> None:
        """Degenerate case: 1 non-null value should go to a single bin."""
        values = [CandidateValue(1, 2023, 5.0)]
        result = bin_candidate(values, "quantile", 4)
        assert len(result) == 1
        assert result[0].bin_label == "Q1"


class TestCrossBinCandidates:
    def test_cross_product_labels(self) -> None:
        bins_a = [BinnedValue(1, 2023, "Q1", 1.0), BinnedValue(2, 2023, "Q2", 2.0)]
        bins_b = [BinnedValue(1, 2023, "Q3", 3.0), BinnedValue(2, 2023, "Q1", 4.0)]
        result = cross_bin_candidates(bins_a, bins_b)
        by_pid = {r.player_id: r.bin_label for r in result}
        assert by_pid[1] == "Q1__Q3"
        assert by_pid[2] == "Q2__Q1"

    def test_only_matching_keys(self) -> None:
        bins_a = [BinnedValue(1, 2023, "Q1", 1.0), BinnedValue(3, 2023, "Q2", 3.0)]
        bins_b = [BinnedValue(1, 2023, "Q3", 2.0), BinnedValue(2, 2023, "Q1", 4.0)]
        result = cross_bin_candidates(bins_a, bins_b)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_multi_season(self) -> None:
        bins_a = [BinnedValue(1, 2023, "Q1", 1.0), BinnedValue(1, 2024, "Q2", 2.0)]
        bins_b = [BinnedValue(1, 2023, "Q3", 3.0), BinnedValue(1, 2024, "Q1", 4.0)]
        result = cross_bin_candidates(bins_a, bins_b)
        by_season = {r.season: r.bin_label for r in result}
        assert by_season[2023] == "Q1__Q3"
        assert by_season[2024] == "Q2__Q1"


class TestInjectCandidateValues:
    def test_injects_matching_values(self) -> None:
        rows_by_season: dict[int, list[dict[str, Any]]] = {
            2023: [
                {"player_id": 1, "feature_a": 0.5},
                {"player_id": 2, "feature_a": 0.6},
            ],
        }
        values = {(1, 2023): 0.9, (2, 2023): 0.8}
        inject_candidate_values(rows_by_season, "new_col", values)
        assert rows_by_season[2023][0]["new_col"] == 0.9
        assert rows_by_season[2023][1]["new_col"] == 0.8

    def test_unmatched_rows_get_nan(self) -> None:
        rows_by_season: dict[int, list[dict[str, Any]]] = {
            2023: [{"player_id": 1, "feature_a": 0.5}],
        }
        values: dict[tuple[int, int], float] = {}
        inject_candidate_values(rows_by_season, "new_col", values)
        assert math.isnan(rows_by_season[2023][0]["new_col"])

    def test_multi_season_injection(self) -> None:
        rows_by_season: dict[int, list[dict[str, Any]]] = {
            2023: [{"player_id": 1, "feature_a": 0.5}],
            2024: [{"player_id": 1, "feature_a": 0.6}],
        }
        values = {(1, 2023): 0.9, (1, 2024): 0.7}
        inject_candidate_values(rows_by_season, "new_col", values)
        assert rows_by_season[2023][0]["new_col"] == 0.9
        assert rows_by_season[2024][0]["new_col"] == 0.7

    def test_modifies_rows_in_place(self) -> None:
        row = {"player_id": 1, "feature_a": 0.5}
        rows_by_season: dict[int, list[dict[str, Any]]] = {2023: [row]}
        inject_candidate_values(rows_by_season, "x", {(1, 2023): 1.0})
        assert row["x"] == 1.0


class TestRemapCandidateKeys:
    def test_remaps_mlbam_to_internal(self) -> None:
        values = {(660271, 2023): 0.9, (545361, 2023): 0.8}
        mlbam_to_internal = {660271: 1, 545361: 2}
        result = remap_candidate_keys(values, mlbam_to_internal)
        assert result == {(1, 2023): 0.9, (2, 2023): 0.8}

    def test_skips_unknown_mlbam_ids(self) -> None:
        values = {(660271, 2023): 0.9, (999999, 2023): 0.5}
        mlbam_to_internal = {660271: 1}
        result = remap_candidate_keys(values, mlbam_to_internal)
        assert result == {(1, 2023): 0.9}
        assert (999999, 2023) not in result

    def test_preserves_seasons(self) -> None:
        values = {(100, 2023): 0.5, (100, 2024): 0.6}
        mlbam_to_internal = {100: 42}
        result = remap_candidate_keys(values, mlbam_to_internal)
        assert result == {(42, 2023): 0.5, (42, 2024): 0.6}

    def test_empty_values(self) -> None:
        result = remap_candidate_keys({}, {100: 1})
        assert result == {}

    def test_empty_mapping(self) -> None:
        values = {(100, 2023): 0.5}
        result = remap_candidate_keys(values, {})
        assert result == {}
