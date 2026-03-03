from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain.correlation_result import (
    CorrelationScanResult,
    MultiColumnRanking,
)
from fantasy_baseball_manager.domain.feature_candidate import BinnedValue
from fantasy_baseball_manager.services.data_profiler import CorrelationScanner, rank_columns

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator

_next_game_pk = 5000


@pytest.fixture
def statcast_conn() -> Generator[sqlite3.Connection]:
    global _next_game_pk  # noqa: PLW0603
    _next_game_pk = 5000
    connection = create_statcast_connection(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def stats_conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


def _insert_pitch(
    conn: sqlite3.Connection,
    *,
    batter_id: int,
    pitcher_id: int,
    game_date: str,
    launch_speed: float | None = None,
    release_speed: float | None = None,
    barrel: int | None = None,
) -> None:
    global _next_game_pk  # noqa: PLW0603
    conn.execute(
        """INSERT INTO statcast_pitch
           (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
            launch_speed, release_speed, barrel)
           VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)""",
        (_next_game_pk, game_date, batter_id, pitcher_id, launch_speed, release_speed, barrel),
    )
    _next_game_pk += 1


def _insert_player(conn: sqlite3.Connection, *, player_id: int, mlbam_id: int) -> None:
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, mlbam_id) VALUES (?, 'Test', 'Player', ?)",
        (player_id, mlbam_id),
    )


def _insert_batting_stats(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    season: int,
    avg: float = 0.250,
    obp: float = 0.330,
    slg: float = 0.400,
    woba: float = 0.320,
    h: int = 150,
    hr: int = 20,
    ab: int = 500,
    so: int = 100,
    sf: int = 5,
) -> None:
    conn.execute(
        """INSERT INTO batting_stats
           (player_id, season, source, avg, obp, slg, woba, h, hr, ab, so, sf)
           VALUES (?, ?, 'fangraphs', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (player_id, season, avg, obp, slg, woba, h, hr, ab, so, sf),
    )


def _insert_pitching_stats(
    conn: sqlite3.Connection,
    *,
    player_id: int,
    season: int,
    era: float = 3.50,
    fip: float = 3.60,
    k_per_9: float = 9.0,
    bb_per_9: float = 3.0,
    whip: float = 1.20,
    hr: int = 15,
    ip: float = 180.0,
    h: int = 160,
    so: int = 180,
) -> None:
    conn.execute(
        """INSERT INTO pitching_stats
           (player_id, season, source, era, fip, k_per_9, bb_per_9, whip, hr, ip, h, so)
           VALUES (?, ?, 'fangraphs', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (player_id, season, era, fip, k_per_9, bb_per_9, whip, hr, ip, h, so),
    )


def _setup_batter_scenario(
    statcast_conn: sqlite3.Connection,
    stats_conn: sqlite3.Connection,
    *,
    n_players: int = 20,
    season: int = 2023,
) -> None:
    """Create n batters with linear relationship: launch_speed -> slg."""
    for i in range(1, n_players + 1):
        mlbam_id = 1000 + i
        player_id = i
        # launch_speed linearly increasing
        ls = 80.0 + i * 1.0  # 81, 82, ..., 100
        # slg linearly correlated with launch_speed
        slg = 0.300 + i * 0.01  # 0.31, 0.32, ..., 0.50
        avg = 0.220 + i * 0.005
        obp = avg + 0.060
        woba = 0.280 + i * 0.008

        _insert_pitch(
            statcast_conn,
            batter_id=mlbam_id,
            pitcher_id=9999,
            game_date=f"{season}-06-01",
            launch_speed=ls,
        )
        _insert_player(stats_conn, player_id=player_id, mlbam_id=mlbam_id)
        _insert_batting_stats(
            stats_conn,
            player_id=player_id,
            season=season,
            avg=avg,
            obp=obp,
            slg=slg,
            woba=woba,
            h=150 + i,
            hr=10 + i,
            ab=500,
            so=100,
            sf=5,
        )
    statcast_conn.commit()
    stats_conn.commit()


class TestCorrelationScannerBatter:
    """Test batter correlation scanning with known linear relationships."""

    def test_strong_positive_correlation(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        _setup_batter_scenario(statcast_conn, stats_conn, n_players=20, season=2023)
        scanner = CorrelationScanner(statcast_conn, stats_conn)

        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        assert isinstance(result, CorrelationScanResult)
        assert result.column_spec == "launch_speed"
        assert result.player_type == "batter"

        # Check pooled correlations exist for all batter targets
        target_names = {c.target for c in result.pooled.correlations}
        assert "slg" in target_names
        assert "avg" in target_names
        assert "iso" in target_names
        assert "babip" in target_names

        # slg should have strong positive correlation with launch_speed
        slg_corr = next(c for c in result.pooled.correlations if c.target == "slg")
        assert slg_corr.pearson_r > 0.95
        assert slg_corr.pearson_p < 0.05
        assert slg_corr.spearman_rho > 0.95
        assert slg_corr.n == 20

    def test_per_season_results(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        _setup_batter_scenario(statcast_conn, stats_conn, n_players=20, season=2023)
        # Add a second season
        for i in range(1, 11):
            mlbam_id = 1000 + i
            ls = 80.0 + i * 1.0
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2024-06-01",
                launch_speed=ls,
            )
            _insert_batting_stats(
                stats_conn,
                player_id=i,
                season=2024,
                slg=0.300 + i * 0.01,
                avg=0.220 + i * 0.005,
                obp=0.280 + i * 0.005,
                woba=0.280 + i * 0.008,
            )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023, 2024], "batter")

        assert len(result.per_season) == 2
        seasons = {s.season for s in result.per_season}
        assert seasons == {2023, 2024}

        # Per-season should have strong correlations too
        s2023 = next(s for s in result.per_season if s.season == 2023)
        slg_2023 = next(c for c in s2023.correlations if c.target == "slg")
        assert slg_2023.pearson_r > 0.95
        assert slg_2023.n == 20

        s2024 = next(s for s in result.per_season if s.season == 2024)
        slg_2024 = next(c for c in s2024.correlations if c.target == "slg")
        assert slg_2024.n == 10

    def test_no_correlation(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """When launch_speed is constant, correlation should be ~0."""
        for i in range(1, 21):
            mlbam_id = 1000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=90.0,  # constant
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(
                stats_conn,
                player_id=i,
                season=2023,
                slg=0.300 + i * 0.01,
            )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        slg_corr = next(c for c in result.pooled.correlations if c.target == "slg")
        # Constant input -> 0 correlation
        assert slg_corr.pearson_r == 0.0
        assert slg_corr.spearman_rho == 0.0

    def test_derived_target_iso(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """ISO = slg - avg should be computed and correlated."""
        _setup_batter_scenario(statcast_conn, stats_conn, n_players=15, season=2023)
        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        iso_corr = next(c for c in result.pooled.correlations if c.target == "iso")
        assert iso_corr.n == 15
        # ISO = slg - avg, both linearly correlated -> iso also correlated
        assert abs(iso_corr.pearson_r) > 0.5

    def test_derived_target_babip(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """BABIP should be computed from component stats."""
        _setup_batter_scenario(statcast_conn, stats_conn, n_players=15, season=2023)
        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        babip_corr = next(c for c in result.pooled.correlations if c.target == "babip")
        assert babip_corr.n == 15


class TestCorrelationScannerPitcher:
    """Test pitcher correlation scanning."""

    def test_pitcher_targets(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        for i in range(1, 16):
            mlbam_id = 2000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=9999,
                pitcher_id=mlbam_id,
                game_date="2023-06-01",
                release_speed=85.0 + i * 0.5,
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_pitching_stats(
                stats_conn,
                player_id=i,
                season=2023,
                era=4.50 - i * 0.1,
                fip=4.50 - i * 0.1,
                k_per_9=7.0 + i * 0.2,
                bb_per_9=4.0 - i * 0.1,
                whip=1.50 - i * 0.02,
                hr=20 - i,
                ip=150.0 + i * 2.0,
                h=160 - i * 2,
                so=150 + i * 3,
            )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("release_speed", [2023], "pitcher")

        target_names = {c.target for c in result.pooled.correlations}
        assert "era" in target_names
        assert "fip" in target_names
        assert "k_per_9" in target_names
        assert "bb_per_9" in target_names
        assert "hr_per_9" in target_names
        assert "babip" in target_names
        assert "whip" in target_names

    def test_pitcher_derived_hr_per_9(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        for i in range(1, 16):
            mlbam_id = 2000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=9999,
                pitcher_id=mlbam_id,
                game_date="2023-06-01",
                release_speed=85.0 + i * 0.5,
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_pitching_stats(
                stats_conn,
                player_id=i,
                season=2023,
                hr=20 - i,
                ip=150.0 + i * 2.0,
                h=160,
                so=150,
            )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("release_speed", [2023], "pitcher")

        hr9_corr = next(c for c in result.pooled.correlations if c.target == "hr_per_9")
        assert hr9_corr.n == 15


class TestCorrelationScannerExpressions:
    """Test SQL aggregation expression column specs."""

    def test_expression_with_where_clause(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        """Expression like 'AVG(launch_speed) WHERE barrel = 1' should work."""
        for i in range(1, 16):
            mlbam_id = 1000 + i
            # Insert barrel = 1 pitches with varying launch_speed
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=95.0 + i * 0.5,
                barrel=1,
            )
            # Insert barrel = 0 pitches (should be filtered out)
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=70.0,
                barrel=0,
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(
                stats_conn,
                player_id=i,
                season=2023,
                slg=0.300 + i * 0.01,
            )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("AVG(launch_speed) WHERE barrel = 1", [2023], "batter")

        slg_corr = next(c for c in result.pooled.correlations if c.target == "slg")
        assert slg_corr.n == 15
        assert slg_corr.pearson_r > 0.95


class TestCorrelationScannerEdgeCases:
    """Test edge cases."""

    def test_insufficient_data_returns_zero_correlation(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        """With < 3 matched players, correlations should be 0."""
        # Only 2 players
        for i in range(1, 3):
            mlbam_id = 1000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=90.0 + i,
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(stats_conn, player_id=i, season=2023)
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        for corr in result.pooled.correlations:
            assert corr.pearson_r == 0.0
            assert corr.pearson_p == 1.0
            assert corr.n <= 2

    def test_no_matching_players_returns_empty_correlations(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        """No overlap between statcast and stats -> 0 correlations."""
        # Statcast has mlbam_id=1001, stats has player_id=1 with different mlbam_id=9999
        _insert_pitch(
            statcast_conn,
            batter_id=1001,
            pitcher_id=9999,
            game_date="2023-06-01",
            launch_speed=90.0,
        )
        statcast_conn.commit()
        _insert_player(stats_conn, player_id=1, mlbam_id=9999)
        _insert_batting_stats(stats_conn, player_id=1, season=2023)
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2023], "batter")

        for corr in result.pooled.correlations:
            assert corr.n == 0
            assert corr.pearson_r == 0.0


class TestScanMultiple:
    """Test scanning multiple columns."""

    def test_returns_results_per_column(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        for i in range(1, 16):
            mlbam_id = 1000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=80.0 + i,
                release_speed=85.0 + i * 0.5,
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(stats_conn, player_id=i, season=2023, slg=0.300 + i * 0.01)
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        results = scanner.scan_multiple(["launch_speed", "release_speed"], [2023], "batter")

        assert len(results) == 2
        specs = {r.column_spec for r in results}
        assert specs == {"launch_speed", "release_speed"}


class TestRankColumns:
    """Test the rank_columns function."""

    def test_ranks_by_avg_abs_pearson(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        # launch_speed strongly correlated with slg, release_speed weakly
        for i in range(1, 21):
            mlbam_id = 1000 + i
            _insert_pitch(
                statcast_conn,
                batter_id=mlbam_id,
                pitcher_id=9999,
                game_date="2023-06-01",
                launch_speed=80.0 + i,  # linear relationship
                release_speed=90.0,  # constant -> no correlation
            )
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(stats_conn, player_id=i, season=2023, slg=0.300 + i * 0.01)
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        results = scanner.scan_multiple(["launch_speed", "release_speed"], [2023], "batter")
        rankings = rank_columns(results)

        assert len(rankings) == 2
        assert all(isinstance(r, MultiColumnRanking) for r in rankings)
        # launch_speed should rank first (higher avg correlation)
        assert rankings[0].column_spec == "launch_speed"
        assert rankings[0].avg_abs_pearson > rankings[1].avg_abs_pearson


class TestPooledAcrossSeasons:
    """Test that pooled correlations combine data across seasons."""

    def test_pooled_n_combines_seasons(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        for i in range(1, 11):
            mlbam_id = 1000 + i
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            for season in [2022, 2023]:
                _insert_pitch(
                    statcast_conn,
                    batter_id=mlbam_id,
                    pitcher_id=9999,
                    game_date=f"{season}-06-01",
                    launch_speed=80.0 + i,
                )
                _insert_batting_stats(
                    stats_conn,
                    player_id=i,
                    season=season,
                    slg=0.300 + i * 0.01,
                )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_target_correlations("launch_speed", [2022, 2023], "batter")

        slg_corr = next(c for c in result.pooled.correlations if c.target == "slg")
        # 10 players * 2 seasons = 20 observations pooled
        assert slg_corr.n == 20


class TestScanFromValues:
    """Test scan_from_values with pre-computed candidate values."""

    def test_returns_correlation_scan_result(
        self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection
    ) -> None:
        """Pre-computed values with known linear relationship to slg should show high correlation."""
        # Set up players and batting stats
        for i in range(1, 21):
            mlbam_id = 1000 + i
            player_id = i
            slg = 0.300 + i * 0.01
            _insert_player(stats_conn, player_id=player_id, mlbam_id=mlbam_id)
            _insert_batting_stats(stats_conn, player_id=player_id, season=2023, slg=slg)
        stats_conn.commit()

        # Pre-computed candidate values: mlbam_id keyed, linearly related to slg
        candidate_values: dict[tuple[int, int], float] = {}
        for i in range(1, 21):
            mlbam_id = 1000 + i
            candidate_values[(mlbam_id, 2023)] = 80.0 + i * 1.0

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_from_values("custom_feature", candidate_values, [2023], "batter")

        assert isinstance(result, CorrelationScanResult)
        assert result.column_spec == "custom_feature"
        assert result.player_type == "batter"

        slg_corr = next(c for c in result.pooled.correlations if c.target == "slg")
        assert slg_corr.pearson_r > 0.95
        assert slg_corr.n == 20

    def test_per_season_results(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """Multi-season pre-computed values should have per_season and pooled."""
        for i in range(1, 11):
            mlbam_id = 1000 + i
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            for season in [2022, 2023]:
                _insert_batting_stats(stats_conn, player_id=i, season=season, slg=0.300 + i * 0.01)
        stats_conn.commit()

        candidate_values: dict[tuple[int, int], float] = {}
        for i in range(1, 11):
            mlbam_id = 1000 + i
            for season in [2022, 2023]:
                candidate_values[(mlbam_id, season)] = 80.0 + i

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.scan_from_values("custom", candidate_values, [2022, 2023], "batter")

        assert len(result.per_season) == 2
        seasons = {s.season for s in result.per_season}
        assert seasons == {2022, 2023}

        slg_pooled = next(c for c in result.pooled.correlations if c.target == "slg")
        assert slg_pooled.n == 20  # 10 players * 2 seasons


class TestComputeBinTargetMeans:
    """Test compute_bin_target_means with binned values."""

    def test_returns_per_bin_means(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """2 bins with known target values should return correct per-bin means."""
        # Set up: 4 players, 2 in each bin
        for i in range(1, 5):
            mlbam_id = 1000 + i
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            # Players 1-2: low slg (0.300, 0.310)
            # Players 3-4: high slg (0.400, 0.410)
            slg = 0.300 + (i - 1) * 0.010 if i <= 2 else 0.400 + (i - 3) * 0.010
            _insert_batting_stats(stats_conn, player_id=i, season=2023, slg=slg)
        stats_conn.commit()

        # Bin assignment: players 1-2 in Q1, players 3-4 in Q2
        binned = [
            BinnedValue(player_id=1001, season=2023, bin_label="Q1", value=80.0),
            BinnedValue(player_id=1002, season=2023, bin_label="Q1", value=82.0),
            BinnedValue(player_id=1003, season=2023, bin_label="Q2", value=95.0),
            BinnedValue(player_id=1004, season=2023, bin_label="Q2", value=97.0),
        ]

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.compute_bin_target_means(binned, [2023], "batter")

        # Find slg means per bin
        q1_slg = next(r for r in result if r.bin_label == "Q1" and r.target == "slg")
        q2_slg = next(r for r in result if r.bin_label == "Q2" and r.target == "slg")

        assert q1_slg.mean == pytest.approx((0.300 + 0.310) / 2)
        assert q1_slg.count == 2
        assert q2_slg.mean == pytest.approx((0.400 + 0.410) / 2)
        assert q2_slg.count == 2

    def test_multiple_targets(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        """All batter targets should be reported per bin."""
        for i in range(1, 5):
            mlbam_id = 1000 + i
            _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
            _insert_batting_stats(
                stats_conn,
                player_id=i,
                season=2023,
                avg=0.250 + i * 0.010,
                obp=0.330 + i * 0.010,
                slg=0.400 + i * 0.010,
                woba=0.320 + i * 0.010,
            )
        stats_conn.commit()

        binned = [
            BinnedValue(player_id=1001, season=2023, bin_label="Q1", value=80.0),
            BinnedValue(player_id=1002, season=2023, bin_label="Q1", value=82.0),
            BinnedValue(player_id=1003, season=2023, bin_label="Q2", value=95.0),
            BinnedValue(player_id=1004, season=2023, bin_label="Q2", value=97.0),
        ]

        scanner = CorrelationScanner(statcast_conn, stats_conn)
        result = scanner.compute_bin_target_means(binned, [2023], "batter")

        # Should have results for all batter targets per bin
        targets_reported = {r.target for r in result}
        assert "avg" in targets_reported
        assert "obp" in targets_reported
        assert "slg" in targets_reported
        assert "woba" in targets_reported
        assert "iso" in targets_reported
        assert "babip" in targets_reported

        # Check that each bin has all targets
        q1_targets = {r.target for r in result if r.bin_label == "Q1"}
        q2_targets = {r.target for r in result if r.bin_label == "Q2"}
        assert q1_targets == q2_targets
