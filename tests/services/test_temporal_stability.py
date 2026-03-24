from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain.correlation_result import (
    CorrelationScanResult,
    PooledCorrelationResult,
    SeasonCorrelationResult,
    TargetCorrelation,
)
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.temporal_stability import TargetStability
from fantasy_baseball_manager.services.data_profiler import (
    CorrelationScanner,
    TemporalStabilityChecker,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator

_next_game_pk = 8000


@pytest.fixture
def statcast_conn() -> Generator[sqlite3.Connection]:
    global _next_game_pk  # noqa: PLW0603
    _next_game_pk = 8000
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
) -> None:
    global _next_game_pk  # noqa: PLW0603
    conn.execute(
        """INSERT INTO statcast_pitch
           (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, launch_speed)
           VALUES (?, ?, ?, ?, 1, 1, ?)""",
        (_next_game_pk, game_date, batter_id, pitcher_id, launch_speed),
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


def _make_scan_result(
    per_season_r: dict[int, float],
    target: str = "slg",
) -> CorrelationScanResult:
    """Build a minimal CorrelationScanResult with Pearson r values per season."""
    per_season = tuple(
        SeasonCorrelationResult(
            column_spec="launch_speed",
            season=season,
            player_type=PlayerType.BATTER,
            correlations=(
                TargetCorrelation(
                    target=target,
                    pearson_r=r,
                    pearson_p=0.01,
                    spearman_rho=r,
                    spearman_p=0.01,
                    n=50,
                ),
            ),
        )
        for season, r in sorted(per_season_r.items())
    )
    pooled = PooledCorrelationResult(
        column_spec="launch_speed",
        player_type=PlayerType.BATTER,
        correlations=(
            TargetCorrelation(
                target=target,
                pearson_r=sum(per_season_r.values()) / len(per_season_r),
                pearson_p=0.01,
                spearman_rho=0.5,
                spearman_p=0.01,
                n=50 * len(per_season_r),
            ),
        ),
    )
    return CorrelationScanResult(
        column_spec="launch_speed",
        player_type=PlayerType.BATTER,
        per_season=per_season,
        pooled=pooled,
    )


class TestComputeStabilityStable:
    """A feature with consistent correlations across seasons."""

    def test_stable_feature(self) -> None:
        scan = _make_scan_result({2021: 0.68, 2022: 0.72, 2023: 0.70, 2024: 0.69})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert ts.target == "slg"
        assert len(ts.per_season_r) == 4
        assert ts.classification == "stable"
        assert ts.cv < 0.3
        assert ts.cv >= 0.0


class TestComputeStabilityUnstable:
    """A one-year spike should be classified unstable."""

    def test_unstable_one_year_spike(self) -> None:
        scan = _make_scan_result({2021: 0.05, 2022: 0.60, 2023: 0.03, 2024: 0.02})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert ts.classification == "unstable"
        assert ts.cv > 0.6


class TestComputeStabilityModerate:
    """CV between 0.3 and 0.6 should be moderate."""

    def test_moderate_feature(self) -> None:
        # Mean ≈ 0.40, std ≈ 0.147 → CV ≈ 0.37
        scan = _make_scan_result({2021: 0.25, 2022: 0.55, 2023: 0.50, 2024: 0.30})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert ts.classification == "moderate"
        assert 0.3 <= ts.cv <= 0.6


class TestComputeStabilityNearZero:
    """Near-zero mean triggers sentinel CV and std-based classification."""

    def test_near_zero_mean_high_std(self) -> None:
        # r flips signs → |mean| < 0.05, std > 0.05 → unstable
        scan = _make_scan_result({2021: 0.10, 2022: -0.12, 2023: 0.08, 2024: -0.09})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert abs(ts.mean_r) < 0.05
        assert ts.classification == "unstable"

    def test_near_zero_mean_low_std(self) -> None:
        # Consistently near zero → stable
        scan = _make_scan_result({2021: 0.01, 2022: 0.02, 2023: 0.01, 2024: 0.02})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert abs(ts.mean_r) < 0.05
        assert ts.classification == "stable"

    def test_cv_sentinel(self) -> None:
        scan = _make_scan_result({2021: 0.01, 2022: -0.03, 2023: 0.02, 2024: -0.01})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert ts.cv == -1.0


class TestComputeStabilitySingleSeason:
    """Single season: std=0, classification=stable."""

    def test_single_season(self) -> None:
        scan = _make_scan_result({2023: 0.65})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert ts.std_r == 0.0
        assert ts.cv == 0.0
        assert ts.classification == "stable"
        assert ts.mean_r == 0.65


class TestComputeStabilityTwoSeasons:
    """Two seasons should produce a valid CV."""

    def test_two_seasons(self) -> None:
        scan = _make_scan_result({2022: 0.50, 2023: 0.70})
        ts = TemporalStabilityChecker._compute_stability(scan, "slg")

        assert isinstance(ts, TargetStability)
        assert len(ts.per_season_r) == 2
        assert ts.mean_r == pytest.approx(0.60)
        assert ts.std_r > 0.0


class TestCheckTemporalStabilitySingleTarget:
    """Integration test: full pipeline with DB fixtures for a single target."""

    def test_single_target(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        # Create data with consistent linear relationship across 2 seasons
        for season in [2022, 2023]:
            for i in range(1, 16):
                mlbam_id = 1000 + i
                _insert_pitch(
                    statcast_conn,
                    batter_id=mlbam_id,
                    pitcher_id=9999,
                    game_date=f"{season}-06-01",
                    launch_speed=80.0 + i * 1.0,
                )
                if season == 2022:
                    _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
                _insert_batting_stats(
                    stats_conn,
                    player_id=i,
                    season=season,
                    slg=0.300 + i * 0.01,
                    avg=0.220 + i * 0.005,
                    obp=0.280 + i * 0.005,
                    woba=0.280 + i * 0.008,
                )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(SingleConnectionProvider(statcast_conn), SingleConnectionProvider(stats_conn))
        checker = TemporalStabilityChecker(scanner)
        result = checker.check_temporal_stability("launch_speed", "slg", [2022, 2023], "batter")

        assert result.column_spec == "launch_speed"
        assert result.player_type == "batter"
        assert result.seasons == (2022, 2023)
        assert len(result.target_stabilities) == 1
        ts = result.target_stabilities[0]
        assert ts.target == "slg"
        # Should be stable since the relationship is consistent
        assert ts.classification == "stable"


class TestCheckTemporalStabilityAllTargets:
    """Integration test: target=None returns stability for all batter targets."""

    def test_all_targets(self, statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
        for season in [2022, 2023]:
            for i in range(1, 16):
                mlbam_id = 1000 + i
                _insert_pitch(
                    statcast_conn,
                    batter_id=mlbam_id,
                    pitcher_id=9999,
                    game_date=f"{season}-06-01",
                    launch_speed=80.0 + i * 1.0,
                )
                if season == 2022:
                    _insert_player(stats_conn, player_id=i, mlbam_id=mlbam_id)
                _insert_batting_stats(
                    stats_conn,
                    player_id=i,
                    season=season,
                    slg=0.300 + i * 0.01,
                    avg=0.220 + i * 0.005,
                    obp=0.280 + i * 0.005,
                    woba=0.280 + i * 0.008,
                    h=150 + i,
                    hr=10 + i,
                    ab=500,
                    so=100,
                    sf=5,
                )
        statcast_conn.commit()
        stats_conn.commit()

        scanner = CorrelationScanner(SingleConnectionProvider(statcast_conn), SingleConnectionProvider(stats_conn))
        checker = TemporalStabilityChecker(scanner)
        result = checker.check_temporal_stability("launch_speed", None, [2022, 2023], "batter")

        targets = {ts.target for ts in result.target_stabilities}
        assert targets == {"avg", "obp", "slg", "woba", "iso", "babip"}
