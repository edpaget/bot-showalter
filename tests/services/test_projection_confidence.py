import sqlite3

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.projection_confidence import compute_confidence
from tests.helpers import seed_player


def _seed_projection(
    conn: sqlite3.Connection,
    player_id: int,
    system: str,
    stat_json: dict[str, float],
    *,
    season: int = 2026,
    version: str = "latest",
    player_type: str = "batter",
) -> None:
    repo = SqliteProjectionRepo(conn)
    repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type=player_type,
            stat_json=stat_json,
        )
    )


def _test_league() -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=(
            CategoryConfig(key="hr", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(
                key="obp",
                name="On-Base Percentage",
                stat_type=StatType.RATE,
                direction=Direction.HIGHER,
                numerator="h",
                denominator="ab",
            ),
        ),
        pitching_categories=(
            CategoryConfig(
                key="era",
                name="Earned Run Average",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            ),
            CategoryConfig(key="so", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _get_projections(conn: sqlite3.Connection, season: int = 2026) -> list[Projection]:
    repo = SqliteProjectionRepo(conn)
    return repo.get_by_season(season)


class TestHighAgreement:
    def test_close_projections_yield_high_agreement(self, conn: sqlite3.Connection) -> None:
        """Three systems with close HR/R values should yield high agreement."""
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Juan Soto"},
        )

        assert len(report.players) == 1
        player = report.players[0]
        assert player.agreement_level == "high"
        assert player.overall_cv < 0.10


class TestLowAgreement:
    def test_divergent_projections_yield_low_agreement(self, conn: sqlite3.Connection) -> None:
        """Three systems with wildly different HR/R values should yield low agreement."""
        pid = seed_player(conn, name_first="Wild", name_last="Card", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 10, "r": 40, "obp": 0.300})
        _seed_projection(conn, pid, "zips", {"hr": 35, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "marcel", {"hr": 45, "r": 120, "obp": 0.320})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Wild Card"},
        )

        assert len(report.players) == 1
        player = report.players[0]
        assert player.agreement_level == "low"
        assert player.overall_cv >= 0.25


class TestMinSystemsFilter:
    def test_player_below_min_systems_excluded(self, conn: sqlite3.Connection) -> None:
        """Player with only 2 systems should be excluded when min_systems=3."""
        pid_few = seed_player(conn, name_first="Few", name_last="Systems", mlbam_id=100001)
        _seed_projection(conn, pid_few, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_few, "zips", {"hr": 31, "r": 102, "obp": 0.390})

        pid_enough = seed_player(conn, name_first="Enough", name_last="Systems", mlbam_id=100002)
        _seed_projection(conn, pid_enough, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_enough, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid_enough, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid_few: "Few Systems", pid_enough: "Enough Systems"},
            min_systems=3,
        )

        assert len(report.players) == 1
        assert report.players[0].player_name == "Enough Systems"


class TestRateStatExcludedFromOverallCV:
    def test_only_counting_stats_in_overall_cv(self, conn: sqlite3.Connection) -> None:
        """Counting stats at CV=0 + wildly divergent OBP => overall_cv == 0.0."""
        pid = seed_player(conn, name_first="Rate", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.200})
        _seed_projection(conn, pid, "zips", {"hr": 30, "r": 100, "obp": 0.450})
        _seed_projection(conn, pid, "marcel", {"hr": 30, "r": 100, "obp": 0.350})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Rate Test"},
        )

        assert len(report.players) == 1
        assert report.players[0].overall_cv == 0.0


class TestSortOrder:
    def test_sorted_by_overall_cv_descending(self, conn: sqlite3.Connection) -> None:
        """Report should be sorted by overall_cv descending (most uncertain first)."""
        pid_safe = seed_player(conn, name_first="Safe", name_last="Player", mlbam_id=100001)
        _seed_projection(conn, pid_safe, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_safe, "zips", {"hr": 30, "r": 100, "obp": 0.390})
        _seed_projection(conn, pid_safe, "marcel", {"hr": 30, "r": 100, "obp": 0.410})

        pid_risky = seed_player(conn, name_first="Risky", name_last="Player", mlbam_id=100002)
        _seed_projection(conn, pid_risky, "steamer", {"hr": 10, "r": 40, "obp": 0.300})
        _seed_projection(conn, pid_risky, "zips", {"hr": 35, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_risky, "marcel", {"hr": 45, "r": 120, "obp": 0.320})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid_safe: "Safe Player", pid_risky: "Risky Player"},
        )

        assert len(report.players) == 2
        assert report.players[0].player_name == "Risky Player"
        assert report.players[1].player_name == "Safe Player"
        assert report.players[0].overall_cv >= report.players[1].overall_cv


class TestVersionDedup:
    def test_same_system_different_versions_deduped(self, conn: sqlite3.Connection) -> None:
        """Two versions of the same system should count as one system, not two."""
        pid = seed_player(conn, name_first="Dedup", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400}, version="v1")
        _seed_projection(conn, pid, "steamer", {"hr": 32, "r": 105, "obp": 0.410}, version="v2")
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.380})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Dedup Test"},
            min_systems=3,
        )

        assert len(report.players) == 1
        # Should have 3 systems (steamer, zips, marcel), not 4
        hr_spread = next(s for s in report.players[0].spreads if s.stat == "hr")
        assert len(hr_spread.systems) == 3


class TestPositions:
    def test_position_from_positions_dict(self, conn: sqlite3.Connection) -> None:
        """Position should come from the positions dict if provided."""
        pid = seed_player(conn, name_first="Pos", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Pos Test"},
            positions={pid: "SS"},
        )

        assert report.players[0].position == "SS"

    def test_position_defaults_to_empty(self, conn: sqlite3.Connection) -> None:
        """Position should default to empty string when no positions dict."""
        pid = seed_player(conn, name_first="No", name_last="Pos", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "No Pos"},
        )

        assert report.players[0].position == ""


class TestPitcherStats:
    def test_pitcher_uses_pitching_categories(self, conn: sqlite3.Connection) -> None:
        """Pitchers should use pitching categories (ERA, SO), not batting."""
        pid = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        _seed_projection(conn, pid, "steamer", {"era": 3.00, "so": 200, "ip": 180, "er": 60}, player_type="pitcher")
        _seed_projection(conn, pid, "zips", {"era": 3.20, "so": 190, "ip": 175, "er": 62}, player_type="pitcher")
        _seed_projection(conn, pid, "marcel", {"era": 3.50, "so": 180, "ip": 170, "er": 66}, player_type="pitcher")
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Gerrit Cole"},
        )

        assert len(report.players) == 1
        stat_names = {s.stat for s in report.players[0].spreads}
        assert "so" in stat_names
        assert "era" in stat_names
        assert "hr" not in stat_names
        assert "r" not in stat_names


class TestSystemsListInReport:
    def test_systems_sorted_and_deduped(self, conn: sqlite3.Connection) -> None:
        """report.systems should be sorted and deduplicated."""
        pid = seed_player(conn, name_first="Test", name_last="Player", mlbam_id=100001)
        _seed_projection(conn, pid, "zips", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "steamer", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        # Second version of steamer
        _seed_projection(conn, pid, "steamer", {"hr": 32, "r": 105, "obp": 0.420}, version="v2")
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Test Player"},
        )

        assert report.systems == ["marcel", "steamer", "zips"]


class TestEmptyInputs:
    def test_no_projections_yields_empty_report(self, conn: sqlite3.Connection) -> None:
        """No projections should yield an empty report."""
        report = compute_confidence(
            [],
            _test_league(),
            {},
        )

        assert report.players == []
        assert report.systems == []
