import sqlite3

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator


def _make_evaluator(
    conn: sqlite3.Connection,
) -> tuple[ProjectionEvaluator, SqliteProjectionRepo, SqliteBattingStatsRepo, SqlitePitchingStatsRepo]:
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    evaluator = ProjectionEvaluator(proj_repo, batting_repo, pitching_repo)
    return evaluator, proj_repo, batting_repo, pitching_repo


def _seed_batter_projection(
    proj_repo: SqliteProjectionRepo,
    player_id: int,
    hr: int,
    avg: float,
    system: str = "steamer",
    version: str = "2025.1",
    season: int = 2025,
    source_type: str = "third_party",
) -> None:
    proj_repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"hr": hr, "avg": avg},
            source_type=source_type,
        )
    )


def _seed_batting_actuals(
    batting_repo: SqliteBattingStatsRepo,
    player_id: int,
    hr: int,
    avg: float,
    season: int = 2025,
    source: str = "fangraphs",
) -> None:
    batting_repo.upsert(
        BattingStats(
            player_id=player_id,
            season=season,
            source=source,
            hr=hr,
            avg=avg,
        )
    )


def _seed_pitcher_projection(
    proj_repo: SqliteProjectionRepo,
    player_id: int,
    era: float,
    so: int,
    system: str = "steamer",
    version: str = "2025.1",
    season: int = 2025,
) -> None:
    proj_repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="pitcher",
            stat_json={"era": era, "so": so},
            source_type="third_party",
        )
    )


def _seed_pitching_actuals(
    pitching_repo: SqlitePitchingStatsRepo,
    player_id: int,
    era: float,
    so: int,
    season: int = 2025,
    source: str = "fangraphs",
) -> None:
    pitching_repo.upsert(
        PitchingStats(
            player_id=player_id,
            season=season,
            source=source,
            era=era,
            so=so,
        )
    )


def _seed_player(conn: sqlite3.Connection, player_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (?, 'Player', ?, '1990-01-01', 'R')",
        (player_id, str(player_id)),
    )
    conn.commit()


class TestEvaluateBattingSystem:
    def test_evaluate_batting_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert result.system == "steamer"
        assert result.version == "2025.1"
        assert "hr" in result.metrics
        assert "avg" in result.metrics
        assert result.metrics["hr"].n == 3
        assert result.metrics["avg"].n == 3


class TestEvaluatePitchingSystem:
    def test_evaluate_pitching_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, _, pitching_repo = _make_evaluator(conn)
        for pid in (10, 11):
            _seed_player(conn, pid)
        _seed_pitcher_projection(proj_repo, 10, era=3.20, so=200)
        _seed_pitcher_projection(proj_repo, 11, era=4.00, so=150)
        _seed_pitching_actuals(pitching_repo, 10, era=3.50, so=190)
        _seed_pitching_actuals(pitching_repo, 11, era=3.80, so=160)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert "era" in result.metrics
        assert "so" in result.metrics
        assert result.metrics["era"].n == 2
        assert result.metrics["so"].n == 2


class TestEvaluateMixedPlayerTypes:
    def test_evaluate_mixed_player_types(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, pitching_repo = _make_evaluator(conn)
        for pid in (1, 10):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_pitcher_projection(proj_repo, 10, era=3.20, so=200)
        _seed_pitching_actuals(pitching_repo, 10, era=3.50, so=190)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert "hr" in result.metrics
        assert "era" in result.metrics


class TestEvaluateFiltersBySeason:
    def test_evaluate_filters_by_season(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        _seed_player(conn, 1)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, season=2024)
        _seed_batter_projection(proj_repo, 1, hr=35, avg=0.290, season=2025)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, season=2025)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert result.metrics["hr"].n == 1


class TestEvaluateSkipsPlayersWithoutActuals:
    def test_evaluate_skips_players_without_actuals(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        # Player 2 has no actuals

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert result.metrics["hr"].n == 1


class TestEvaluateWithStatFilter:
    def test_evaluate_with_stat_filter(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.evaluate("steamer", "2025.1", 2025, stats=["hr"])
        assert "hr" in result.metrics
        assert "avg" not in result.metrics


class TestEvaluateSourceType:
    def test_source_type_from_projections(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        _seed_player(conn, 1)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, source_type="third_party")
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)

        result = evaluator.evaluate("steamer", "2025.1", 2025)

        assert result.source_type == "third_party"

    def test_source_type_default_when_no_projections(self, conn: sqlite3.Connection) -> None:
        evaluator, _, _, _ = _make_evaluator(conn)

        result = evaluator.evaluate("steamer", "2025.1", 2025)

        assert result.source_type == "first_party"


class TestCompareMultipleSystems:
    def test_compare_multiple_systems(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, system="steamer", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300, system="steamer", version="2025.1")
        _seed_batter_projection(proj_repo, 1, hr=32, avg=0.275, system="zips", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=22, avg=0.295, system="zips", version="2025.1")
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.compare(
            [("steamer", "2025.1"), ("zips", "2025.1")],
            season=2025,
        )
        assert result.season == 2025
        assert len(result.systems) == 2
        system_names = [s.system for s in result.systems]
        assert "steamer" in system_names
        assert "zips" in system_names


class TestCompareComposedSystems:
    def test_compare_includes_ensemble_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=29, avg=0.270, system="ensemble", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=23, avg=0.295, system="ensemble", version="2025.1")
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.compare(
            [("ensemble", "2025.1")],
            season=2025,
        )
        assert len(result.systems) == 1
        assert result.systems[0].system == "ensemble"
        assert "hr" in result.systems[0].metrics
        assert result.systems[0].metrics["hr"].n == 2

    def test_compare_includes_composite_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, system="composite", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=24, avg=0.290, system="composite", version="2025.1")
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.compare(
            [("composite", "2025.1")],
            season=2025,
        )
        assert len(result.systems) == 1
        assert result.systems[0].system == "composite"
        assert "hr" in result.systems[0].metrics

    def test_compare_ensemble_alongside_components(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)
        # Component systems
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, system="marcel", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300, system="marcel", version="2025.1")
        _seed_batter_projection(proj_repo, 1, hr=28, avg=0.270, system="steamer", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=22, avg=0.290, system="steamer", version="2025.1")
        # Ensemble system
        _seed_batter_projection(proj_repo, 1, hr=29, avg=0.275, system="ensemble", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=24, avg=0.295, system="ensemble", version="2025.1")
        # Actuals
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.compare(
            [("marcel", "2025.1"), ("steamer", "2025.1"), ("ensemble", "2025.1")],
            season=2025,
        )
        assert len(result.systems) == 3
        system_names = [s.system for s in result.systems]
        assert "marcel" in system_names
        assert "steamer" in system_names
        assert "ensemble" in system_names
