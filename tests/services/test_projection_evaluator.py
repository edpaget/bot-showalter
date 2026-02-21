import sqlite3

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.domain.evaluation import StratifiedComparisonResult
from fantasy_baseball_manager.domain.pt_normalization import ConsensusLookup
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator
from tests.helpers import seed_player


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
    war: float | None = None,
    pa: int | None = None,
) -> None:
    batting_repo.upsert(
        BattingStats(
            player_id=player_id,
            season=season,
            source=source,
            hr=hr,
            avg=avg,
            war=war,
            pa=pa,
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
    war: float | None = None,
    ip: float | None = None,
) -> None:
    pitching_repo.upsert(
        PitchingStats(
            player_id=player_id,
            season=season,
            source=source,
            era=era,
            so=so,
            war=war,
            ip=ip,
        )
    )


class TestEvaluateBattingSystem:
    def test_evaluate_batting_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
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
            seed_player(conn, player_id=pid)
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
            seed_player(conn, player_id=pid)
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
        seed_player(conn, player_id=1)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, season=2024)
        _seed_batter_projection(proj_repo, 1, hr=35, avg=0.290, season=2025)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, season=2025)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        assert result.metrics["hr"].n == 1


class TestEvaluateSkipsPlayersWithoutActuals:
    def test_evaluate_skips_players_without_actuals(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            seed_player(conn, player_id=pid)
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
            seed_player(conn, player_id=pid)
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
        seed_player(conn, player_id=1)
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
            seed_player(conn, player_id=pid)
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


class TestEvaluateActualsDriven:
    def test_unprojected_player_included_with_zero_for_counting(self, conn: sqlite3.Connection) -> None:
        """A player with actuals but no projection should appear with projected=0 for counting stats."""
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            seed_player(conn, player_id=pid)
        # Player 1 has projection and actuals
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        # Player 2 has only actuals — no projection
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        # counting stats: both players contribute
        assert result.metrics["hr"].n == 2
        # rate stats: missing player skipped (projected=0 is nonsensical for rates)
        assert result.metrics["avg"].n == 1

    def test_unprojected_pitcher_included_with_zero_for_counting(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, _, pitching_repo = _make_evaluator(conn)
        for pid in (10, 11):
            seed_player(conn, player_id=pid)
        _seed_pitcher_projection(proj_repo, 10, era=3.20, so=200)
        _seed_pitching_actuals(pitching_repo, 10, era=3.50, so=190)
        # Player 11 has only actuals
        _seed_pitching_actuals(pitching_repo, 11, era=4.00, so=160)

        result = evaluator.evaluate("steamer", "2025.1", 2025)
        # rate stats: missing player skipped
        assert result.metrics["era"].n == 1
        # counting stats: both players contribute
        assert result.metrics["so"].n == 2


class TestEvaluateTopNFilter:
    def test_top_filters_by_war(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, war=5.0)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310, war=3.0)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240, war=1.0)

        result = evaluator.evaluate("steamer", "2025.1", 2025, top=2)
        # Only top 2 by WAR should be included
        assert result.metrics["hr"].n == 2

    def test_top_none_includes_all(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, war=5.0)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310, war=3.0)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240, war=1.0)

        result = evaluator.evaluate("steamer", "2025.1", 2025, top=None)
        assert result.metrics["hr"].n == 3

    def test_top_filters_pitchers_by_war(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, _, pitching_repo = _make_evaluator(conn)
        for pid in (10, 11, 12):
            seed_player(conn, player_id=pid)
        _seed_pitcher_projection(proj_repo, 10, era=3.20, so=200)
        _seed_pitcher_projection(proj_repo, 11, era=4.00, so=150)
        _seed_pitcher_projection(proj_repo, 12, era=5.00, so=100)
        _seed_pitching_actuals(pitching_repo, 10, era=3.50, so=190, war=6.0)
        _seed_pitching_actuals(pitching_repo, 11, era=3.80, so=160, war=4.0)
        _seed_pitching_actuals(pitching_repo, 12, era=4.50, so=120, war=1.0)

        result = evaluator.evaluate("steamer", "2025.1", 2025, top=2)
        assert result.metrics["era"].n == 2
        assert result.metrics["so"].n == 2


class TestEvaluateMinPaIpFilter:
    def test_min_pa_filters_low_pa_batters(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, pa=500)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310, pa=100)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240, pa=30)

        result = evaluator.evaluate("steamer", "2025.1", 2025, min_pa=100)
        assert result.metrics["hr"].n == 2

    def test_min_ip_filters_low_ip_pitchers(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, _, pitching_repo = _make_evaluator(conn)
        for pid in (10, 11, 12):
            seed_player(conn, player_id=pid)
        _seed_pitcher_projection(proj_repo, 10, era=3.20, so=200)
        _seed_pitcher_projection(proj_repo, 11, era=4.00, so=150)
        _seed_pitcher_projection(proj_repo, 12, era=5.00, so=100)
        _seed_pitching_actuals(pitching_repo, 10, era=3.50, so=190, ip=180.0)
        _seed_pitching_actuals(pitching_repo, 11, era=3.80, so=160, ip=50.0)
        _seed_pitching_actuals(pitching_repo, 12, era=4.50, so=120, ip=5.0)

        result = evaluator.evaluate("steamer", "2025.1", 2025, min_ip=50)
        assert result.metrics["era"].n == 2

    def test_min_pa_none_includes_all(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, pa=500)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310, pa=100)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240, pa=30)

        result = evaluator.evaluate("steamer", "2025.1", 2025, min_pa=None)
        assert result.metrics["hr"].n == 3

    def test_min_pa_and_top_compose(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265, pa=500, war=5.0)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310, pa=100, war=3.0)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240, pa=30, war=1.0)

        # top=3 keeps all 3, then min_pa=100 filters out player 3 (pa=30)
        result = evaluator.evaluate("steamer", "2025.1", 2025, top=3, min_pa=100)
        assert result.metrics["hr"].n == 2


class TestCompareComposedSystems:
    def test_compare_includes_ensemble_system(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            seed_player(conn, player_id=pid)
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
            seed_player(conn, player_id=pid)
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
            seed_player(conn, player_id=pid)
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


class TestEvaluateStratified:
    def test_evaluate_stratified_splits_by_cohort(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batter_projection(proj_repo, 3, hr=15, avg=0.250)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)
        _seed_batting_actuals(batting_repo, 3, hr=18, avg=0.240)

        cohorts = {1: "young", 2: "young", 3: "veteran"}
        result = evaluator.evaluate_stratified("steamer", "2025.1", 2025, cohort_assignments=cohorts)

        assert "young" in result
        assert "veteran" in result
        assert result["young"].metrics["hr"].n == 2
        assert result["veteran"].metrics["hr"].n == 1

    def test_evaluate_stratified_excludes_unassigned(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        # Only player 1 is assigned
        cohorts = {1: "young"}
        result = evaluator.evaluate_stratified("steamer", "2025.1", 2025, cohort_assignments=cohorts)

        assert "young" in result
        assert result["young"].metrics["hr"].n == 1


class TestCompareStratified:
    def test_compare_stratified_returns_per_cohort(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1, 2):
            seed_player(conn, player_id=pid)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280, system="steamer", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=25, avg=0.300, system="steamer", version="2025.1")
        _seed_batter_projection(proj_repo, 1, hr=32, avg=0.275, system="zips", version="2025.1")
        _seed_batter_projection(proj_repo, 2, hr=22, avg=0.295, system="zips", version="2025.1")
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)
        _seed_batting_actuals(batting_repo, 2, hr=20, avg=0.310)

        cohorts = {1: "young", 2: "veteran"}
        result = evaluator.compare_stratified(
            [("steamer", "2025.1"), ("zips", "2025.1")],
            season=2025,
            cohort_assignments=cohorts,
            dimension="age",
        )

        assert isinstance(result, StratifiedComparisonResult)
        assert result.dimension == "age"
        assert result.season == 2025
        assert "young" in result.cohorts
        assert "veteran" in result.cohorts
        assert len(result.cohorts["young"].systems) == 2
        assert len(result.cohorts["veteran"].systems) == 2


class TestEvaluateWithNormalizePt:
    def test_evaluate_with_normalize_pt_rescales_counting(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        seed_player(conn, player_id=1)
        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30, "pa": 600, "avg": 0.280},
                source_type="third_party",
            )
        )
        _seed_batting_actuals(batting_repo, 1, hr=25, avg=0.280)

        lookup = ConsensusLookup(batting_pt={1: 500}, pitching_pt={})
        result = evaluator.evaluate("steamer", "2025.1", 2025, normalize_pt=lookup)
        # HR should be rescaled: 30 * (500/600) = 25.0 → error = 0
        assert result.metrics["hr"].mae == 0.0

    def test_evaluate_normalize_pt_none_is_noop(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        seed_player(conn, player_id=1)
        _seed_batter_projection(proj_repo, 1, hr=30, avg=0.280)
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)

        result_with = evaluator.evaluate("steamer", "2025.1", 2025, normalize_pt=None)
        result_without = evaluator.evaluate("steamer", "2025.1", 2025)
        assert result_with.metrics["hr"].mae == result_without.metrics["hr"].mae

    def test_evaluate_normalize_pt_skips_uncovered_player(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        seed_player(conn, player_id=1)
        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30, "pa": 600, "avg": 0.280},
                source_type="third_party",
            )
        )
        _seed_batting_actuals(batting_repo, 1, hr=28, avg=0.265)

        # Lookup doesn't include player 1
        lookup = ConsensusLookup(batting_pt={}, pitching_pt={})
        result = evaluator.evaluate("steamer", "2025.1", 2025, normalize_pt=lookup)
        # Should use original HR=30 → error=2
        assert result.metrics["hr"].mae == 2.0

    def test_compare_passes_normalize_pt_through(self, conn: sqlite3.Connection) -> None:
        evaluator, proj_repo, batting_repo, _ = _make_evaluator(conn)
        for pid in (1,):
            seed_player(conn, player_id=pid)
        for system in ("steamer", "zips"):
            proj_repo.upsert(
                Projection(
                    player_id=1,
                    season=2025,
                    system=system,
                    version="2025.1",
                    player_type="batter",
                    stat_json={"hr": 30, "pa": 600, "avg": 0.280},
                    source_type="third_party",
                )
            )
        _seed_batting_actuals(batting_repo, 1, hr=25, avg=0.280)

        lookup = ConsensusLookup(batting_pt={1: 500}, pitching_pt={})
        result = evaluator.compare(
            [("steamer", "2025.1"), ("zips", "2025.1")],
            season=2025,
            normalize_pt=lookup,
        )
        for sys_metrics in result.systems:
            assert sys_metrics.metrics["hr"].mae == 0.0
