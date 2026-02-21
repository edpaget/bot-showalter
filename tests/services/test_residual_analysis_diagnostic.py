import sqlite3


from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.residual_analysis_diagnostic import ResidualAnalysisDiagnostic
from tests.helpers import seed_player


def _make_service(
    conn: sqlite3.Connection,
) -> tuple[ResidualAnalysisDiagnostic, SqliteProjectionRepo, SqliteBattingStatsRepo, SqlitePitchingStatsRepo]:
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    player_repo = SqlitePlayerRepo(conn)
    service = ResidualAnalysisDiagnostic(proj_repo, batting_repo, pitching_repo, player_repo)
    return service, proj_repo, batting_repo, pitching_repo


def _seed_batter_data(
    proj_repo: SqliteProjectionRepo,
    batting_repo: SqliteBattingStatsRepo,
    conn: sqlite3.Connection,
    *,
    n_players: int = 10,
    seasons: tuple[int, ...] = (2024,),
    bias: float = 0.0,
) -> None:
    """Seed batter projections and actuals with controllable bias."""
    for pid in range(1, n_players + 1):
        seed_player(conn, player_id=pid, name_last=f"Batter{pid}")
        for season in seasons:
            est_avg = 0.250 + pid * 0.005
            actual_avg = est_avg + bias + (pid % 3 - 1) * 0.005  # some variation
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=season,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": est_avg, "obp": est_avg + 0.050, "war": 2.0 + pid * 0.5},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=season,
                    source="fangraphs",
                    avg=actual_avg,
                    obp=actual_avg + 0.050,
                    pa=400 + pid * 10,
                    war=2.0 + pid * 0.5,
                )
            )
    conn.commit()


def _seed_pitcher_data(
    proj_repo: SqliteProjectionRepo,
    pitching_repo: SqlitePitchingStatsRepo,
    conn: sqlite3.Connection,
    *,
    n_players: int = 10,
    seasons: tuple[int, ...] = (2024,),
    bias: float = 0.0,
) -> None:
    """Seed pitcher projections and actuals with controllable bias."""
    for pid in range(101, 101 + n_players):
        seed_player(conn, player_id=pid, name_last=f"Pitcher{pid}")
        for season in seasons:
            est_era = 3.50 + (pid - 101) * 0.10
            actual_era = est_era + bias + ((pid - 101) % 3 - 1) * 0.15
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=season,
                    system="test-sys",
                    version="v1",
                    player_type="pitcher",
                    stat_json={"era": est_era, "whip": 1.20 + (pid - 101) * 0.02, "war": 2.0},
                )
            )
            pitching_repo.upsert(
                PitchingStats(
                    player_id=pid,
                    season=season,
                    source="fangraphs",
                    era=actual_era,
                    whip=1.20 + (pid - 101) * 0.02 + bias,
                    ip=100.0 + (pid - 101) * 5,
                    war=2.0,
                )
            )
    conn.commit()


class TestBothPlayerTypes:
    def test_produces_batter_and_pitcher_analyses(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, pitching_repo = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, seasons=(2024,))
        _seed_pitcher_data(proj_repo, pitching_repo, conn, seasons=(2024,))

        report = service.analyze("test-sys", "v1", seasons=[2024])

        batter_stats = [a for a in report.stat_analyses if a.player_type == "batter"]
        pitcher_stats = [a for a in report.stat_analyses if a.player_type == "pitcher"]
        assert len(batter_stats) > 0
        assert len(pitcher_stats) > 0

    def test_report_metadata(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, pitching_repo = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, seasons=(2024,))
        _seed_pitcher_data(proj_repo, pitching_repo, conn, seasons=(2024,))

        report = service.analyze("test-sys", "v1", seasons=[2024])

        assert report.system == "test-sys"
        assert report.version == "v1"
        assert report.seasons == [2024]
        assert report.top is None


class TestBiasDetection:
    def test_no_bias_detected_with_balanced_residuals(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=50, seasons=(2024,), bias=0.0)

        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"])

        avg_analysis = report.stat_analyses[0]
        assert avg_analysis.stat_name == "avg"
        assert abs(avg_analysis.mean_residual) < 0.01

    def test_positive_bias_detected_with_systematic_underprediction(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=80, seasons=(2024,), bias=0.030)

        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"])

        avg_analysis = report.stat_analyses[0]
        assert avg_analysis.mean_residual > 0.02
        assert avg_analysis.bias_significant


class TestTopNFiltering:
    def test_top_n_filters_by_war(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=20, seasons=(2024,))

        # top=5 should use only the 5 highest-WAR players per season
        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"], top=5)

        avg_analysis = report.stat_analyses[0]
        assert avg_analysis.n_observations == 5


class TestMultiSeasonPooling:
    def test_observations_pooled_across_seasons(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=10, seasons=(2023, 2024))

        report = service.analyze("test-sys", "v1", seasons=[2023, 2024], stats=["avg"])

        avg_analysis = report.stat_analyses[0]
        # 10 players × 2 seasons = 20 observations
        assert avg_analysis.n_observations == 20


class TestSummaryLogic:
    def test_calibration_recommended_when_any_bias_significant(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=80, seasons=(2024,), bias=0.030)

        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"])

        assert report.summary.calibration_recommended

    def test_calibration_not_recommended_when_no_bias(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, n_players=50, seasons=(2024,), bias=0.0)

        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"])

        # With balanced residuals, bias should not be significant
        assert not report.summary.calibration_recommended


class TestStatFilter:
    def test_filters_to_requested_stats_only(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)
        _seed_batter_data(proj_repo, batting_repo, conn, seasons=(2024,))

        report = service.analyze("test-sys", "v1", seasons=[2024], stats=["avg"])

        stat_names = [a.stat_name for a in report.stat_analyses]
        assert stat_names == ["avg"]
