import sqlite3

import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.true_talent_evaluator import TrueTalentEvaluator


def _make_service(
    conn: sqlite3.Connection,
) -> tuple[TrueTalentEvaluator, SqliteProjectionRepo, SqliteBattingStatsRepo, SqlitePitchingStatsRepo]:
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    service = TrueTalentEvaluator(proj_repo, batting_repo, pitching_repo)
    return service, proj_repo, batting_repo, pitching_repo


def _seed_player(conn: sqlite3.Connection, player_id: int, first: str = "Player", last: str | None = None) -> None:
    last = last or str(player_id)
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (?, ?, ?, '1990-01-01', 'R')",
        (player_id, first, last),
    )
    conn.commit()


class TestDataAssembly:
    def test_returning_players_and_metadata(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        for pid in (1, 2, 3):
            _seed_player(conn, pid)

        # Season N (2024): all 3 players have projections + actuals
        for pid in (1, 2, 3):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280 + pid * 0.01},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    avg=0.275 + pid * 0.01,
                    pa=400,
                )
            )

        # Season N+1 (2025): only players 1 and 2 return
        for pid in (1, 2):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.285 + pid * 0.01},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    avg=0.280 + pid * 0.01,
                    pa=400,
                )
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])

        assert report.system == "test-sys"
        assert report.version == "v1"
        assert report.season_n == 2024
        assert report.season_n1 == 2025
        assert report.player_type == "batter"
        assert len(report.stat_metrics) == 1
        m = report.stat_metrics[0]
        assert m.stat_name == "avg"
        assert m.n_season_n == 3
        assert m.n_returning == 2


class TestPredictiveValidity:
    def test_model_beats_raw(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Create 6 players with model estimates that track N+1 better than raw
        for pid in range(1, 7):
            _seed_player(conn, pid)

        # Model estimates in season N: close to true talent
        model_estimates = [0.270, 0.280, 0.290, 0.300, 0.310, 0.320]
        # Raw actuals in season N: noisier
        raw_actuals_n = [0.310, 0.250, 0.320, 0.260, 0.330, 0.270]
        # Actuals in season N+1: close to model estimates
        actuals_n1 = [0.272, 0.278, 0.292, 0.298, 0.312, 0.318]

        for i, pid in enumerate(range(1, 7)):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_estimates[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    avg=raw_actuals_n[i],
                    pa=400,
                )
            )
            # N+1 data
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_estimates[i] + 0.002},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    avg=actuals_n1[i],
                    pa=400,
                )
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        m = report.stat_metrics[0]
        assert m.model_next_season_corr > m.raw_next_season_corr
        assert m.predictive_validity_pass is True


class TestResidualNonPersistence:
    def test_low_residual_correlation_passes(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Use 8 players where residuals in season N don't predict residuals in N+1
        estimates_n = [0.270, 0.280, 0.290, 0.300, 0.310, 0.320, 0.275, 0.295]
        actuals_n = [0.280, 0.275, 0.285, 0.310, 0.305, 0.330, 0.270, 0.300]
        # Residuals_N = actuals_n - estimates_n: [0.01, -0.005, -0.005, 0.01, -0.005, 0.01, -0.005, 0.005]
        estimates_n1 = [0.272, 0.282, 0.292, 0.302, 0.312, 0.322, 0.277, 0.297]
        # N+1 actuals designed so residuals don't correlate with season N residuals
        actuals_n1 = [0.277, 0.277, 0.302, 0.307, 0.302, 0.317, 0.287, 0.292]
        # Residuals_N1: [0.005, -0.005, 0.01, 0.005, -0.01, -0.005, 0.01, -0.005]

        for i, pid in enumerate(range(1, 9)):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": estimates_n[i]},
                )
            )
            batting_repo.upsert(BattingStats(player_id=pid, season=2024, source="fangraphs", avg=actuals_n[i], pa=400))
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": estimates_n1[i]},
                )
            )
            batting_repo.upsert(BattingStats(player_id=pid, season=2025, source="fangraphs", avg=actuals_n1[i], pa=400))
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        m = report.stat_metrics[0]
        assert abs(m.residual_yoy_corr) < 0.15
        assert m.residual_non_persistence_pass is True


class TestShrinkageQuality:
    def test_shrinkage_ratio_below_one(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Estimates have lower variance than raw actuals
        model_estimates = [0.285, 0.290, 0.295, 0.300, 0.305]
        raw_actuals = [0.250, 0.270, 0.300, 0.330, 0.350]

        for i, pid in enumerate(range(1, 6)):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_estimates[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2024, source="fangraphs", avg=raw_actuals[i], pa=400)
            )
            # Need N+1 data too for returning players
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_estimates[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2025, source="fangraphs", avg=raw_actuals[i], pa=400)
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        m = report.stat_metrics[0]
        assert m.shrinkage_ratio < 1.0
        assert m.shrinkage_pass is True


class TestRSquaredDecomposition:
    def test_r_squared_and_buckets(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Model estimates closely match actuals
        estimates = [0.280, 0.300, 0.260, 0.290, 0.310]
        actuals = [0.282, 0.298, 0.262, 0.288, 0.312]
        pa_values = [100, 300, 500, 150, 450]

        for i, pid in enumerate(range(1, 6)):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": estimates[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2024, source="fangraphs", avg=actuals[i], pa=pa_values[i])
            )
            # Need N+1 for returning players
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": estimates[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2025, source="fangraphs", avg=actuals[i], pa=pa_values[i])
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        m = report.stat_metrics[0]
        assert m.r_squared > 0.9
        assert m.r_squared_pass is True
        assert "<200" in m.residual_by_bucket
        assert "200-400" in m.residual_by_bucket
        assert "400+" in m.residual_by_bucket


class TestStatFilter:
    def test_only_requested_stats_appear(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        for pid in range(1, 4):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280 + pid * 0.01, "obp": 0.350 + pid * 0.01},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    avg=0.285 + pid * 0.01,
                    obp=0.355 + pid * 0.01,
                    pa=400,
                )
            )
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.282 + pid * 0.01, "obp": 0.352 + pid * 0.01},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    avg=0.283 + pid * 0.01,
                    obp=0.353 + pid * 0.01,
                    pa=400,
                )
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        assert len(report.stat_metrics) == 1
        assert report.stat_metrics[0].stat_name == "avg"


class TestNoReturningPlayers:
    def test_graceful_zero_metrics(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Only season N data, no N+1 data
        for pid in range(1, 4):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280},
                )
            )
            batting_repo.upsert(BattingStats(player_id=pid, season=2024, source="fangraphs", avg=0.285, pa=400))
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg"])
        m = report.stat_metrics[0]
        assert m.n_returning == 0
        assert m.model_next_season_corr == 0.0
        assert m.raw_next_season_corr == 0.0
        assert m.residual_yoy_corr == 0.0
        assert m.r_squared == 0.0
        assert m.regression_rate == pytest.approx(1.0)


class TestPitcherEvaluation:
    def test_pitcher_type_works(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, pitching_repo = _make_service(conn)

        for pid in range(10, 16):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="pitcher",
                    stat_json={"era": 3.50 + (pid - 10) * 0.1},
                )
            )
            pitching_repo.upsert(
                PitchingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    era=3.55 + (pid - 10) * 0.1,
                    ip=150.0,
                )
            )
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="pitcher",
                    stat_json={"era": 3.52 + (pid - 10) * 0.1},
                )
            )
            pitching_repo.upsert(
                PitchingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    era=3.53 + (pid - 10) * 0.1,
                    ip=150.0,
                )
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "pitcher", stats=["era"])
        assert report.player_type == "pitcher"
        assert len(report.stat_metrics) == 1
        assert report.stat_metrics[0].stat_name == "era"
        assert report.stat_metrics[0].n_returning == 6


class TestSummaryAggregation:
    def test_summary_counts_passes(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Create 6 players with good data for avg and obp
        model_est_avg = [0.270, 0.280, 0.290, 0.300, 0.310, 0.320]
        model_est_obp = [0.340, 0.350, 0.360, 0.370, 0.380, 0.390]
        raw_avg = [0.310, 0.250, 0.320, 0.260, 0.330, 0.270]
        raw_obp = [0.380, 0.320, 0.390, 0.330, 0.400, 0.340]
        n1_avg = [0.272, 0.278, 0.292, 0.298, 0.312, 0.318]
        n1_obp = [0.342, 0.348, 0.362, 0.368, 0.382, 0.388]

        for i, pid in enumerate(range(1, 7)):
            _seed_player(conn, pid)
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_est_avg[i], "obp": model_est_obp[i]},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    avg=raw_avg[i],
                    obp=raw_obp[i],
                    pa=400,
                )
            )
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": model_est_avg[i] + 0.002, "obp": model_est_obp[i] + 0.002},
                )
            )
            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    avg=n1_avg[i],
                    obp=n1_obp[i],
                    pa=400,
                )
            )
        conn.commit()

        report = service.evaluate("test-sys", "v1", 2024, 2025, "batter", stats=["avg", "obp"])
        assert len(report.stat_metrics) == 2
        # Summary totals should equal number of stats
        assert report.summary.predictive_validity_total == 2
        assert report.summary.shrinkage_total == 2
