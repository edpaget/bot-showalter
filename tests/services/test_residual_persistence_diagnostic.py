import sqlite3

import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.residual_persistence_diagnostic import ResidualPersistenceDiagnostic
from tests.helpers import seed_player


def _make_service(
    conn: sqlite3.Connection,
) -> tuple[ResidualPersistenceDiagnostic, SqliteProjectionRepo, SqliteBattingStatsRepo, SqlitePlayerRepo]:
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    player_repo = SqlitePlayerRepo(conn)
    service = ResidualPersistenceDiagnostic(proj_repo, batting_repo, player_repo)
    return service, proj_repo, batting_repo, player_repo


class TestDataAssembly:
    def test_returning_players_and_metadata(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        for pid in (1, 2, 3):
            seed_player(conn, player_id=pid, name_last=f"P{pid}")

        # Season N: all 3 players
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

        # Season N+1: only players 1 and 2 return
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

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])

        assert report.system == "test-sys"
        assert report.version == "v1"
        assert report.season_n == 2024
        assert report.season_n1 == 2025
        assert len(report.stat_metrics) == 1
        m = report.stat_metrics[0]
        assert m.stat_name == "avg"
        assert m.n_returning == 2


class TestResidualCorrelation:
    def test_overall_and_per_bucket(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # 8 players with systematic residual pattern
        # Players with positive residuals in N also have positive in N+1 → positive correlation
        estimates_n = [0.270, 0.280, 0.290, 0.300, 0.260, 0.310, 0.275, 0.295]
        actuals_n = [0.290, 0.300, 0.280, 0.280, 0.280, 0.330, 0.295, 0.275]
        # Residuals_N:  [+.020, +.020, -.010, -.020, +.020, +.020, +.020, -.020]

        estimates_n1 = [0.272, 0.282, 0.292, 0.302, 0.262, 0.312, 0.277, 0.297]
        actuals_n1 = [0.292, 0.302, 0.282, 0.282, 0.282, 0.332, 0.297, 0.277]
        # Residuals_N1: [+.020, +.020, -.010, -.020, +.020, +.020, +.020, -.020]

        pa_values = [100, 150, 250, 350, 450, 500, 300, 550]

        for i, pid in enumerate(range(1, 9)):
            seed_player(conn, player_id=pid, name_last=f"P{pid}")
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
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2024, source="fangraphs", avg=actuals_n[i], pa=pa_values[i])
            )
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
            batting_repo.upsert(
                BattingStats(player_id=pid, season=2025, source="fangraphs", avg=actuals_n1[i], pa=pa_values[i])
            )
        conn.commit()

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])
        m = report.stat_metrics[0]

        # Overall correlation should be high (residuals are identical across seasons)
        assert m.residual_corr_overall == pytest.approx(1.0)
        assert m.persistence_pass is True  # r > 0.10
        # Per-bucket correlations should exist
        assert len(m.residual_corr_by_bucket) > 0


class TestChronicPerformers:
    def test_identifies_over_and_underperformers(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # 6 players: 2 chronic overperformers, 1 chronic underperformer, rest normal
        estimates_n = [0.280, 0.280, 0.280, 0.280, 0.280, 0.280]
        # Overperformers: pid 1,2 have large positive residuals
        # Underperformer: pid 3 has large negative residual
        # Normal: pid 4,5,6 near zero
        actuals_n = [0.340, 0.330, 0.220, 0.282, 0.278, 0.281]

        estimates_n1 = [0.282, 0.282, 0.282, 0.282, 0.282, 0.282]
        actuals_n1 = [0.342, 0.332, 0.222, 0.280, 0.284, 0.283]

        for i, pid in enumerate(range(1, 7)):
            first = ["Over1", "Over2", "Under1", "Normal1", "Normal2", "Normal3"][i]
            seed_player(conn, player_id=pid, name_first=first, name_last=f"L{pid}")
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

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])
        m = report.stat_metrics[0]

        assert len(m.chronic_overperformers) == 2
        assert len(m.chronic_underperformers) == 1
        assert m.chronic_overperformers[0].player_name == "Over1 L1"
        assert m.chronic_underperformers[0].player_name == "Under1 L3"


class TestRmseCeiling:
    def test_persistent_residuals_yield_improvement(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Players with systematic bias: estimates consistently under-predict
        estimates_n = [0.270, 0.280, 0.290, 0.300]
        actuals_n = [0.290, 0.300, 0.310, 0.320]  # all +0.020 residual

        estimates_n1 = [0.272, 0.282, 0.292, 0.302]
        actuals_n1 = [0.292, 0.302, 0.312, 0.322]  # same +0.020 bias

        for i, pid in enumerate(range(1, 5)):
            seed_player(conn, player_id=pid)
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

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])
        m = report.stat_metrics[0]

        assert m.rmse_baseline > 0
        assert m.rmse_corrected < m.rmse_baseline
        assert m.rmse_improvement_pct > 0


class TestGoNoGoSummary:
    def test_summary_counts_and_go_flag(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # 6 players with per-player biases that persist across seasons.
        # Each player's bias differs so residuals have nonzero variance (needed for r > 0).
        biases = [0.010, 0.020, 0.030, -0.010, -0.020, 0.015]

        for i, pid in enumerate(range(1, 7)):
            seed_player(conn, player_id=pid)

            avg_est = 0.270 + i * 0.010
            obp_est = 0.340 + i * 0.010
            slg_est = 0.420 + i * 0.010
            woba_est = 0.320 + i * 0.010
            iso_est = slg_est - avg_est
            babip_est = 0.290 + i * 0.010

            bias = biases[i]

            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2024,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={
                        "avg": avg_est,
                        "obp": obp_est,
                        "slg": slg_est,
                        "woba": woba_est,
                        "iso": iso_est,
                        "babip": babip_est,
                    },
                )
            )

            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2024,
                    source="fangraphs",
                    avg=avg_est + bias,
                    obp=obp_est + bias,
                    slg=slg_est + bias,
                    woba=woba_est + bias,
                    pa=400,
                    ab=380,
                    h=int(380 * (avg_est + bias)),
                    hr=10,
                    so=80,
                    sf=5,
                )
            )

            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test-sys",
                    version="v1",
                    player_type="batter",
                    stat_json={
                        "avg": avg_est + 0.002,
                        "obp": obp_est + 0.002,
                        "slg": slg_est + 0.002,
                        "woba": woba_est + 0.002,
                        "iso": iso_est + 0.002,
                        "babip": babip_est + 0.002,
                    },
                )
            )

            batting_repo.upsert(
                BattingStats(
                    player_id=pid,
                    season=2025,
                    source="fangraphs",
                    avg=avg_est + 0.002 + bias,
                    obp=obp_est + 0.002 + bias,
                    slg=slg_est + 0.002 + bias,
                    woba=woba_est + 0.002 + bias,
                    pa=420,
                    ab=400,
                    h=int(400 * (avg_est + 0.002 + bias)),
                    hr=12,
                    so=85,
                    sf=4,
                )
            )
        conn.commit()

        report = service.diagnose("test-sys", "v1", 2024, 2025)
        s = report.summary

        # With all 6 stats having persistent residuals, should have many passes
        assert s.persistence_total == 6
        assert s.ceiling_total == 6
        # Go requires persistence_passes >= 3 AND ceiling_passes >= 2
        # With per-player persistent biases, most stats should pass both
        assert s.persistence_passes >= 3
        assert s.ceiling_passes >= 2
        assert s.go is True


class TestNoReturningPlayers:
    def test_graceful_zeros(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        # Only season N data
        for pid in range(1, 4):
            seed_player(conn, player_id=pid)
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

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])
        m = report.stat_metrics[0]
        assert m.n_returning == 0
        assert m.residual_corr_overall == 0.0
        assert m.rmse_baseline == 0.0
        assert m.rmse_corrected == 0.0
        assert m.rmse_improvement_pct == 0.0
        assert m.persistence_pass is False
        assert m.ceiling_pass is False


class TestStatFilter:
    def test_only_requested_stats_returned(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, batting_repo, _ = _make_service(conn)

        for pid in range(1, 4):
            seed_player(conn, player_id=pid)
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

        report = service.diagnose("test-sys", "v1", 2024, 2025, stats=["avg"])
        assert len(report.stat_metrics) == 1
        assert report.stat_metrics[0].stat_name == "avg"
