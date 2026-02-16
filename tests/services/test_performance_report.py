import sqlite3

import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.performance_report import PerformanceReportService


_ServiceTuple = tuple[
    PerformanceReportService, SqliteProjectionRepo, SqlitePlayerRepo, SqliteBattingStatsRepo, SqlitePitchingStatsRepo
]


def _make_service(conn: sqlite3.Connection) -> _ServiceTuple:
    proj_repo = SqliteProjectionRepo(conn)
    player_repo = SqlitePlayerRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    service = PerformanceReportService(proj_repo, player_repo, batting_repo, pitching_repo)
    return service, proj_repo, player_repo, batting_repo, pitching_repo


def _seed_player(conn: sqlite3.Connection, player_id: int, first: str = "Player", last: str | None = None) -> None:
    last = last or str(player_id)
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (?, ?, ?, '1990-01-01', 'R')",
        (player_id, first, last),
    )
    conn.commit()


class TestComputeDeltasBatterDirectStats:
    def test_compute_deltas_batter_direct_stats(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)

        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"avg": 0.280, "obp": 0.350},
            )
        )
        proj_repo.upsert(
            Projection(
                player_id=2,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"avg": 0.300, "obp": 0.370},
            )
        )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.290, obp=0.360))
        batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", avg=0.280, obp=0.350))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg", "obp"])
        avg_deltas = [d for d in deltas if d.stat_name == "avg"]
        obp_deltas = [d for d in deltas if d.stat_name == "obp"]
        assert len(avg_deltas) == 2
        assert len(obp_deltas) == 2

        # Player 1: avg actual 0.290 - expected 0.280 = +0.010
        p1_avg = [d for d in avg_deltas if d.player_id == 1][0]
        assert p1_avg.delta == pytest.approx(0.010, abs=1e-6)
        assert p1_avg.performance_delta == pytest.approx(0.010, abs=1e-6)


class TestComputeDeltasPitcherDirectStats:
    def test_compute_deltas_pitcher_direct_stats(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        for pid in (10, 11):
            _seed_player(conn, pid)

        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 3.50, "fip": 3.20},
            )
        )
        proj_repo.upsert(
            Projection(
                player_id=11,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 4.00, "fip": 3.80},
            )
        )
        pitching_repo.upsert(PitchingStats(player_id=10, season=2025, source="fangraphs", era=3.20, fip=3.00))
        pitching_repo.upsert(PitchingStats(player_id=11, season=2025, source="fangraphs", era=4.20, fip=4.00))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["era", "fip"])
        era_deltas = [d for d in deltas if d.stat_name == "era"]
        assert len(era_deltas) == 2


class TestComputeDeltasDerivedBatterIso:
    def test_compute_deltas_derived_batter_iso(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        _seed_player(conn, 1)

        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"iso": 0.200},
            )
        )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.280, slg=0.500))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["iso"])
        assert len(deltas) == 1
        # iso = slg - avg = 0.500 - 0.280 = 0.220
        assert deltas[0].actual == pytest.approx(0.220, abs=1e-6)
        assert deltas[0].delta == pytest.approx(0.020, abs=1e-6)


class TestComputeDeltasDerivedBatterBabip:
    def test_compute_deltas_derived_batter_babip(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        _seed_player(conn, 1)

        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"babip": 0.300},
            )
        )
        # babip = (h - hr) / (ab - so - hr + sf)
        # = (150 - 30) / (500 - 100 - 30 + 5) = 120 / 375 = 0.320
        batting_repo.upsert(
            BattingStats(
                player_id=1,
                season=2025,
                source="fangraphs",
                h=150,
                hr=30,
                ab=500,
                so=100,
                sf=5,
            )
        )
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["babip"])
        assert len(deltas) == 1
        assert deltas[0].actual == pytest.approx(0.320, abs=1e-6)
        assert deltas[0].delta == pytest.approx(0.020, abs=1e-6)


class TestComputeDeltasDerivedPitcherHrPer9:
    def test_compute_deltas_derived_pitcher_hr_per_9(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        _seed_player(conn, 10)

        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"hr_per_9": 1.00},
            )
        )
        # hr_per_9 = hr * 9 / ip = 20 * 9 / 180 = 1.00
        pitching_repo.upsert(PitchingStats(player_id=10, season=2025, source="fangraphs", hr=20, ip=180.0))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["hr_per_9"])
        assert len(deltas) == 1
        assert deltas[0].actual == pytest.approx(1.0, abs=1e-6)
        assert deltas[0].delta == pytest.approx(0.0, abs=1e-6)


class TestComputeDeltasDerivedPitcherBabip:
    def test_compute_deltas_derived_pitcher_babip(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        _seed_player(conn, 10)

        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"babip": 0.290},
            )
        )
        # babip = (h - hr) / (ip * 3 + h - so - hr)
        # = (170 - 20) / (180 * 3 + 170 - 180 - 20) = 150 / 510 = 0.29412...
        pitching_repo.upsert(
            PitchingStats(
                player_id=10,
                season=2025,
                source="fangraphs",
                h=170,
                hr=20,
                ip=180.0,
                so=180,
            )
        )
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["babip"])
        assert len(deltas) == 1
        assert deltas[0].actual == pytest.approx(150 / 510, abs=1e-6)


class TestComputeDeltasSkipsMissingProjection:
    def test_compute_deltas_skips_missing_projection(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        _seed_player(conn, 1)
        _seed_player(conn, 2)

        # Only player 1 has a projection
        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"avg": 0.280},
            )
        )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.290))
        batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", avg=0.310))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg"])
        assert len(deltas) == 1
        assert deltas[0].player_id == 1


class TestComputeDeltasSkipsMissingActual:
    def test_compute_deltas_skips_missing_actual(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        _seed_player(conn, 1)
        _seed_player(conn, 2)

        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"avg": 0.280},
            )
        )
        proj_repo.upsert(
            Projection(
                player_id=2,
                season=2025,
                system="test",
                version="v1",
                player_type="batter",
                stat_json={"avg": 0.300},
            )
        )
        # Only player 1 has actuals
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.290))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg"])
        assert len(deltas) == 1
        assert deltas[0].player_id == 1


class TestComputeDeltasPercentileRanking:
    def test_compute_deltas_percentile_ranking(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        for pid in (1, 2, 3):
            _seed_player(conn, pid)

        # All project avg=0.280, actuals differ to get different deltas
        for pid in (1, 2, 3):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280},
                )
            )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.260))  # delta = -0.020
        batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", avg=0.280))  # delta = 0.000
        batting_repo.upsert(BattingStats(player_id=3, season=2025, source="fangraphs", avg=0.300))  # delta = +0.020
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg"])
        assert len(deltas) == 3
        by_pid = {d.player_id: d for d in deltas}
        # (rank - 1) / (n - 1) * 100
        # Player 1: worst delta → rank 1 → percentile = 0
        # Player 2: middle → rank 2 → percentile = 50
        # Player 3: best delta → rank 3 → percentile = 100
        assert by_pid[1].percentile == pytest.approx(0.0)
        assert by_pid[2].percentile == pytest.approx(50.0)
        assert by_pid[3].percentile == pytest.approx(100.0)


class TestComputeDeltasInvertedStatPerformanceDelta:
    def test_compute_deltas_inverted_stat_performance_delta(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        for pid in (10, 11):
            _seed_player(conn, pid)

        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 4.00},
            )
        )
        proj_repo.upsert(
            Projection(
                player_id=11,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 3.50},
            )
        )
        # Player 10: actual ERA 3.50 < expected 4.00 → outperformed (lower is better)
        pitching_repo.upsert(PitchingStats(player_id=10, season=2025, source="fangraphs", era=3.50))
        # Player 11: actual ERA 4.00 > expected 3.50 → underperformed
        pitching_repo.upsert(PitchingStats(player_id=11, season=2025, source="fangraphs", era=4.00))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["era"])
        by_pid = {d.player_id: d for d in deltas}

        # Player 10: delta = 3.50 - 4.00 = -0.50, performance_delta = +0.50 (inverted)
        assert by_pid[10].delta == pytest.approx(-0.50)
        assert by_pid[10].performance_delta == pytest.approx(0.50)

        # Player 11: delta = 4.00 - 3.50 = +0.50, performance_delta = -0.50 (inverted)
        assert by_pid[11].delta == pytest.approx(0.50)
        assert by_pid[11].performance_delta == pytest.approx(-0.50)


class TestComputeDeltasFiltersByStat:
    def test_compute_deltas_filters_by_stat(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        _seed_player(conn, 10)

        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2025,
                system="test",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 3.50, "fip": 3.20, "whip": 1.10},
            )
        )
        pitching_repo.upsert(
            PitchingStats(
                player_id=10,
                season=2025,
                source="fangraphs",
                era=3.20,
                fip=3.00,
                whip=1.05,
            )
        )
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["era"])
        assert len(deltas) == 1
        assert deltas[0].stat_name == "era"


class TestMinPaFiltersBatters:
    def test_min_pa_filters_low_pa_batters(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)

        for pid in (1, 2):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280},
                )
            )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", pa=400, avg=0.290))
        batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", pa=50, avg=0.310))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg"], min_pa=100)
        assert len(deltas) == 1
        assert deltas[0].player_id == 1


class TestMinPaFiltersLowIpPitchers:
    def test_min_pa_filters_low_ip_pitchers(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, _, pitching_repo = _make_service(conn)
        for pid in (10, 11):
            _seed_player(conn, pid)

        for pid in (10, 11):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test",
                    version="v1",
                    player_type="pitcher",
                    stat_json={"era": 3.50},
                )
            )
        pitching_repo.upsert(PitchingStats(player_id=10, season=2025, source="fangraphs", ip=150.0, era=3.20))
        pitching_repo.upsert(PitchingStats(player_id=11, season=2025, source="fangraphs", ip=20.0, era=5.00))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "pitcher", stats=["era"], min_pa=50)
        assert len(deltas) == 1
        assert deltas[0].player_id == 10


class TestMinPaNoneIncludesAll:
    def test_min_pa_none_includes_all(self, conn: sqlite3.Connection) -> None:
        service, proj_repo, _, batting_repo, _ = _make_service(conn)
        for pid in (1, 2):
            _seed_player(conn, pid)

        for pid in (1, 2):
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2025,
                    system="test",
                    version="v1",
                    player_type="batter",
                    stat_json={"avg": 0.280},
                )
            )
        batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", pa=400, avg=0.290))
        batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", pa=50, avg=0.310))
        conn.commit()

        deltas = service.compute_deltas("test", "v1", 2025, "batter", stats=["avg"], min_pa=None)
        assert len(deltas) == 2
