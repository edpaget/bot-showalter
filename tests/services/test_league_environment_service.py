import sqlite3

import pytest

from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats
from fantasy_baseball_manager.repos.league_environment_repo import SqliteLeagueEnvironmentRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.services.league_environment_service import LeagueEnvironmentService
from tests.helpers import seed_player


def _make_stats(player_id: int, **overrides: object) -> MinorLeagueBattingStats:
    defaults: dict[str, object] = {
        "player_id": player_id,
        "season": 2024,
        "level": "AAA",
        "league": "International League",
        "team": "Syracuse Mets",
        "g": 100,
        "pa": 400,
        "ab": 350,
        "h": 91,
        "doubles": 20,
        "triples": 2,
        "hr": 12,
        "r": 50,
        "rbi": 45,
        "bb": 35,
        "so": 80,
        "sb": 10,
        "cs": 3,
        "avg": 0.260,
        "obp": 0.330,
        "slg": 0.420,
        "age": 24.5,
        "hbp": 5,
        "sf": 4,
        "sh": 1,
    }
    defaults.update(overrides)
    return MinorLeagueBattingStats(**defaults)  # type: ignore[arg-type]


def _make_service(conn: sqlite3.Connection) -> tuple[LeagueEnvironmentService, SqliteLeagueEnvironmentRepo]:
    stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)
    env_repo = SqliteLeagueEnvironmentRepo(conn)
    service = LeagueEnvironmentService(stats_repo, env_repo)
    return service, env_repo


class TestLeagueEnvironmentService:
    def test_compute_from_known_aggregate(self, conn: sqlite3.Connection) -> None:
        p1 = seed_player(conn, mlbam_id=1)
        p2 = seed_player(conn, mlbam_id=2)
        stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)

        # Player 1: 400 PA, 350 AB, 91 H, 20 2B, 2 3B, 12 HR, 50 R, 100 G, 35 BB, 80 SO, 5 HBP, 4 SF
        stats_repo.upsert(_make_stats(p1))
        # Player 2: 200 PA, 175 AB, 49 H, 10 2B, 1 3B, 8 HR, 30 R, 50 G, 20 BB, 40 SO, 3 HBP, 2 SF
        stats_repo.upsert(
            _make_stats(
                p2,
                pa=200,
                ab=175,
                h=49,
                doubles=10,
                triples=1,
                hr=8,
                r=30,
                g=50,
                bb=20,
                so=40,
                hbp=3,
                sf=2,
                age=23.0,
                avg=0.280,
                obp=0.360,
                slg=0.460,
                team="Buffalo Bisons",
            )
        )
        conn.commit()

        service, _ = _make_service(conn)
        env = service.compute_for_league("International League", 2024, "AAA")

        # Totals: PA=600, AB=525, H=140, 2B=30, 3B=3, HR=20, R=80, G=150, BB=55, SO=120, HBP=8, SF=6
        # avg = 140/525
        assert env.avg == pytest.approx(140 / 525, abs=1e-6)
        # obp = (140+55+8) / (525+55+8+6) = 203/594
        assert env.obp == pytest.approx(203 / 594, abs=1e-6)
        # total_bases = 140 + 30 + 2*3 + 3*20 = 140 + 30 + 6 + 60 = 236
        # slg = 236/525
        assert env.slg == pytest.approx(236 / 525, abs=1e-6)
        # k_pct = 120/600
        assert env.k_pct == pytest.approx(120 / 600, abs=1e-6)
        # bb_pct = 55/600
        assert env.bb_pct == pytest.approx(55 / 600, abs=1e-6)
        # hr_per_pa = 20/600
        assert env.hr_per_pa == pytest.approx(20 / 600, abs=1e-6)
        # babip = (140-20) / (525 - 120 - 20 + 6) = 120/391
        assert env.babip == pytest.approx(120 / 391, abs=1e-6)
        # runs_per_game = 80/150
        assert env.runs_per_game == pytest.approx(80 / 150, abs=1e-6)

        assert env.league == "International League"
        assert env.season == 2024
        assert env.level == "AAA"

    def test_compute_and_persist(self, conn: sqlite3.Connection) -> None:
        p1 = seed_player(conn, mlbam_id=1)
        stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats_repo.upsert(_make_stats(p1))
        conn.commit()

        service, env_repo = _make_service(conn)
        env = service.compute_and_persist("International League", 2024, "AAA")
        conn.commit()

        stored = env_repo.get_by_league_season_level("International League", 2024, "AAA")
        assert stored is not None
        assert stored.avg == env.avg
        assert stored.obp == env.obp
        assert stored.id is not None

    def test_compute_for_season_level_all_leagues(self, conn: sqlite3.Connection) -> None:
        p1 = seed_player(conn, mlbam_id=1)
        p2 = seed_player(conn, mlbam_id=2)
        stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats_repo.upsert(_make_stats(p1, league="International League"))
        stats_repo.upsert(_make_stats(p2, league="Pacific Coast League", team="Reno Aces"))
        conn.commit()

        service, env_repo = _make_service(conn)
        count = service.compute_for_season_level(2024, "AAA")
        conn.commit()

        assert count == 2
        envs = env_repo.get_by_season_level(2024, "AAA")
        assert len(envs) == 2
        leagues = {e.league for e in envs}
        assert leagues == {"International League", "Pacific Coast League"}

    def test_empty_stats_raises(self, conn: sqlite3.Connection) -> None:
        service, _ = _make_service(conn)
        with pytest.raises(ValueError, match="No stats found"):
            service.compute_for_league("Nonexistent League", 2024, "AAA")

    def test_compute_handles_none_sf_and_hbp(self, conn: sqlite3.Connection) -> None:
        """When sf/hbp are None, they default to 0 in aggregation."""
        p1 = seed_player(conn, mlbam_id=1)
        stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats_repo.upsert(_make_stats(p1, hbp=None, sf=None))
        conn.commit()

        service, _ = _make_service(conn)
        env = service.compute_for_league("International League", 2024, "AAA")

        # With sf=0, hbp=0: obp = (91+35+0)/(350+35+0+0) = 126/385
        assert env.obp == pytest.approx(126 / 385, abs=1e-6)
        # babip with sf=0: (91-12)/(350-80-12+0) = 79/258
        assert env.babip == pytest.approx(79 / 258, abs=1e-6)
