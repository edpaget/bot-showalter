import sqlite3

from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from tests.helpers import seed_player


def _make_stats(player_id: int, **overrides: object) -> MinorLeagueBattingStats:
    defaults: dict[str, object] = {
        "player_id": player_id,
        "season": 2024,
        "level": "AAA",
        "league": "International League",
        "team": "Syracuse Mets",
        "g": 120,
        "pa": 500,
        "ab": 450,
        "h": 130,
        "doubles": 25,
        "triples": 3,
        "hr": 18,
        "r": 70,
        "rbi": 65,
        "bb": 40,
        "so": 100,
        "sb": 15,
        "cs": 5,
        "avg": 0.289,
        "obp": 0.350,
        "slg": 0.480,
        "age": 24.5,
    }
    defaults.update(overrides)
    return MinorLeagueBattingStats(**defaults)  # type: ignore[arg-type]


class TestMinorLeagueBattingStatsRepo:
    def test_upsert_and_get_by_player(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats = _make_stats(player_id)

        repo.upsert(stats)
        conn.commit()

        results = repo.get_by_player(player_id)
        assert len(results) == 1
        row = results[0]
        assert row.player_id == player_id
        assert row.season == 2024
        assert row.level == "AAA"
        assert row.league == "International League"
        assert row.team == "Syracuse Mets"
        assert row.g == 120
        assert row.pa == 500
        assert row.ab == 450
        assert row.h == 130
        assert row.doubles == 25
        assert row.triples == 3
        assert row.hr == 18
        assert row.r == 70
        assert row.rbi == 65
        assert row.bb == 40
        assert row.so == 100
        assert row.sb == 15
        assert row.cs == 5
        assert row.id is not None

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats = _make_stats(player_id, hr=18)

        repo.upsert(stats)
        conn.commit()

        updated_stats = _make_stats(player_id, hr=20)
        repo.upsert(updated_stats)
        conn.commit()

        results = repo.get_by_player(player_id)
        assert len(results) == 1
        assert results[0].hr == 20

    def test_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)

        repo.upsert(_make_stats(player_id, season=2023))
        repo.upsert(_make_stats(player_id, season=2024))
        conn.commit()

        results_2024 = repo.get_by_player_season(player_id, 2024)
        assert len(results_2024) == 1
        assert results_2024[0].season == 2024

        results_2023 = repo.get_by_player_season(player_id, 2023)
        assert len(results_2023) == 1
        assert results_2023[0].season == 2023

    def test_get_by_season_level(self, conn: sqlite3.Connection) -> None:
        p1 = seed_player(conn, mlbam_id=545361)
        p2 = seed_player(conn, mlbam_id=660271)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)

        repo.upsert(_make_stats(p1, level="AAA"))
        repo.upsert(_make_stats(p2, level="AAA"))
        repo.upsert(_make_stats(p1, level="AA"))
        conn.commit()

        aaa_results = repo.get_by_season_level(2024, "AAA")
        assert len(aaa_results) == 2

        aa_results = repo.get_by_season_level(2024, "AA")
        assert len(aa_results) == 1

    def test_multiple_levels_same_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)

        repo.upsert(_make_stats(player_id, level="AA", team="Binghamton Rumble Ponies"))
        repo.upsert(_make_stats(player_id, level="AAA", team="Syracuse Mets"))
        conn.commit()

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 2
        levels = {r.level for r in results}
        assert levels == {"AA", "AAA"}
