from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.yahoo_team_stats import TeamSeasonStats
from fantasy_baseball_manager.repos.yahoo_team_stats_repo import (
    SqliteYahooTeamStatsRepo,
)

if TYPE_CHECKING:
    import sqlite3


def _make_stats(**overrides: object) -> TeamSeasonStats:
    defaults: dict[str, object] = {
        "team_key": "458.l.135575.t.10",
        "league_key": "458.l.135575",
        "season": 2025,
        "team_name": "My Team",
        "final_rank": 1,
        "stat_values": {"hr": 250.0, "era": 3.45},
    }
    defaults.update(overrides)
    return TeamSeasonStats(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooTeamStatsRepo:
    def test_upsert_and_get_by_league_season(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_stats())
        results = repo.get_by_league_season("458.l.135575", 2025)
        assert len(results) == 1
        assert results[0].team_key == "458.l.135575.t.10"
        assert results[0].team_name == "My Team"
        assert results[0].final_rank == 1
        assert results[0].stat_values == {"hr": 250.0, "era": 3.45}
        assert results[0].id is not None

    def test_upsert_conflict_updates(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_stats(team_name="Old Name", final_rank=5))
        repo.upsert(_make_stats(team_name="New Name", final_rank=1))
        results = repo.get_by_league_season("458.l.135575", 2025)
        assert len(results) == 1
        assert results[0].team_name == "New Name"
        assert results[0].final_rank == 1

    def test_get_by_league_season_ordered_by_rank(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_stats(team_key="458.l.135575.t.3", final_rank=3, team_name="Third"))
        repo.upsert(_make_stats(team_key="458.l.135575.t.1", final_rank=1, team_name="First"))
        repo.upsert(_make_stats(team_key="458.l.135575.t.2", final_rank=2, team_name="Second"))
        results = repo.get_by_league_season("458.l.135575", 2025)
        assert [r.final_rank for r in results] == [1, 2, 3]

    def test_get_by_league_season_returns_empty(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        assert repo.get_by_league_season("nonexistent", 2025) == []

    def test_get_all_seasons(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_stats(league_key="458.l.100", season=2024, team_key="t1", final_rank=1))
        repo.upsert(_make_stats(league_key="458.l.100", season=2025, team_key="t2", final_rank=1))
        results = repo.get_all_seasons("458.l.100")
        assert len(results) == 2
        assert results[0].season == 2024
        assert results[1].season == 2025

    def test_json_round_trip(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        stat_values = {"r": 764.0, "hr": 198.0, "rbi": 667.0, "sb": 184.0, "obp": 0.340, "era": 3.55, "whip": 1.15}
        repo.upsert(_make_stats(stat_values=stat_values))
        results = repo.get_by_league_season("458.l.135575", 2025)
        assert results[0].stat_values == stat_values

    def test_upsert_returns_row_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn))
        row_id = repo.upsert(_make_stats())
        assert isinstance(row_id, int)
        assert row_id > 0
