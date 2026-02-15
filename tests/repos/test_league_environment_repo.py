import sqlite3

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.repos.league_environment_repo import SqliteLeagueEnvironmentRepo


def _make_env(**overrides: object) -> LeagueEnvironment:
    defaults: dict[str, object] = {
        "league": "International League",
        "season": 2024,
        "level": "AAA",
        "runs_per_game": 4.8,
        "avg": 0.260,
        "obp": 0.330,
        "slg": 0.420,
        "k_pct": 0.230,
        "bb_pct": 0.085,
        "hr_per_pa": 0.030,
        "babip": 0.300,
    }
    defaults.update(overrides)
    return LeagueEnvironment(**defaults)  # type: ignore[arg-type]


class TestLeagueEnvironmentRepo:
    def test_upsert_and_get_by_league_season_level(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLeagueEnvironmentRepo(conn)
        env = _make_env()

        row_id = repo.upsert(env)
        conn.commit()

        result = repo.get_by_league_season_level("International League", 2024, "AAA")
        assert result is not None
        assert result.id == row_id
        assert result.league == "International League"
        assert result.season == 2024
        assert result.level == "AAA"
        assert result.runs_per_game == 4.8
        assert result.avg == 0.260
        assert result.obp == 0.330
        assert result.slg == 0.420
        assert result.k_pct == 0.230
        assert result.bb_pct == 0.085
        assert result.hr_per_pa == 0.030
        assert result.babip == 0.300

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLeagueEnvironmentRepo(conn)

        repo.upsert(_make_env(avg=0.260))
        conn.commit()

        repo.upsert(_make_env(avg=0.275))
        conn.commit()

        result = repo.get_by_league_season_level("International League", 2024, "AAA")
        assert result is not None
        assert result.avg == 0.275

    def test_get_by_season_level(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLeagueEnvironmentRepo(conn)

        repo.upsert(_make_env(league="International League", level="AAA"))
        repo.upsert(_make_env(league="Pacific Coast League", level="AAA"))
        repo.upsert(_make_env(league="Eastern League", level="AA"))
        conn.commit()

        results = repo.get_by_season_level(2024, "AAA")
        assert len(results) == 2
        leagues = {r.league for r in results}
        assert leagues == {"International League", "Pacific Coast League"}

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLeagueEnvironmentRepo(conn)

        repo.upsert(_make_env(league="International League", level="AAA"))
        repo.upsert(_make_env(league="Eastern League", level="AA"))
        repo.upsert(_make_env(league="South Atlantic League", level="A"))
        conn.commit()

        results = repo.get_by_season(2024)
        assert len(results) == 3
        levels = {r.level for r in results}
        assert levels == {"AAA", "AA", "A"}

    def test_get_nonexistent_returns_none(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLeagueEnvironmentRepo(conn)
        result = repo.get_by_league_season_level("Nonexistent", 2024, "AAA")
        assert result is None
