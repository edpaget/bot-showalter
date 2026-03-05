from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Team
from fantasy_baseball_manager.ingest.roster_api import ingest_roster_api
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from tests.helpers import seed_player


def _make_fetcher(data: dict[int, str]) -> object:
    """Return a callable that returns the given data regardless of season."""

    def fetcher(season: int) -> dict[int, str]:
        return data

    return fetcher


class TestIngestRosterApi:
    def test_happy_path(self) -> None:
        conn = create_connection(":memory:")
        player_repo = SqlitePlayerRepo(conn)
        team_repo = SqliteTeamRepo(conn)
        stint_repo = SqliteRosterStintRepo(conn)

        # Seed players and a team
        seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
        seed_player(conn, name_first="Shohei", name_last="Ohtani", mlbam_id=660271)
        team_repo.upsert(Team(abbreviation="LAA", name="Los Angeles Angels", league="AL", division="W"))
        team_repo.upsert(Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W"))
        conn.commit()

        fetcher = _make_fetcher({545361: "LAA", 660271: "LAD"})
        result = ingest_roster_api(fetcher, player_repo, team_repo, stint_repo, season=2026, as_of="2026-03-04")
        conn.commit()

        assert result.loaded == 2
        assert result.skipped == 0

        stints = stint_repo.get_by_season(2026)
        assert len(stints) == 2
        conn.close()

    def test_unknown_player_skipped(self) -> None:
        conn = create_connection(":memory:")
        player_repo = SqlitePlayerRepo(conn)
        team_repo = SqliteTeamRepo(conn)
        stint_repo = SqliteRosterStintRepo(conn)

        # Only seed one player — mlbam_id 999999 is unknown
        seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
        conn.commit()

        fetcher = _make_fetcher({545361: "LAA", 999999: "NYY"})
        result = ingest_roster_api(fetcher, player_repo, team_repo, stint_repo, season=2026, as_of="2026-03-04")
        conn.commit()

        assert result.loaded == 1
        assert result.skipped == 1

        stints = stint_repo.get_by_season(2026)
        assert len(stints) == 1
        conn.close()

    def test_unknown_team_auto_upserted(self) -> None:
        conn = create_connection(":memory:")
        player_repo = SqlitePlayerRepo(conn)
        team_repo = SqliteTeamRepo(conn)
        stint_repo = SqliteRosterStintRepo(conn)

        seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
        conn.commit()

        # No teams seeded — "LAA" should be auto-upserted
        fetcher = _make_fetcher({545361: "LAA"})
        result = ingest_roster_api(fetcher, player_repo, team_repo, stint_repo, season=2026, as_of="2026-03-04")
        conn.commit()

        assert result.loaded == 1
        team = team_repo.get_by_abbreviation("LAA")
        assert team is not None

        stints = stint_repo.get_by_season(2026)
        assert len(stints) == 1
        conn.close()

    def test_rerun_idempotent(self) -> None:
        conn = create_connection(":memory:")
        player_repo = SqlitePlayerRepo(conn)
        team_repo = SqliteTeamRepo(conn)
        stint_repo = SqliteRosterStintRepo(conn)

        seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
        conn.commit()

        fetcher = _make_fetcher({545361: "LAA"})

        # Run twice with same data
        ingest_roster_api(fetcher, player_repo, team_repo, stint_repo, season=2026, as_of="2026-03-04")
        conn.commit()
        ingest_roster_api(fetcher, player_repo, team_repo, stint_repo, season=2026, as_of="2026-03-04")
        conn.commit()

        stints = stint_repo.get_by_season(2026)
        assert len(stints) == 1  # No duplicates
        conn.close()
