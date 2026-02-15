from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.domain.roster_stint import RosterStint
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo


def _seed_player(conn, *, mlbam_id: int = 545361) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=mlbam_id))


def _seed_team(conn, *, abbreviation: str = "LAA") -> int:
    repo = SqliteTeamRepo(conn)
    return repo.upsert(Team(abbreviation=abbreviation, name="Los Angeles Angels", league="AL", division="W"))


class TestSqliteRosterStintRepo:
    def test_upsert_and_get_by_player_season(self, conn) -> None:
        player_id = _seed_player(conn)
        team_id = _seed_team(conn)
        repo = SqliteRosterStintRepo(conn)
        stint = RosterStint(
            player_id=player_id,
            team_id=team_id,
            season=2024,
            start_date="2024-03-28",
        )

        stint_id = repo.upsert(stint)

        assert stint_id is not None
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].player_id == player_id
        assert results[0].team_id == team_id
        assert results[0].season == 2024
        assert results[0].start_date == "2024-03-28"
        assert results[0].id == stint_id

    def test_upsert_idempotency(self, conn) -> None:
        player_id = _seed_player(conn)
        team_id = _seed_team(conn)
        repo = SqliteRosterStintRepo(conn)
        repo.upsert(
            RosterStint(
                player_id=player_id,
                team_id=team_id,
                season=2024,
                start_date="2024-03-28",
            )
        )

        repo.upsert(
            RosterStint(
                player_id=player_id,
                team_id=team_id,
                season=2024,
                start_date="2024-03-28",
                end_date="2024-07-30",
            )
        )

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].end_date == "2024-07-30"

    def test_get_by_player_returns_all_seasons(self, conn) -> None:
        player_id = _seed_player(conn)
        team_id = _seed_team(conn)
        repo = SqliteRosterStintRepo(conn)
        repo.upsert(
            RosterStint(
                player_id=player_id,
                team_id=team_id,
                season=2023,
                start_date="2023-03-30",
            )
        )
        repo.upsert(
            RosterStint(
                player_id=player_id,
                team_id=team_id,
                season=2024,
                start_date="2024-03-28",
            )
        )

        results = repo.get_by_player(player_id)
        assert len(results) == 2
        seasons = {r.season for r in results}
        assert seasons == {2023, 2024}

    def test_get_by_team_season(self, conn) -> None:
        p1 = _seed_player(conn, mlbam_id=545361)
        p2 = _seed_player(conn, mlbam_id=660271)
        team_id = _seed_team(conn)
        repo = SqliteRosterStintRepo(conn)
        repo.upsert(RosterStint(player_id=p1, team_id=team_id, season=2024, start_date="2024-03-28"))
        repo.upsert(RosterStint(player_id=p2, team_id=team_id, season=2024, start_date="2024-03-28"))

        results = repo.get_by_team_season(team_id, 2024)
        assert len(results) == 2

    def test_get_by_season(self, conn) -> None:
        p1 = _seed_player(conn, mlbam_id=545361)
        team_id = _seed_team(conn)
        repo = SqliteRosterStintRepo(conn)
        repo.upsert(RosterStint(player_id=p1, team_id=team_id, season=2024, start_date="2024-03-28"))
        repo.upsert(RosterStint(player_id=p1, team_id=team_id, season=2023, start_date="2023-03-30"))

        results = repo.get_by_season(2024)
        assert len(results) == 1

        results_2023 = repo.get_by_season(2023)
        assert len(results_2023) == 1

    def test_get_by_player_empty(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqliteRosterStintRepo(conn)

        results = repo.get_by_player(player_id)
        assert results == []
