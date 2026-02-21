import sqlite3

from fantasy_baseball_manager.domain.yahoo_league import YahooLeague, YahooTeam


class SqliteYahooLeagueRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, league: YahooLeague) -> int:
        cursor = self._conn.execute(
            "INSERT INTO yahoo_league"
            "    (league_key, name, season, num_teams, draft_type, is_keeper, game_key)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(league_key) DO UPDATE SET"
            "    name=excluded.name,"
            "    season=excluded.season,"
            "    num_teams=excluded.num_teams,"
            "    draft_type=excluded.draft_type,"
            "    is_keeper=excluded.is_keeper,"
            "    game_key=excluded.game_key",
            (
                league.league_key,
                league.name,
                league.season,
                league.num_teams,
                league.draft_type,
                int(league.is_keeper),
                league.game_key,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_league_key(self, league_key: str) -> YahooLeague | None:
        row = self._conn.execute(
            self._select_sql() + " WHERE league_key = ?",
            (league_key,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_league(row)

    def get_all(self) -> list[YahooLeague]:
        rows = self._conn.execute(self._select_sql()).fetchall()
        return [self._row_to_league(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, league_key, name, season, num_teams, draft_type, is_keeper, game_key FROM yahoo_league"

    @staticmethod
    def _row_to_league(row: sqlite3.Row) -> YahooLeague:
        return YahooLeague(
            id=row["id"],
            league_key=row["league_key"],
            name=row["name"],
            season=row["season"],
            num_teams=row["num_teams"],
            draft_type=row["draft_type"],
            is_keeper=bool(row["is_keeper"]),
            game_key=row["game_key"],
        )


class SqliteYahooTeamRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, team: YahooTeam) -> int:
        cursor = self._conn.execute(
            "INSERT INTO yahoo_team"
            "    (team_key, league_key, team_id, name, manager_name, is_owned_by_user)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(team_key) DO UPDATE SET"
            "    league_key=excluded.league_key,"
            "    team_id=excluded.team_id,"
            "    name=excluded.name,"
            "    manager_name=excluded.manager_name,"
            "    is_owned_by_user=excluded.is_owned_by_user",
            (
                team.team_key,
                team.league_key,
                team.team_id,
                team.name,
                team.manager_name,
                int(team.is_owned_by_user),
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_league_key(self, league_key: str) -> list[YahooTeam]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE league_key = ?",
            (league_key,),
        ).fetchall()
        return [self._row_to_team(row) for row in rows]

    def get_user_team(self, league_key: str) -> YahooTeam | None:
        row = self._conn.execute(
            self._select_sql() + " WHERE league_key = ? AND is_owned_by_user = 1",
            (league_key,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_team(row)

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, team_key, league_key, team_id, name, manager_name, is_owned_by_user FROM yahoo_team"

    @staticmethod
    def _row_to_team(row: sqlite3.Row) -> YahooTeam:
        return YahooTeam(
            id=row["id"],
            team_key=row["team_key"],
            league_key=row["league_key"],
            team_id=row["team_id"],
            name=row["name"],
            manager_name=row["manager_name"],
            is_owned_by_user=bool(row["is_owned_by_user"]),
        )
