import datetime
import sqlite3

from fantasy_baseball_manager.domain.roster import Roster, RosterEntry


class SqliteYahooRosterRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_snapshot(self, roster: Roster) -> int:
        as_of_str = roster.as_of.isoformat()

        # Upsert snapshot header
        cursor = self._conn.execute(
            "INSERT INTO yahoo_roster_snapshot"
            "    (team_key, league_key, season, week, as_of)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(team_key, league_key, season, week, as_of) DO UPDATE SET"
            "    season=excluded.season",
            (roster.team_key, roster.league_key, roster.season, roster.week, as_of_str),
        )
        snapshot_id = cursor.lastrowid
        assert snapshot_id is not None

        # Delete existing entries and re-insert
        self._conn.execute(
            "DELETE FROM yahoo_roster_entry WHERE snapshot_id = ?",
            (snapshot_id,),
        )

        for entry in roster.entries:
            self._conn.execute(
                "INSERT INTO yahoo_roster_entry"
                "    (snapshot_id, player_id, yahoo_player_key, player_name,"
                "     position, roster_status, acquisition_type)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    entry.player_id,
                    entry.yahoo_player_key,
                    entry.player_name,
                    entry.position,
                    entry.roster_status,
                    entry.acquisition_type,
                ),
            )

        return snapshot_id

    def get_latest_by_team(self, team_key: str, league_key: str) -> Roster | None:
        row = self._conn.execute(
            "SELECT id, team_key, league_key, season, week, as_of"
            " FROM yahoo_roster_snapshot"
            " WHERE team_key = ? AND league_key = ?"
            " ORDER BY as_of DESC, week DESC LIMIT 1",
            (team_key, league_key),
        ).fetchone()
        if row is None:
            return None
        return self._load_roster(row)

    def get_by_league_latest(self, league_key: str) -> list[Roster]:
        rows = self._conn.execute(
            "SELECT s.id, s.team_key, s.league_key, s.season, s.week, s.as_of"
            " FROM yahoo_roster_snapshot s"
            " INNER JOIN ("
            "   SELECT team_key, MAX(as_of) AS max_as_of"
            "   FROM yahoo_roster_snapshot"
            "   WHERE league_key = ?"
            "   GROUP BY team_key"
            " ) latest ON s.team_key = latest.team_key AND s.as_of = latest.max_as_of"
            " WHERE s.league_key = ?",
            (league_key, league_key),
        ).fetchall()
        return [self._load_roster(row) for row in rows]

    def _load_roster(self, row: sqlite3.Row) -> Roster:
        snapshot_id = row["id"]
        entry_rows = self._conn.execute(
            "SELECT player_id, yahoo_player_key, player_name,"
            " position, roster_status, acquisition_type"
            " FROM yahoo_roster_entry WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchall()

        entries = tuple(
            RosterEntry(
                player_id=e["player_id"],
                yahoo_player_key=e["yahoo_player_key"],
                player_name=e["player_name"],
                position=e["position"],
                roster_status=e["roster_status"],
                acquisition_type=e["acquisition_type"],
            )
            for e in entry_rows
        )

        return Roster(
            id=row["id"],
            team_key=row["team_key"],
            league_key=row["league_key"],
            season=row["season"],
            week=row["week"],
            as_of=datetime.date.fromisoformat(row["as_of"]),
            entries=entries,
        )
