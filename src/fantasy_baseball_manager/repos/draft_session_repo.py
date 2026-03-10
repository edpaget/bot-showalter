import json
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import DraftSessionPick, DraftSessionRecord

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteDraftSessionRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def create_session(self, record: DraftSessionRecord) -> int:
        with self._provider.connection() as conn:
            cur = conn.execute(
                "INSERT INTO draft_session"
                " (league, season, teams, format, user_team, roster_slots,"
                "  budget, status, created_at, updated_at, system, version, keeper_player_ids)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.league,
                    record.season,
                    record.teams,
                    record.format,
                    record.user_team,
                    json.dumps(record.roster_slots),
                    record.budget,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.system,
                    record.version,
                    json.dumps(sorted(record.keeper_player_ids)) if record.keeper_player_ids is not None else None,
                ),
            )
            assert cur.lastrowid is not None  # noqa: S101
            return cur.lastrowid

    def save_pick(self, pick: DraftSessionPick) -> None:
        with self._provider.connection() as conn:
            conn.execute(
                "INSERT INTO draft_session_pick"
                " (session_id, pick_number, team, player_id, player_name, position, price)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    pick.session_id,
                    pick.pick_number,
                    pick.team,
                    pick.player_id,
                    pick.player_name,
                    pick.position,
                    pick.price,
                ),
            )

    def delete_pick(self, session_id: int, pick_number: int) -> None:
        with self._provider.connection() as conn:
            conn.execute(
                "DELETE FROM draft_session_pick WHERE session_id = ? AND pick_number = ?",
                (session_id, pick_number),
            )

    def load_session(self, session_id: int) -> DraftSessionRecord | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT id, league, season, teams, format, user_team, roster_slots,"
                " budget, status, created_at, updated_at, system, version, keeper_player_ids"
                " FROM draft_session WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    def load_picks(self, session_id: int) -> list[DraftSessionPick]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT id, session_id, pick_number, team, player_id, player_name, position, price"
                " FROM draft_session_pick WHERE session_id = ? ORDER BY pick_number",
                (session_id,),
            ).fetchall()
            return [self._row_to_pick(row) for row in rows]

    def list_sessions(self, *, league: str | None = None, season: int | None = None) -> list[DraftSessionRecord]:
        clauses: list[str] = []
        params: list[str | int] = []
        if league is not None:
            clauses.append("league = ?")
            params.append(league)
        if season is not None:
            clauses.append("season = ?")
            params.append(season)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT id, league, season, teams, format, user_team, roster_slots,"  # noqa: S608
                f" budget, status, created_at, updated_at, system, version, keeper_player_ids FROM draft_session{where}"
                " ORDER BY created_at DESC",
                params,
            ).fetchall()
            return [self._row_to_session(row) for row in rows]

    def update_status(self, session_id: int, status: str) -> None:
        with self._provider.connection() as conn:
            conn.execute(
                "UPDATE draft_session SET status = ? WHERE id = ?",
                (status, session_id),
            )

    def update_timestamp(self, session_id: int, updated_at: str) -> None:
        with self._provider.connection() as conn:
            conn.execute(
                "UPDATE draft_session SET updated_at = ? WHERE id = ?",
                (updated_at, session_id),
            )

    def delete_session(self, session_id: int) -> None:
        with self._provider.connection() as conn:
            conn.execute("DELETE FROM draft_session_pick WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM draft_session WHERE id = ?", (session_id,))

    def count_picks(self, session_id: int) -> int:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM draft_session_pick WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row[0]

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> DraftSessionRecord:
        raw_keeper_ids = row["keeper_player_ids"]
        keeper_player_ids: list[int] | None = json.loads(raw_keeper_ids) if raw_keeper_ids is not None else None
        return DraftSessionRecord(
            id=row["id"],
            league=row["league"],
            season=row["season"],
            teams=row["teams"],
            format=row["format"],
            user_team=row["user_team"],
            roster_slots=json.loads(row["roster_slots"]),
            budget=row["budget"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            system=row["system"],
            version=row["version"],
            keeper_player_ids=keeper_player_ids,
        )

    @staticmethod
    def _row_to_pick(row: sqlite3.Row) -> DraftSessionPick:
        return DraftSessionPick(
            id=row["id"],
            session_id=row["session_id"],
            pick_number=row["pick_number"],
            team=row["team"],
            player_id=row["player_id"],
            player_name=row["player_name"],
            position=row["position"],
            price=row["price"],
        )
