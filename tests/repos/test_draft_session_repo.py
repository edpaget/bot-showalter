from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.domain import DraftSessionPick, DraftSessionRecord
from fantasy_baseball_manager.repos.draft_session_repo import SqliteDraftSessionRepo

if TYPE_CHECKING:
    import sqlite3


class _FakeProvider:
    """Minimal ConnectionProvider backed by an existing connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def connection(self):  # noqa: ANN201
        @contextmanager
        def _ctx():  # noqa: ANN202
            yield self._conn

        return _ctx()


def _make_record(**overrides: object) -> DraftSessionRecord:
    defaults: dict[str, object] = {
        "league": "test-league",
        "season": 2026,
        "teams": 12,
        "format": "snake",
        "user_team": 1,
        "roster_slots": {"C": 1, "1B": 1, "OF": 3},
        "budget": 0,
        "status": "in_progress",
        "created_at": "2026-03-07T10:00:00",
        "updated_at": "2026-03-07T10:00:00",
    }
    defaults.update(overrides)
    return DraftSessionRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repo(conn: sqlite3.Connection) -> SqliteDraftSessionRepo:
    return SqliteDraftSessionRepo(_FakeProvider(conn))


class TestCreateSessionAndLoad:
    def test_roundtrip(self, repo: SqliteDraftSessionRepo) -> None:
        record = _make_record()
        session_id = repo.create_session(record)
        assert session_id >= 1

        loaded = repo.load_session(session_id)
        assert loaded is not None
        assert loaded.id == session_id
        assert loaded.league == "test-league"
        assert loaded.season == 2026
        assert loaded.teams == 12
        assert loaded.format == "snake"
        assert loaded.user_team == 1
        assert loaded.roster_slots == {"C": 1, "1B": 1, "OF": 3}
        assert loaded.budget == 0
        assert loaded.status == "in_progress"


class TestSavePickAndLoadPicks:
    def test_save_and_load_multiple(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record())

        picks = [
            DraftSessionPick(
                session_id=session_id, pick_number=1, team=1, player_id=100, player_name="Player A", position="OF"
            ),
            DraftSessionPick(
                session_id=session_id,
                pick_number=2,
                team=2,
                player_id=101,
                player_name="Player B",
                position="1B",
                price=25,
            ),
            DraftSessionPick(
                session_id=session_id, pick_number=3, team=1, player_id=102, player_name="Player C", position="C"
            ),
        ]
        for p in picks:
            repo.save_pick(p)

        loaded = repo.load_picks(session_id)
        assert len(loaded) == 3
        assert loaded[0].pick_number == 1
        assert loaded[1].pick_number == 2
        assert loaded[2].pick_number == 3
        assert loaded[1].price == 25
        assert loaded[0].price is None


class TestDeletePick:
    def test_delete_middle_pick(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record())

        for i in range(1, 4):
            repo.save_pick(
                DraftSessionPick(
                    session_id=session_id,
                    pick_number=i,
                    team=1,
                    player_id=100 + i,
                    player_name=f"Player {i}",
                    position="OF",
                )
            )

        repo.delete_pick(session_id, pick_number=2)
        remaining = repo.load_picks(session_id)
        assert len(remaining) == 2
        assert [p.pick_number for p in remaining] == [1, 3]


class TestListSessionsFilters:
    def test_filter_by_league_and_season(self, repo: SqliteDraftSessionRepo) -> None:
        repo.create_session(_make_record(league="AL", season=2025, created_at="2025-01-01T00:00:00"))
        repo.create_session(_make_record(league="NL", season=2025, created_at="2025-01-02T00:00:00"))
        repo.create_session(_make_record(league="AL", season=2026, created_at="2026-01-01T00:00:00"))

        al_sessions = repo.list_sessions(league="AL")
        assert len(al_sessions) == 2

        s2025 = repo.list_sessions(season=2025)
        assert len(s2025) == 2

        al_2025 = repo.list_sessions(league="AL", season=2025)
        assert len(al_2025) == 1
        assert al_2025[0].league == "AL"
        assert al_2025[0].season == 2025

    def test_list_all(self, repo: SqliteDraftSessionRepo) -> None:
        repo.create_session(_make_record(created_at="2026-01-01T00:00:00"))
        repo.create_session(_make_record(created_at="2026-01-02T00:00:00"))
        all_sessions = repo.list_sessions()
        assert len(all_sessions) == 2
        # Ordered by created_at DESC
        assert all_sessions[0].created_at == "2026-01-02T00:00:00"


class TestUpdateStatus:
    def test_update_status(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record())
        repo.update_status(session_id, "complete")

        loaded = repo.load_session(session_id)
        assert loaded is not None
        assert loaded.status == "complete"


class TestUpdateTimestamp:
    def test_update_timestamp(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record(updated_at="2026-03-07T10:00:00"))
        repo.update_timestamp(session_id, "2026-03-07T12:00:00")

        loaded = repo.load_session(session_id)
        assert loaded is not None
        assert loaded.updated_at == "2026-03-07T12:00:00"


class TestDeleteSession:
    def test_delete_session_removes_session_and_picks(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record())
        for i in range(1, 4):
            repo.save_pick(
                DraftSessionPick(
                    session_id=session_id,
                    pick_number=i,
                    team=1,
                    player_id=100 + i,
                    player_name=f"Player {i}",
                    position="OF",
                )
            )

        repo.delete_session(session_id)
        assert repo.load_session(session_id) is None
        assert repo.load_picks(session_id) == []


class TestCountPicks:
    def test_count_picks(self, repo: SqliteDraftSessionRepo) -> None:
        session_id = repo.create_session(_make_record())
        assert repo.count_picks(session_id) == 0

        for i in range(1, 4):
            repo.save_pick(
                DraftSessionPick(
                    session_id=session_id,
                    pick_number=i,
                    team=1,
                    player_id=100 + i,
                    player_name=f"Player {i}",
                    position="OF",
                )
            )
        assert repo.count_picks(session_id) == 3


class TestLoadSessionNotFound:
    def test_returns_none(self, repo: SqliteDraftSessionRepo) -> None:
        assert repo.load_session(9999) is None
