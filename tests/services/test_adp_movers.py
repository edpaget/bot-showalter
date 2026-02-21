import sqlite3

import pytest

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.services.adp_movers import ADPMoversService
from tests.helpers import seed_player


def _seed_adp(
    conn: sqlite3.Connection,
    player_id: int,
    as_of: str,
    overall_pick: float = 10.0,
    rank: int = 10,
    positions: str = "OF",
    season: int = 2026,
    provider: str = "fantasypros",
) -> None:
    repo = SqliteADPRepo(conn)
    repo.upsert(
        ADP(
            player_id=player_id,
            season=season,
            provider=provider,
            overall_pick=overall_pick,
            rank=rank,
            positions=positions,
            as_of=as_of,
        )
    )


def _make_service(conn: sqlite3.Connection) -> ADPMoversService:
    return ADPMoversService(SqliteADPRepo(conn), SqlitePlayerRepo(conn))


class TestBasicMovers:
    def test_riser_detected(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Rising", name_last="Star")
        _seed_adp(conn, pid, as_of="2026-02-01", rank=50, overall_pick=50.0)
        _seed_adp(conn, pid, as_of="2026-02-20", rank=20, overall_pick=20.0)

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.risers) == 1
        riser = report.risers[0]
        assert riser.player_name == "Rising Star"
        assert riser.current_rank == 20
        assert riser.previous_rank == 50
        assert riser.rank_delta == 30
        assert riser.direction == "riser"

    def test_faller_detected(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Falling", name_last="Player")
        _seed_adp(conn, pid, as_of="2026-02-01", rank=10, overall_pick=10.0)
        _seed_adp(conn, pid, as_of="2026-02-20", rank=40, overall_pick=40.0)

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.fallers) == 1
        faller = report.fallers[0]
        assert faller.current_rank == 40
        assert faller.previous_rank == 10
        assert faller.rank_delta == -30
        assert faller.direction == "faller"

    def test_risers_sorted_descending(self, conn: sqlite3.Connection) -> None:
        pids = [seed_player(conn, name_first=f"R{i}", name_last="Player") for i in range(3)]
        deltas = [10, 30, 20]  # expected sort: 30, 20, 10
        for pid, d in zip(pids, deltas):
            _seed_adp(conn, pid, as_of="2026-02-01", rank=50, overall_pick=50.0)
            _seed_adp(conn, pid, as_of="2026-02-20", rank=50 - d, overall_pick=float(50 - d))

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.risers) == 3
        assert report.risers[0].rank_delta == 30
        assert report.risers[1].rank_delta == 20
        assert report.risers[2].rank_delta == 10

    def test_fallers_sorted_ascending(self, conn: sqlite3.Connection) -> None:
        pids = [seed_player(conn, name_first=f"F{i}", name_last="Player") for i in range(3)]
        deltas = [-10, -30, -20]  # expected sort: -30, -20, -10
        for pid, d in zip(pids, deltas):
            _seed_adp(conn, pid, as_of="2026-02-01", rank=20, overall_pick=20.0)
            _seed_adp(conn, pid, as_of="2026-02-20", rank=20 - d, overall_pick=float(20 - d))

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.fallers) == 3
        assert report.fallers[0].rank_delta == -30
        assert report.fallers[1].rank_delta == -20
        assert report.fallers[2].rank_delta == -10

    def test_top_limits_applied(self, conn: sqlite3.Connection) -> None:
        for i in range(25):
            pid = seed_player(conn, name_first=f"Riser{i}", name_last="P")
            _seed_adp(conn, pid, as_of="2026-02-01", rank=100 + i, overall_pick=float(100 + i))
            _seed_adp(conn, pid, as_of="2026-02-20", rank=50 + i, overall_pick=float(50 + i))

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01", top=20)

        assert len(report.risers) == 20

    def test_zero_delta_excluded(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Steady", name_last="Player")
        _seed_adp(conn, pid, as_of="2026-02-01", rank=25, overall_pick=25.0)
        _seed_adp(conn, pid, as_of="2026-02-20", rank=25, overall_pick=25.0)

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.risers) == 0
        assert len(report.fallers) == 0


class TestNewAndDropped:
    def test_new_entry(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="New", name_last="Guy")
        _seed_adp(conn, pid, as_of="2026-02-20", rank=30, overall_pick=30.0)

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.new_entries) == 1
        entry = report.new_entries[0]
        assert entry.direction == "new"
        assert entry.current_rank == 30
        assert entry.previous_rank == 0
        assert entry.rank_delta == 0

    def test_dropped_entry(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Gone", name_last="Player")
        _seed_adp(conn, pid, as_of="2026-02-01", rank=15, overall_pick=15.0)

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.dropped_entries) == 1
        entry = report.dropped_entries[0]
        assert entry.direction == "dropped"
        assert entry.current_rank == 0
        assert entry.previous_rank == 15
        assert entry.rank_delta == 0

    def test_new_sorted_by_rank(self, conn: sqlite3.Connection) -> None:
        pids = [seed_player(conn, name_first=f"N{i}", name_last="P") for i in range(3)]
        ranks = [50, 10, 30]
        for pid, r in zip(pids, ranks):
            _seed_adp(conn, pid, as_of="2026-02-20", rank=r, overall_pick=float(r))

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.new_entries) == 3
        assert report.new_entries[0].current_rank == 10
        assert report.new_entries[1].current_rank == 30
        assert report.new_entries[2].current_rank == 50

    def test_dropped_sorted_by_rank(self, conn: sqlite3.Connection) -> None:
        pids = [seed_player(conn, name_first=f"D{i}", name_last="P") for i in range(3)]
        ranks = [50, 10, 30]
        for pid, r in zip(pids, ranks):
            _seed_adp(conn, pid, as_of="2026-02-01", rank=r, overall_pick=float(r))

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.dropped_entries) == 3
        assert report.dropped_entries[0].previous_rank == 10
        assert report.dropped_entries[1].previous_rank == 30
        assert report.dropped_entries[2].previous_rank == 50


class TestWindowResolution:
    def test_resolve_window_picks_closest(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn)
        _seed_adp(conn, pid, as_of="2026-01-15", rank=1, overall_pick=1.0)
        _seed_adp(conn, pid, as_of="2026-02-01", rank=1, overall_pick=1.0, positions="DH")
        _seed_adp(conn, pid, as_of="2026-02-20", rank=1, overall_pick=1.0, positions="SS")

        svc = _make_service(conn)
        current, previous = svc.resolve_window(2026, "fantasypros", window_days=14)

        assert current == "2026-02-20"
        assert previous == "2026-02-01"

    def test_resolve_window_single_snapshot_raises(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn)
        _seed_adp(conn, pid, as_of="2026-02-20", rank=1, overall_pick=1.0)

        svc = _make_service(conn)
        with pytest.raises(ValueError, match="at least 2 snapshots"):
            svc.resolve_window(2026, "fantasypros")

    def test_resolve_window_no_snapshots_raises(self, conn: sqlite3.Connection) -> None:
        seed_player(conn)
        svc = _make_service(conn)
        with pytest.raises(ValueError, match="at least 2 snapshots"):
            svc.resolve_window(2026, "fantasypros")


class TestEdgeCases:
    def test_two_way_player_uses_best_pick(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        # Two entries per snapshot (batter + pitcher positions)
        _seed_adp(conn, pid, as_of="2026-02-01", rank=3, overall_pick=3.0, positions="DH")
        _seed_adp(conn, pid, as_of="2026-02-01", rank=40, overall_pick=40.0, positions="SP")
        _seed_adp(conn, pid, as_of="2026-02-20", rank=1, overall_pick=1.0, positions="DH")
        _seed_adp(conn, pid, as_of="2026-02-20", rank=35, overall_pick=35.0, positions="SP")

        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert len(report.risers) == 1
        riser = report.risers[0]
        assert riser.current_rank == 1
        assert riser.previous_rank == 3
        assert riser.rank_delta == 2

    def test_empty_snapshots_returns_empty_report(self, conn: sqlite3.Connection) -> None:
        seed_player(conn)
        svc = _make_service(conn)
        report = svc.compute_adp_movers(2026, "fantasypros", "2026-02-20", "2026-02-01")

        assert report.risers == []
        assert report.fallers == []
        assert report.new_entries == []
        assert report.dropped_entries == []
