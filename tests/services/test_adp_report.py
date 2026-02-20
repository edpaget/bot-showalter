import sqlite3

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.services.adp_report import ADPReportService
from tests.helpers import seed_player


def _seed_valuation(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2026,
    system: str = "zar",
    version: str = "1.0",
    player_type: str = "batter",
    position: str = "OF",
    value: float = 25.0,
    rank: int = 1,
) -> None:
    repo = SqliteValuationRepo(conn)
    repo.upsert(
        Valuation(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2026.1",
            player_type=player_type,
            position=position,
            value=value,
            rank=rank,
            category_scores={"hr": 1.0},
        )
    )


def _seed_adp(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2026,
    provider: str = "fantasypros",
    overall_pick: float = 10.0,
    rank: int = 10,
    positions: str = "OF",
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
        )
    )


def _make_service(conn: sqlite3.Connection) -> ADPReportService:
    return ADPReportService(
        SqlitePlayerRepo(conn),
        SqliteValuationRepo(conn),
        SqliteADPRepo(conn),
    )


class TestBasicBuyAndAvoid:
    def test_buy_target_positive_delta(self, conn: sqlite3.Connection) -> None:
        """Player ranked higher by ZAR (rank 5) than ADP (rank 50) => buy target."""
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, rank=5, value=40.0, position="OF")
        _seed_adp(conn, pid, overall_pick=50.0, rank=50, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 1
        assert len(report.buy_targets) == 1
        assert len(report.avoid_list) == 0

        target = report.buy_targets[0]
        assert target.player_name == "Juan Soto"
        assert target.zar_rank == 5
        assert target.adp_rank == 50
        assert target.zar_value == 40.0
        assert target.adp_pick == 50.0
        assert target.player_type == "batter"
        assert target.position == "OF"
        assert target.adp_positions == "OF"
        assert target.provider == "fantasypros"

    def test_avoid_negative_delta(self, conn: sqlite3.Connection) -> None:
        """Player ranked lower by ZAR (rank 50) than ADP (rank 5) => avoid."""
        pid = seed_player(conn, name_first="Overvalued", name_last="Guy", mlbam_id=100001)
        _seed_valuation(conn, pid, rank=50, value=5.0, position="1B")
        _seed_adp(conn, pid, overall_pick=5.0, rank=5, positions="1B")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 1
        assert len(report.buy_targets) == 0
        assert len(report.avoid_list) == 1

        avoid = report.avoid_list[0]
        assert avoid.player_name == "Overvalued Guy"
        assert avoid.zar_rank == 50
        assert avoid.rank_delta < 0

    def test_buy_and_avoid_sorted(self, conn: sqlite3.Connection) -> None:
        """Multiple players: buy targets sorted desc by delta, avoid sorted asc."""
        pid1 = seed_player(conn, name_first="Big", name_last="Buy", mlbam_id=100001)
        pid2 = seed_player(conn, name_first="Small", name_last="Buy", mlbam_id=100002)
        pid3 = seed_player(conn, name_first="Big", name_last="Avoid", mlbam_id=100003)
        pid4 = seed_player(conn, name_first="Small", name_last="Avoid", mlbam_id=100004)

        # Buy targets: ZAR ranks them higher than ADP
        _seed_valuation(conn, pid1, rank=1, value=50.0)
        _seed_adp(conn, pid1, overall_pick=100.0, rank=100, positions="OF")

        _seed_valuation(conn, pid2, rank=2, value=45.0)
        _seed_adp(conn, pid2, overall_pick=50.0, rank=50, positions="OF")

        # Avoid: ADP ranks them higher than ZAR
        _seed_valuation(conn, pid3, rank=200, value=1.0)
        _seed_adp(conn, pid3, overall_pick=3.0, rank=3, positions="OF")

        _seed_valuation(conn, pid4, rank=100, value=5.0)
        _seed_adp(conn, pid4, overall_pick=10.0, rank=10, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 4
        assert len(report.buy_targets) == 2
        assert len(report.avoid_list) == 2

        # Buy targets sorted by rank_delta desc (biggest positive first)
        assert report.buy_targets[0].rank_delta >= report.buy_targets[1].rank_delta

        # Avoid list sorted by rank_delta asc (most negative first)
        assert report.avoid_list[0].rank_delta <= report.avoid_list[1].rank_delta


class TestUnrankedValuable:
    def test_player_with_valuation_but_no_adp_is_sleeper(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Hidden", name_last="Gem", mlbam_id=100001)
        _seed_valuation(conn, pid, rank=50, value=20.0)

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 0
        assert len(report.unranked_valuable) == 1
        assert report.unranked_valuable[0].player_name == "Hidden Gem"
        assert report.unranked_valuable[0].zar_rank == 50

    def test_unranked_outside_top300_excluded(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Marginal", name_last="Player", mlbam_id=100001)
        _seed_valuation(conn, pid, rank=301, value=0.5)

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert len(report.unranked_valuable) == 0


class TestFilterByPlayerType:
    def test_only_batters(self, conn: sqlite3.Connection) -> None:
        pid_bat = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid_pit = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        _seed_valuation(conn, pid_bat, rank=1, value=40.0, player_type="batter")
        _seed_valuation(conn, pid_pit, rank=2, value=30.0, player_type="pitcher", position="SP")
        _seed_adp(conn, pid_bat, overall_pick=5.0, rank=5, positions="OF")
        _seed_adp(conn, pid_pit, overall_pick=10.0, rank=10, positions="SP")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", player_type="batter")

        assert report.n_matched == 1
        assert report.buy_targets[0].player_type == "batter" or report.avoid_list[0].player_type == "batter"
        all_entries = report.buy_targets + report.avoid_list + report.unranked_valuable
        for entry in all_entries:
            assert entry.player_type == "batter"

    def test_only_pitchers(self, conn: sqlite3.Connection) -> None:
        pid_bat = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid_pit = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        _seed_valuation(conn, pid_bat, rank=1, value=40.0, player_type="batter")
        _seed_valuation(conn, pid_pit, rank=2, value=30.0, player_type="pitcher", position="SP")
        _seed_adp(conn, pid_bat, overall_pick=5.0, rank=5, positions="OF")
        _seed_adp(conn, pid_pit, overall_pick=10.0, rank=10, positions="SP")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", player_type="pitcher")

        assert report.n_matched == 1
        all_entries = report.buy_targets + report.avoid_list + report.unranked_valuable
        for entry in all_entries:
            assert entry.player_type == "pitcher"


class TestFilterByPosition:
    def test_only_of(self, conn: sqlite3.Connection) -> None:
        pid_of = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid_1b = seed_player(conn, name_first="Freddie", name_last="Freeman", mlbam_id=518692)
        _seed_valuation(conn, pid_of, rank=1, value=40.0, position="OF")
        _seed_valuation(conn, pid_1b, rank=2, value=35.0, position="1B")
        _seed_adp(conn, pid_of, overall_pick=5.0, rank=5, positions="OF")
        _seed_adp(conn, pid_1b, overall_pick=10.0, rank=10, positions="1B")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", position="OF")

        assert report.n_matched == 1
        all_entries = report.buy_targets + report.avoid_list + report.unranked_valuable
        for entry in all_entries:
            assert entry.position == "OF"


class TestTopLimitsSections:
    def test_top_limits_each_section(self, conn: sqlite3.Connection) -> None:
        # Create 10 buy targets
        for i in range(10):
            pid = seed_player(conn, name_first=f"Buy{i}", name_last="Target", mlbam_id=100000 + i)
            _seed_valuation(conn, pid, rank=i + 1, value=float(50 - i))
            _seed_adp(conn, pid, overall_pick=float(200 + i), rank=200 + i, positions="OF")

        # Create 10 avoid players
        for i in range(10):
            pid = seed_player(conn, name_first=f"Avoid{i}", name_last="Player", mlbam_id=200000 + i)
            _seed_valuation(conn, pid, rank=200 + i, value=float(5 - i * 0.1))
            _seed_adp(conn, pid, overall_pick=float(i + 1), rank=i + 1, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", top=5)

        assert len(report.buy_targets) == 5
        assert len(report.avoid_list) == 5


class TestEmptyInputs:
    def test_no_valuations(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_adp(conn, pid, overall_pick=5.0, rank=5, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 0
        assert report.buy_targets == []
        assert report.avoid_list == []
        assert report.unranked_valuable == []

    def test_no_adp(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, rank=1, value=40.0)

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 0
        # Player is in top 300 so should be sleeper
        assert len(report.unranked_valuable) == 1


class TestTwoWayPlayer:
    def test_picks_lowest_rank_adp(self, conn: sqlite3.Connection) -> None:
        """Ohtani has batter and pitcher ADP entries; use lowest overall_pick."""
        pid = seed_player(conn, name_first="Shohei", name_last="Ohtani", mlbam_id=660271)
        _seed_valuation(conn, pid, rank=5, value=50.0, player_type="batter")
        # Batter ADP: pick 2, rank 2 (better)
        _seed_adp(conn, pid, overall_pick=2.0, rank=2, positions="DH")
        # Pitcher ADP: pick 25, rank 25
        _seed_adp(conn, pid, overall_pick=25.0, rank=25, positions="SP")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 1
        # Should use the lowest overall_pick (2.0) entry with rank 2
        all_entries = report.buy_targets + report.avoid_list
        assert len(all_entries) == 1
        assert all_entries[0].adp_pick == 2.0
        assert all_entries[0].adp_rank == 2

    def test_two_way_player_type_filter_picks_matching(self, conn: sqlite3.Connection) -> None:
        """When filtering by player_type, pick the ADP entry matching that type."""
        pid = seed_player(conn, name_first="Shohei", name_last="Ohtani", mlbam_id=660271)
        _seed_valuation(conn, pid, rank=3, value=35.0, player_type="pitcher", position="SP")
        # Batter ADP: pick 1
        _seed_adp(conn, pid, overall_pick=1.0, rank=1, positions="DH")
        # Pitcher ADP: pick 25
        _seed_adp(conn, pid, overall_pick=25.0, rank=25, positions="SP")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", player_type="pitcher")

        assert report.n_matched == 1
        all_entries = report.buy_targets + report.avoid_list
        assert len(all_entries) == 1
        # Should use the pitcher ADP entry (SP positions)
        assert all_entries[0].adp_pick == 25.0
        assert all_entries[0].adp_positions == "SP"


class TestVersionFilter:
    def test_only_matching_version_included(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, rank=1, value=40.0, version="1.0")
        _seed_valuation(conn, pid, rank=5, value=35.0, version="2.0")
        _seed_adp(conn, pid, overall_pick=5.0, rank=5, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0")

        assert report.n_matched == 1
        all_entries = report.buy_targets + report.avoid_list
        assert len(all_entries) == 1
        assert all_entries[0].zar_rank == 1
        assert all_entries[0].zar_value == 40.0


class TestReportMetadata:
    def test_report_metadata(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, rank=1, value=40.0)
        _seed_adp(conn, pid, overall_pick=5.0, rank=5, positions="OF")

        svc = _make_service(conn)
        report = svc.compute_value_over_adp(2026, "zar", "1.0", provider="fantasypros")

        assert report.season == 2026
        assert report.system == "zar"
        assert report.version == "1.0"
        assert report.provider == "fantasypros"
