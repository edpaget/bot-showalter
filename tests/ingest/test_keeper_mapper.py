from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.keeper_mapper import import_keeper_costs
from fantasy_baseball_manager.repos.keeper_repo import SqliteKeeperCostRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo


def _round_translator(round_num: int) -> float:
    return 100.0 / round_num


def _make_players(conn: object) -> list[Player]:
    repo = SqlitePlayerRepo(SingleConnectionProvider(conn))  # type: ignore[arg-type]
    pid1 = repo.upsert(Player(name_first="Mike", name_last="Trout"))
    pid2 = repo.upsert(Player(name_first="Shohei", name_last="Ohtani"))
    conn.commit()  # type: ignore[union-attr]
    players = repo.all()
    assert len(players) == 2
    assert {p.id for p in players} == {pid1, pid2}
    return players


class TestImportKeeperCosts:
    def test_success(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [
            {"Player": "Mike Trout", "Cost": "25"},
            {"Player": "Shohei Ohtani", "Cost": "15"},
        ]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 2
        assert result.skipped == 0
        assert result.unmatched == []

        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        costs = {s.cost for s in stored}
        assert costs == {25.0, 15.0}
        conn.close()

    def test_unmatched_tracking(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [
            {"Player": "Mike Trout", "Cost": "25"},
            {"Player": "Nobody McFakerson", "Cost": "5"},
        ]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        assert result.unmatched == ["Nobody McFakerson"]
        conn.close()

    def test_optional_years_and_source(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [
            {"Player": "Mike Trout", "Cost": "25", "Years": "3", "Source": "contract"},
        ]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].years_remaining == 3
        assert stored[0].source == "contract"
        conn.close()

    def test_name_normalization(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        # "Michael Trout" should match "Mike Trout" via nickname alias
        rows = [{"Player": "Michael Trout", "Cost": "25"}]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        assert result.unmatched == []
        conn.close()

    def test_name_column_alias(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [{"Name": "Mike Trout", "Cost": "25"}]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        conn.close()

    def test_dollar_sign_in_cost(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [{"Player": "Mike Trout", "Cost": "$25"}]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].cost == 25.0
        conn.close()

    def test_empty_name_skipped(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [{"Player": "", "Cost": "10"}]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")

        assert result.loaded == 0
        assert result.skipped == 1
        conn.close()

    def test_round_import_with_translator(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [
            {"Player": "Mike Trout", "Round": "2"},
            {"Player": "Shohei Ohtani", "Round": "5"},
        ]
        result = import_keeper_costs(
            rows, repo, players, season=2026, league="dynasty", cost_translator=_round_translator
        )
        conn.commit()

        assert result.loaded == 2
        assert result.skipped == 0

        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        by_pid = {s.player_id: s for s in stored}
        trout = by_pid[players[0].id]
        ohtani = by_pid[players[1].id]
        assert trout.cost == 50.0  # 100/2
        assert trout.source == "draft_round"
        assert trout.original_round == 2
        assert ohtani.cost == 20.0  # 100/5
        assert ohtani.original_round == 5
        conn.close()

    def test_round_import_empty_round_skips(self) -> None:
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [{"Player": "Mike Trout", "Round": ""}]
        result = import_keeper_costs(
            rows, repo, players, season=2026, league="dynasty", cost_translator=_round_translator
        )

        assert result.loaded == 0
        assert result.skipped == 1
        conn.close()

    def test_dollar_import_unchanged(self) -> None:
        """Existing dollar import path works without a translator (regression guard)."""
        conn = create_connection(":memory:")
        players = _make_players(conn)
        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))

        rows = [{"Player": "Mike Trout", "Cost": "25"}]
        result = import_keeper_costs(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].cost == 25.0
        assert stored[0].source == "auction"
        assert stored[0].original_round is None
        conn.close()
