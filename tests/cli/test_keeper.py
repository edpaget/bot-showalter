import csv
import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import KeeperContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    ADP,
    CategoryConfig,
    Direction,
    KeeperCost,
    LeagueFormat,
    LeagueSettings,
    Projection,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.keeper_repo import SqliteKeeperCostRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.services import PlayerEligibilityService
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from pathlib import Path

    import pytest

runner = CliRunner()


def _build_test_keeper_context(conn: sqlite3.Connection) -> Any:
    @contextmanager
    def _ctx(data_dir: str) -> Iterator[KeeperContext]:
        position_repo = SqlitePositionAppearanceRepo(SingleConnectionProvider(conn))
        pitching_stats_repo = SqlitePitchingStatsRepo(SingleConnectionProvider(conn))
        eligibility_service = PlayerEligibilityService(position_repo, pitching_stats_repo=pitching_stats_repo)
        yield KeeperContext(
            conn=conn,
            keeper_repo=SqliteKeeperCostRepo(SingleConnectionProvider(conn)),
            player_repo=SqlitePlayerRepo(SingleConnectionProvider(conn)),
            valuation_repo=SqliteValuationRepo(SingleConnectionProvider(conn)),
            projection_repo=SqliteProjectionRepo(SingleConnectionProvider(conn)),
            eligibility_service=eligibility_service,
            adp_repo=SqliteADPRepo(SingleConnectionProvider(conn)),
        )

    return _ctx


class TestKeeperImport:
    def test_import_success(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")
        seed_player(conn, name_first="Shohei", name_last="Ohtani")

        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text(
            textwrap.dedent("""\
            Player,Cost,Years
            Mike Trout,25,2
            Shohei Ohtani,15,1
        """)
        )

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        assert result.exit_code == 0, result.output
        assert "Loaded 2" in result.output

        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        conn.close()

    def test_import_file_not_found(self) -> None:
        result = runner.invoke(
            app, ["keeper", "import", "/nonexistent/file.csv", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "file not found" in result.output

    def test_import_idempotent(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text(
            textwrap.dedent("""\
            Player,Cost
            Mike Trout,25
        """)
        )

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Import twice
        runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        result = runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        assert result.exit_code == 0, result.output

        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        conn.close()


def _draft_pick_league() -> LeagueSettings:
    """Minimal league for draft-pick import tests: 2 teams, 3 batters, 2 pitchers = 10 picks."""
    return LeagueSettings(
        name="dynasty",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="W", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _seed_adp_and_valuations(conn: sqlite3.Connection, player_ids: list[int]) -> None:
    """Seed ADP + valuations for draft-pick curve building."""
    adp_repo = SqliteADPRepo(SingleConnectionProvider(conn))
    val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
    for i, pid in enumerate(player_ids, 1):
        adp_repo.upsert(
            ADP(player_id=pid, season=2026, provider="yahoo", overall_pick=float(i), rank=i, positions="OF")
        )
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="OF",
                value=round(50.0 / i, 2),
                rank=i,
                category_scores={},
            )
        )
    conn.commit()


class TestKeeperImportDraftPick:
    def test_import_draft_pick_format(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
        pid2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        # Seed enough ADP/val data to build a curve (need at least roster slots * teams picks)
        extra_pids = [seed_player(conn, name_first=f"P{i}", name_last=f"L{i}") for i in range(10)]
        all_pids = [pid1, pid2, *extra_pids]
        _seed_adp_and_valuations(conn, all_pids)

        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text(
            textwrap.dedent("""\
            Player,Round
            Mike Trout,1
            Shohei Ohtani,3
        """)
        )

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.load_league",
            lambda name, path: _draft_pick_league(),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "import",
                str(csv_file),
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--format",
                "draft-pick",
                "--system",
                "zar",
                "--provider",
                "yahoo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Loaded 2" in result.output

        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        for kc in stored:
            assert kc.source == "draft_round"
            assert kc.original_round is not None
            assert kc.cost > 0
        conn.close()

    def test_import_draft_pick_missing_system_errors(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text("Player,Round\nMike Trout,1\n")

        conn = create_connection(":memory:")
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "import",
                str(csv_file),
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--format",
                "draft-pick",
                "--provider",
                "yahoo",
            ],
        )
        assert result.exit_code == 1
        conn.close()

    def test_import_draft_pick_missing_provider_errors(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text("Player,Round\nMike Trout,1\n")

        conn = create_connection(":memory:")
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "import",
                str(csv_file),
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--format",
                "draft-pick",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 1
        conn.close()


class TestKeeperSet:
    def test_set_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Trout", "--cost", "25", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "$25" in result.output

        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        assert stored[0].cost == 25.0
        conn.close()

    def test_set_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Nobody", "--cost", "10", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "no player found" in result.output

    def test_set_ambiguous_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Smith")
        seed_player(conn, name_first="John", name_last="Smith")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Smith", "--cost", "10", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "ambiguous" in result.output
        conn.close()


class TestKeeperSetRound:
    def test_set_round_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        extra_pids = [seed_player(conn, name_first=f"P{i}", name_last=f"L{i}") for i in range(10)]
        _seed_adp_and_valuations(conn, [pid, *extra_pids])

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.load_league",
            lambda name, path: _draft_pick_league(),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "set",
                "Trout",
                "--round",
                "3",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--provider",
                "yahoo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Round 3" in result.output

        repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        assert stored[0].source == "draft_round"
        assert stored[0].original_round == 3
        assert stored[0].cost > 0
        conn.close()

    def test_set_round_and_cost_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "set",
                "Trout",
                "--cost",
                "25",
                "--round",
                "3",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--provider",
                "yahoo",
            ],
        )
        assert result.exit_code == 1
        conn.close()

    def test_set_neither_cost_nor_round_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            ["keeper", "set", "Trout", "--season", "2026", "--league", "dynasty"],
        )
        assert result.exit_code == 1
        conn.close()

    def test_set_round_missing_system_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "set",
                "Trout",
                "--round",
                "3",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--provider",
                "yahoo",
            ],
        )
        assert result.exit_code == 1
        conn.close()


class TestKeeperDecisions:
    def test_decisions_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
        pid2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")

        keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        keeper_repo.upsert_batch(
            [
                KeeperCost(player_id=pid1, season=2026, league="dynasty", cost=10.0, source="auction"),
                KeeperCost(player_id=pid2, season=2026, league="dynasty", cost=30.0, source="auction"),
            ]
        )

        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        val_repo.upsert(
            Valuation(
                player_id=pid1,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="CF",
                value=25.0,
                rank=1,
                category_scores={},
            )
        )
        val_repo.upsert(
            Valuation(
                player_id=pid2,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="DH",
                value=20.0,
                rank=2,
                category_scores={},
            )
        )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "decisions", "--season", "2026", "--league", "dynasty", "--system", "zar"]
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Shohei Ohtani" in result.output
        assert "$15.0" in result.output  # Trout surplus: 25-10=15
        conn.close()

    def test_decisions_no_keepers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "decisions", "--season", "2026", "--league", "dynasty", "--system", "zar"]
        )
        assert result.exit_code == 0, result.output
        assert "No keeper costs found" in result.output

    def test_decisions_version_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When two versions exist, only the specified version's valuation is used."""
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")

        keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        keeper_repo.upsert_batch(
            [KeeperCost(player_id=pid, season=2026, league="dynasty", cost=10.0, source="auction")]
        )

        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        # Production version: value=25
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="CF",
                value=25.0,
                rank=1,
                category_scores={},
            )
        )
        # Experimental version: value=50
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="experimental",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="CF",
                value=50.0,
                rank=1,
                category_scores={},
            )
        )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Default --version is "production" → surplus = 25-10 = $15
        result = runner.invoke(
            app, ["keeper", "decisions", "--season", "2026", "--league", "dynasty", "--system", "zar"]
        )
        assert result.exit_code == 0, result.output
        assert "$15.0" in result.output

        # Explicit --version experimental → surplus = 50-10 = $40
        result = runner.invoke(
            app,
            [
                "keeper",
                "decisions",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--version",
                "experimental",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "$40.0" in result.output
        conn.close()


def _adjusted_league() -> LeagueSettings:
    return LeagueSettings(
        name="dynasty",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=100,
        roster_batters=2,
        roster_pitchers=0,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(),
        positions={"c": 1, "of": 1},
        roster_util=0,
    )


class TestKeeperAdjustedRankings:
    def test_adjusted_rankings_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
        pid2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        pid3 = seed_player(conn, name_first="Aaron", name_last="Judge")

        keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        keeper_repo.upsert_batch(
            [
                KeeperCost(player_id=pid1, season=2026, league="dynasty", cost=5.0, source="auction"),
            ]
        )

        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        for pid, value, rank in [(pid1, 30.0, 1), (pid2, 20.0, 2), (pid3, 10.0, 3)]:
            val_repo.upsert(
                Valuation(
                    player_id=pid,
                    season=2026,
                    system="zar",
                    version="production",
                    projection_system="composite",
                    projection_version="v1",
                    player_type="batter",
                    position="c",
                    value=value,
                    rank=rank,
                    category_scores={},
                )
            )

        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
        for pid, hr in [(pid1, 40), (pid2, 30), (pid3, 20)]:
            proj_repo.upsert(
                Projection(
                    player_id=pid,
                    season=2026,
                    system="composite",
                    version="production",
                    player_type="batter",
                    stat_json={"pa": 600, "hr": hr},
                )
            )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.load_league",
            lambda name, path: _adjusted_league(),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "adjusted-rankings",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Shohei Ohtani" in result.output
        assert "Aaron Judge" in result.output
        # Trout is kept — should not appear in adjusted rankings
        assert "Mike Trout" not in result.output
        conn.close()

    def test_adjusted_rankings_no_keepers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.load_league",
            lambda name, path: _adjusted_league(),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "adjusted-rankings",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No keeper costs found" in result.output
        conn.close()
        conn.close()


class TestKeeperTradeEval:
    def test_trade_eval_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
        pid2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")

        keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
        keeper_repo.upsert_batch(
            [
                KeeperCost(player_id=pid1, season=2026, league="dynasty", cost=10.0, source="auction"),
                KeeperCost(player_id=pid2, season=2026, league="dynasty", cost=5.0, source="auction"),
            ]
        )

        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        val_repo.upsert(
            Valuation(
                player_id=pid1,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="CF",
                value=25.0,
                rank=1,
                category_scores={},
            )
        )
        val_repo.upsert(
            Valuation(
                player_id=pid2,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="DH",
                value=30.0,
                rank=2,
                category_scores={},
            )
        )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-eval",
                "--gives",
                "Trout",
                "--receives",
                "Ohtani",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Shohei Ohtani" in result.output
        assert "surplus delta" in result.output.lower()
        conn.close()

    def test_trade_eval_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-eval",
                "--gives",
                "Nobody",
                "--receives",
                "Ohtani",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 1
        assert "no player found" in result.output
        conn.close()

    def test_trade_eval_ambiguous_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Smith")
        seed_player(conn, name_first="John", name_last="Smith")
        seed_player(conn, name_first="Shohei", name_last="Ohtani")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-eval",
                "--gives",
                "Smith",
                "--receives",
                "Ohtani",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
            ],
        )
        assert result.exit_code == 1
        assert "ambiguous" in result.output
        conn.close()


def _seed_optimize_data(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    """Seed 4 players with keeper costs and valuations for optimization tests."""
    pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
    pid2 = seed_player(conn, name_first="Aaron", name_last="Judge")
    pid3 = seed_player(conn, name_first="Mookie", name_last="Betts")
    pid4 = seed_player(conn, name_first="Juan", name_last="Soto")

    keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
    keeper_repo.upsert_batch(
        [
            KeeperCost(player_id=pid1, season=2026, league="dynasty", cost=10.0, source="auction"),
            KeeperCost(player_id=pid2, season=2026, league="dynasty", cost=15.0, source="auction"),
            KeeperCost(player_id=pid3, season=2026, league="dynasty", cost=20.0, source="auction"),
            KeeperCost(player_id=pid4, season=2026, league="dynasty", cost=5.0, source="auction"),
        ]
    )

    val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
    for pid, value, rank, pos in [
        (pid1, 30.0, 1, "cf"),
        (pid2, 35.0, 2, "of"),
        (pid3, 28.0, 3, "of"),
        (pid4, 20.0, 4, "c"),
    ]:
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position=pos,
                value=value,
                rank=rank,
                category_scores={},
            )
        )
    conn.commit()
    return pid1, pid2, pid3, pid4


class TestKeeperOptimize:
    def test_basic_optimize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keeper Set" in result.output
        # Top 2 by surplus: Trout (30-10=20), Judge (35-15=20), Soto (20-5=15), Betts (28-20=8)
        assert "Mike Trout" in result.output
        assert "Aaron Judge" in result.output
        conn.close()

    def test_optimize_with_max_per_position(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Limit OF to 1, so can't keep both Trout (cf) and Judge (of) if of limit is 1
        # Actually cf != of, so let's limit cf=0 to force Trout out
        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--max-per-position",
                "cf=0",
            ],
        )
        assert result.exit_code == 0, result.output
        # Trout is CF, can't keep him with cf=0
        assert "Mike Trout" not in result.output
        assert "Aaron Judge" in result.output
        conn.close()

    def test_optimize_with_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Require Betts (worst surplus) — he must appear even though he's suboptimal
        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--required",
                "Betts",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Mookie Betts" in result.output
        conn.close()

    def test_optimize_no_keeper_costs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No keeper costs found" in result.output
        conn.close()

    def test_optimize_with_league_keepers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)
        # Seed an extra player that another team keeps
        pid5 = seed_player(conn, name_first="Freddie", name_last="Freeman")
        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        val_repo.upsert(
            Valuation(
                player_id=pid5,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="1b",
                value=25.0,
                rank=5,
                category_scores={},
            )
        )
        conn.commit()

        csv_path = tmp_path / "league_keepers.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["player_name"])
            writer.writeheader()
            writer.writerow({"player_name": "Freddie Freeman"})

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.load_league",
            lambda name, path: _adjusted_league(),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--league-keepers",
                str(csv_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keeper Set" in result.output
        conn.close()


def _seed_optimize_round_data(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Seed 3 players with round-based keeper costs and valuations."""
    pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
    pid2 = seed_player(conn, name_first="Aaron", name_last="Judge")
    pid3 = seed_player(conn, name_first="Juan", name_last="Soto")

    keeper_repo = SqliteKeeperCostRepo(SingleConnectionProvider(conn))
    keeper_repo.upsert_batch(
        [
            KeeperCost(
                player_id=pid1, season=2026, league="dynasty", cost=25.0, source="draft_round", original_round=3
            ),
            KeeperCost(
                player_id=pid2, season=2026, league="dynasty", cost=18.0, source="draft_round", original_round=3
            ),
            KeeperCost(player_id=pid3, season=2026, league="dynasty", cost=5.0, source="draft_round", original_round=8),
        ]
    )

    val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
    for pid, value, rank, pos in [
        (pid1, 40.0, 1, "cf"),
        (pid2, 35.0, 2, "of"),
        (pid3, 20.0, 3, "ss"),
    ]:
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position=pos,
                value=value,
                rank=rank,
                category_scores={},
            )
        )
    conn.commit()
    return pid1, pid2, pid3


class TestKeeperOptimizeRoundConstraints:
    def test_optimize_with_round_escalation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_round_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--round-escalation",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keeper Set" in result.output
        conn.close()

    def test_optimize_with_max_per_round(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_round_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Two players in round 3, max-per-round=1 → can only keep one from round 3
        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--max-per-round",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should keep Trout (round 3, higher surplus) + Soto (round 8)
        assert "Mike Trout" in result.output
        assert "Juan Soto" in result.output
        conn.close()

    def test_optimize_with_protected_rounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_round_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Protect round 3 → both Trout and Judge excluded, only Soto valid
        result = runner.invoke(
            app,
            [
                "keeper",
                "optimize",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "1",
                "--protected-rounds",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        # Only Soto (round 8) can be kept
        assert "Juan Soto" in result.output
        assert "Mike Trout" not in result.output
        assert "Aaron Judge" not in result.output
        conn.close()


class TestKeeperScenario:
    def test_compare_two_scenarios(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "scenario",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--scenario",
                "Best:Trout,Judge",
                "--scenario",
                "Alt:Trout,Soto",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Scenario Comparison" in result.output
        assert "Best" in result.output
        assert "Alt" in result.output
        conn.close()

    def test_scenario_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "scenario",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--scenario",
                "Bad:Nobody,Trout",
            ],
        )
        assert result.exit_code == 1
        assert "no player found" in result.output
        conn.close()


class TestKeeperTradeImpact:
    def test_acquiring_high_value_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)
        # Seed a new player to acquire
        pid5 = seed_player(conn, name_first="Freddie", name_last="Freeman")
        val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        val_repo.upsert(
            Valuation(
                player_id=pid5,
                season=2026,
                system="zar",
                version="production",
                projection_system="composite",
                projection_version="v1",
                player_type="batter",
                position="1b",
                value=40.0,
                rank=0,
                category_scores={},
            )
        )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-impact",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--acquire",
                "Freeman",
                "--acquire-cost",
                "8.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Before" in result.output
        assert "After" in result.output
        assert "Score delta" in result.output
        conn.close()

    def test_releasing_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-impact",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--release",
                "Trout",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Before" in result.output
        assert "After" in result.output
        conn.close()

    def test_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_optimize_data(conn)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "keeper",
                "trade-impact",
                "--season",
                "2026",
                "--league",
                "dynasty",
                "--system",
                "zar",
                "--max-keepers",
                "2",
                "--acquire",
                "Nobody",
                "--acquire-cost",
                "10.0",
            ],
        )
        assert result.exit_code == 1
        assert "no player found" in result.output
        conn.close()
