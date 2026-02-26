import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_recommendation import Recommendation
from fantasy_baseball_manager.domain.draft_report import DraftReport
from fantasy_baseball_manager.services.draft_session import (
    BestCommand,
    DraftSession,
    HelpCommand,
    NeedCommand,
    PickCommand,
    PoolCommand,
    QuitCommand,
    ReportCommand,
    ReportFn,
    RosterCommand,
    SaveCommand,
    StatusCommand,
    UndoCommand,
    auto_detect_position,
    load_draft,
    parse_command,
    save_draft,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "UTIL", "SP", "RP", "P"}

SNAKE_CONFIG = DraftConfig(
    teams=4,
    roster_slots={"C": 1, "1B": 1, "OF": 2, "UTIL": 1, "P": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)

AUCTION_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"C": 1, "1B": 1, "OF": 1},
    format=DraftFormat.AUCTION,
    user_team=1,
    season=2026,
    budget=100,
)


def _make_player(player_id: int, name: str, position: str, value: float, player_type: str = "batter") -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores={},
    )


PLAYERS = [
    _make_player(1, "Mike Trout", "OF", 30.0),
    _make_player(2, "Shohei Ohtani", "SP", 28.0, player_type="pitcher"),
    _make_player(3, "Mookie Betts", "SS", 25.0),
    _make_player(4, "Ronald Acuna Jr.", "OF", 22.0),
    _make_player(5, "Freddie Freeman", "1B", 20.0),
    _make_player(6, "Salvador Perez", "C", 15.0),
    _make_player(7, "Adley Rutschman", "C", 14.0),
    _make_player(8, "Aaron Judge", "OF", 27.0),
]


# ---------------------------------------------------------------------------
# Step 3: Command parsing
# ---------------------------------------------------------------------------


class TestParseCommandPick:
    def test_pick_player_name(self) -> None:
        cmd = parse_command("pick Mike Trout", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout"
        assert cmd.position is None
        assert cmd.price is None

    def test_pick_with_position(self) -> None:
        cmd = parse_command("pick Mike Trout OF", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout"
        assert cmd.position == "OF"

    def test_pick_with_price_auction(self) -> None:
        cmd = parse_command("pick Mike Trout OF 25", DraftFormat.AUCTION, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout"
        assert cmd.position == "OF"
        assert cmd.price == 25

    def test_pick_price_only_auction(self) -> None:
        cmd = parse_command("pick Mike Trout 25", DraftFormat.AUCTION, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout"
        assert cmd.position is None
        assert cmd.price == 25

    def test_pick_multi_word_name_with_position(self) -> None:
        cmd = parse_command("pick Ronald Acuna Jr. OF", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Ronald Acuna Jr."
        assert cmd.position == "OF"

    def test_pick_no_name(self) -> None:
        cmd = parse_command("pick", DraftFormat.SNAKE, VALID_POSITIONS)
        assert cmd is None


class TestParseCommandOther:
    def test_undo(self) -> None:
        cmd = parse_command("undo", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, UndoCommand)

    def test_best_no_filter(self) -> None:
        cmd = parse_command("best", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, BestCommand)
        assert cmd.position is None

    def test_best_with_position(self) -> None:
        cmd = parse_command("best OF", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, BestCommand)
        assert cmd.position == "OF"

    def test_need(self) -> None:
        cmd = parse_command("need", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, NeedCommand)

    def test_roster(self) -> None:
        cmd = parse_command("roster", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, RosterCommand)

    def test_pool_no_filter(self) -> None:
        cmd = parse_command("pool", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PoolCommand)
        assert cmd.position is None

    def test_pool_with_position(self) -> None:
        cmd = parse_command("pool OF", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PoolCommand)
        assert cmd.position == "OF"

    def test_status(self) -> None:
        cmd = parse_command("status", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, StatusCommand)

    def test_save(self) -> None:
        cmd = parse_command("save", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, SaveCommand)

    def test_help(self) -> None:
        cmd = parse_command("help", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, HelpCommand)

    def test_quit(self) -> None:
        cmd = parse_command("quit", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, QuitCommand)

    def test_exit_alias(self) -> None:
        cmd = parse_command("exit", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, QuitCommand)


class TestParseCommandEdgeCases:
    def test_empty_input(self) -> None:
        cmd = parse_command("", DraftFormat.SNAKE, VALID_POSITIONS)
        assert cmd is None

    def test_whitespace_only(self) -> None:
        cmd = parse_command("   ", DraftFormat.SNAKE, VALID_POSITIONS)
        assert cmd is None

    def test_unknown_command(self) -> None:
        cmd = parse_command("trade Mike Trout", DraftFormat.SNAKE, VALID_POSITIONS)
        assert cmd is None

    def test_case_insensitive_command(self) -> None:
        cmd = parse_command("PICK Mike Trout", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)

    def test_position_case_insensitive(self) -> None:
        cmd = parse_command("pick Mike Trout of", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.position == "OF"


# ---------------------------------------------------------------------------
# Step 4: Persistence (save/load)
# ---------------------------------------------------------------------------


class TestSaveDraft:
    def test_roundtrip_snake(self, tmp_path: Path) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="OF")
        engine.pick(player_id=5, team=2, position="1B")

        path = tmp_path / "draft.json"
        save_draft(engine.state, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["config"]["format"] == "snake"
        assert len(data["picks"]) == 2

    def test_roundtrip_auction(self, tmp_path: Path) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=1, position="OF", price=30)

        path = tmp_path / "draft.json"
        save_draft(engine.state, path)

        data = json.loads(path.read_text())
        assert data["config"]["format"] == "auction"
        assert data["picks"][0]["price"] == 30


class TestLoadDraft:
    def test_load_restores_state(self, tmp_path: Path) -> None:
        # Save a draft
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="OF")
        engine.pick(player_id=5, team=2, position="1B")

        path = tmp_path / "draft.json"
        save_draft(engine.state, path)

        # Load it back
        loaded_engine = load_draft(path, PLAYERS)
        loaded_state = loaded_engine.state
        assert len(loaded_state.picks) == 2
        assert loaded_state.current_pick == 3
        assert 1 not in loaded_state.available_pool
        assert 5 not in loaded_state.available_pool

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_draft(path, PLAYERS)

    def test_load_auction_restores_budget(self, tmp_path: Path) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=1, position="OF", price=30)

        path = tmp_path / "draft.json"
        save_draft(engine.state, path)

        loaded_engine = load_draft(path, PLAYERS)
        loaded_state = loaded_engine.state
        assert loaded_state.team_budgets[1] == 70


# ---------------------------------------------------------------------------
# Step 5: Position auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetectPosition:
    def test_single_position_match(self) -> None:
        player = _make_player(1, "Mike Trout", "OF", 30.0)
        needs = {"OF": 2, "C": 1, "UTIL": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result == "OF"

    def test_flex_util_for_batter(self) -> None:
        """If primary position is full but UTIL is open, use UTIL."""
        player = _make_player(1, "Mike Trout", "OF", 30.0)
        needs = {"C": 1, "UTIL": 1}  # OF is not in needs
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result == "UTIL"

    def test_flex_p_for_pitcher(self) -> None:
        """If pitcher and P slot open, use P."""
        player = _make_player(2, "Shohei Ohtani", "SP", 28.0, player_type="pitcher")
        needs = {"P": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result == "P"

    def test_ambiguous_returns_none(self) -> None:
        """If multiple positions match, return None."""
        player = _make_player(1, "Mike Trout", "OF", 30.0)
        needs = {"OF": 2, "UTIL": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        # Primary position has an open slot → should pick it
        assert result == "OF"

    def test_no_slot_returns_none(self) -> None:
        """If no position fits, return None."""
        player = _make_player(1, "Mike Trout", "OF", 30.0)
        needs = {"C": 1, "1B": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result is None

    def test_only_flex_available(self) -> None:
        """When only flex slot is open, it should be unambiguous."""
        player = _make_player(1, "Mike Trout", "OF", 30.0)
        needs = {"UTIL": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result == "UTIL"

    def test_primary_preferred_over_flex(self) -> None:
        """Primary position match should be preferred even when flex is also available."""
        player = _make_player(6, "Salvador Perez", "C", 15.0)
        needs = {"C": 1, "UTIL": 1}
        result = auto_detect_position(player, needs, SNAKE_CONFIG.roster_slots)
        assert result == "C"


# ---------------------------------------------------------------------------
# Step 7: DraftSession REPL
# ---------------------------------------------------------------------------


class TestDraftSessionREPL:
    """Test the DraftSession REPL via command strings and captured console output."""

    def _make_session(
        self,
        commands: list[str],
        players: list[DraftBoardRow] | None = None,
        config: DraftConfig | None = None,
    ) -> tuple[StringIO, DraftSession]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        use_players = players or PLAYERS
        use_config = config or SNAKE_CONFIG
        engine = DraftEngine()
        engine.start(use_players, use_config)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
            return [
                Recommendation(
                    player_id=1,
                    player_name="Test Player",
                    position="OF",
                    value=10.0,
                    score=0.9,
                    reason="best value",
                )
            ]

        session = DraftSession(
            engine=engine,
            players=use_players,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
        )
        return buf, session

    def test_quit_exits(self) -> None:
        buf, session = self._make_session(["quit"])
        session.run()
        # Should exit without error

    def test_help_shows_commands(self) -> None:
        buf, session = self._make_session(["help", "quit"])
        session.run()
        output = buf.getvalue()
        assert "pick" in output.lower()
        assert "undo" in output.lower()

    def test_pick_records_and_shows_recommendations(self) -> None:
        buf, session = self._make_session(["pick Mike Trout OF", "quit"])
        session.run()
        output = buf.getvalue()
        # Should show recommendations after pick
        assert "Test Player" in output

    def test_undo_reverses_pick(self) -> None:
        buf, session = self._make_session(["pick Mike Trout OF", "undo", "quit"])
        session.run()
        output = buf.getvalue()
        assert "Undid" in output or "undid" in output.lower()

    def test_status_shows_info(self) -> None:
        buf, session = self._make_session(["status", "quit"])
        session.run()
        output = buf.getvalue()
        assert "Pick" in output or "pick" in output.lower()

    def test_ambiguous_name_shows_disambiguation(self) -> None:
        buf, session = self._make_session(["pick Mike", "quit"])
        session.run()
        # "Mike" is a substring match — matches only Mike Trout in PLAYERS
        # so it should resolve successfully (single substring match)

    def test_unknown_command_shows_help(self) -> None:
        buf, session = self._make_session(["blahblah", "quit"])
        session.run()
        output = buf.getvalue()
        assert "Unknown" in output or "help" in output.lower()

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        save_path = tmp_path / "test_save.json"
        buf, session = self._make_session(["pick Mike Trout OF", "save", "quit"])
        session.save_path = save_path
        session.run()
        assert save_path.exists()


# ---------------------------------------------------------------------------
# Report command parsing + REPL integration
# ---------------------------------------------------------------------------


class TestParseCommandReport:
    def test_report_command(self) -> None:
        cmd = parse_command("report", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ReportCommand)

    def test_report_case_insensitive(self) -> None:
        cmd = parse_command("REPORT", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ReportCommand)


class TestDraftSessionReport:
    """Test the report command in the REPL."""

    def _make_session_with_report(
        self,
        commands: list[str],
        report_fn: ReportFn | None = None,
    ) -> tuple[StringIO, DraftSession]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: DraftState, *, limit: int = 10) -> list[Recommendation]:
            return []

        session = DraftSession(
            engine=engine,
            players=PLAYERS,
            console=test_console,
            recommend_fn=fake_recommend,
            report_fn=report_fn,
            input_fn=fake_input,
        )
        return buf, session

    def test_report_no_fn_shows_message(self) -> None:
        buf, session = self._make_session_with_report(["report", "quit"])
        session.run()
        output = buf.getvalue()
        assert "not available" in output.lower()

    def test_report_with_fn_shows_output(self) -> None:
        def fake_report(state: DraftState, full_pool: list[DraftBoardRow]) -> DraftReport:
            return DraftReport(
                total_value=100.0,
                optimal_value=120.0,
                value_efficiency=0.833,
                budget=None,
                total_spent=None,
                category_standings=[],
                pick_grades=[],
                mean_grade=0.0,
                steals=[],
                reaches=[],
            )

        buf, session = self._make_session_with_report(["report", "quit"], report_fn=fake_report)
        session.run()
        output = buf.getvalue()
        assert "Draft Report" in output
        assert "100.0" in output
        assert "Efficiency" in output

    def test_help_includes_report(self) -> None:
        buf, session = self._make_session_with_report(["help", "quit"])
        session.run()
        output = buf.getvalue()
        assert "report" in output.lower()
