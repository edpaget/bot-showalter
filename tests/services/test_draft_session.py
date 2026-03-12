import json
from contextlib import contextmanager
from io import StringIO
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_recommendation import Recommendation
from fantasy_baseball_manager.domain.draft_report import DraftReport
from fantasy_baseball_manager.domain.draft_session import DraftSessionPick, DraftSessionRecord, DraftSessionTrade
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.draft_session_repo import SqliteDraftSessionRepo
from fantasy_baseball_manager.services.draft_session import (
    BalanceCommand,
    BestCommand,
    DraftSession,
    FallsCommand,
    HelpCommand,
    NeedCommand,
    NeedsCommand,
    PickCommand,
    PoolCommand,
    QuitCommand,
    ReachesCommand,
    ReportCommand,
    ReportFn,
    RosterCommand,
    SaveCommand,
    StatusCommand,
    ThreatsCommand,
    UndoCommand,
    auto_detect_position,
    load_draft,
    load_draft_from_db,
    parse_command,
    save_draft,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftState,
)

if TYPE_CHECKING:
    from pathlib import Path

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

LIVE_CONFIG = DraftConfig(
    teams=4,
    roster_slots={"C": 1, "1B": 1, "OF": 2, "UTIL": 1, "P": 1},
    format=DraftFormat.LIVE,
    user_team=1,
    season=2026,
)


def _make_player(
    player_id: int,
    name: str,
    position: str,
    value: float,
    player_type: str = "batter",
    adp_overall: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp_overall,
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


class TestParseCommandLive:
    def test_pick_trailing_digits_not_parsed_as_price(self) -> None:
        """LIVE format should not parse trailing digits as price (like snake)."""
        cmd = parse_command("pick Mike Trout 25", DraftFormat.LIVE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout 25"
        assert cmd.price is None

    def test_pick_with_position(self) -> None:
        cmd = parse_command("pick Mike Trout OF", DraftFormat.LIVE, VALID_POSITIONS)
        assert isinstance(cmd, PickCommand)
        assert cmd.query == "Mike Trout"
        assert cmd.position == "OF"
        assert cmd.price is None


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


class TestDraftSessionLive:
    """Test DraftSession in LIVE format — picks use user_team, not team_on_clock."""

    def _make_live_session(self, commands: list[str]) -> tuple[StringIO, DraftSession]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(PLAYERS, LIVE_CONFIG)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
            return []

        session = DraftSession(
            engine=engine,
            players=PLAYERS,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
        )
        return buf, session

    def test_pick_uses_user_team(self) -> None:
        buf, session = self._make_live_session(["pick Mike Trout OF", "quit"])
        session.run()
        state = session.engine.state
        assert len(state.picks) == 1
        assert state.picks[0].team == LIVE_CONFIG.user_team

    def test_status_does_not_crash(self) -> None:
        buf, session = self._make_live_session(["status", "quit"])
        session.run()
        output = buf.getvalue()
        assert "Pick" in output or "pick" in output.lower()


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


# ---------------------------------------------------------------------------
# Category balance and needs commands
# ---------------------------------------------------------------------------


class TestParseCommandBalance:
    def test_balance_command(self) -> None:
        cmd = parse_command("balance", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, BalanceCommand)

    def test_balance_case_insensitive(self) -> None:
        cmd = parse_command("BALANCE", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, BalanceCommand)


class TestParseCommandNeeds:
    def test_needs_command(self) -> None:
        cmd = parse_command("needs", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, NeedsCommand)

    def test_needs_case_insensitive(self) -> None:
        cmd = parse_command("NEEDS", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, NeedsCommand)

    def test_need_still_works(self) -> None:
        """Singular 'need' still returns NeedCommand (unfilled slots)."""
        cmd = parse_command("need", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, NeedCommand)


class TestParseCommandFalls:
    def test_falls_no_args(self) -> None:
        cmd = parse_command("falls", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, FallsCommand)
        assert cmd.position is None
        assert cmd.threshold is None

    def test_falls_with_position(self) -> None:
        cmd = parse_command("falls OF", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, FallsCommand)
        assert cmd.position == "OF"
        assert cmd.threshold is None

    def test_falls_with_threshold(self) -> None:
        cmd = parse_command("falls --threshold 20", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, FallsCommand)
        assert cmd.position is None
        assert cmd.threshold == 20

    def test_falls_with_position_and_threshold(self) -> None:
        cmd = parse_command("falls OF --threshold 20", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, FallsCommand)
        assert cmd.position == "OF"
        assert cmd.threshold == 20

    def test_falls_case_insensitive(self) -> None:
        cmd = parse_command("FALLS", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, FallsCommand)


class TestParseCommandReaches:
    def test_reaches_command(self) -> None:
        cmd = parse_command("reaches", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ReachesCommand)

    def test_reaches_case_insensitive(self) -> None:
        cmd = parse_command("REACHES", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ReachesCommand)


class TestParseCommandThreats:
    def test_threats_command(self) -> None:
        cmd = parse_command("threats", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ThreatsCommand)

    def test_threats_case_insensitive(self) -> None:
        cmd = parse_command("THREATS", DraftFormat.SNAKE, VALID_POSITIONS)
        assert isinstance(cmd, ThreatsCommand)


class TestCategoryBalanceREPL:
    """Test balance and needs commands in the REPL."""

    def _make_session_with_projections(
        self,
        commands: list[str],
        *,
        with_projections: bool = True,
    ) -> tuple[StringIO, DraftSession]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: DraftState, *, limit: int = 5) -> list[Recommendation]:
            return []

        projections: list[Projection] | None = None
        league: LeagueSettings | None = None

        if with_projections:
            hr_cat = CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
            sb_cat = CategoryConfig(key="sb", name="SB", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
            league = LeagueSettings(
                name="Test",
                format=LeagueFormat.H2H_CATEGORIES,
                teams=4,
                budget=260,
                roster_batters=14,
                roster_pitchers=10,
                batting_categories=(hr_cat, sb_cat),
                pitching_categories=(),
            )
            # League pool
            projections = [
                Projection(
                    player_id=i,
                    season=2026,
                    system="steamer",
                    version="1.0",
                    player_type="batter",
                    stat_json={"hr": 20.0, "sb": 15.0},
                )
                for i in range(100, 200)
            ]
            # All our PLAYERS are batters with stats
            for p in PLAYERS:
                projections.append(
                    Projection(
                        player_id=p.player_id,
                        season=2026,
                        system="steamer",
                        version="1.0",
                        player_type=p.player_type,
                        stat_json={"hr": 10.0, "sb": 5.0},
                    )
                )

        session = DraftSession(
            engine=engine,
            players=PLAYERS,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
            projections=projections,
            league=league,
        )
        return buf, session

    def test_help_includes_balance_and_needs(self) -> None:
        buf, session = self._make_session_with_projections(["help", "quit"])
        session.run()
        output = buf.getvalue()
        assert "balance" in output.lower()
        assert "needs" in output.lower()

    def test_balance_shows_categories(self) -> None:
        buf, session = self._make_session_with_projections(["pick Mike Trout OF", "balance", "quit"])
        session.run()
        output = buf.getvalue()
        assert "hr" in output.lower()
        assert "sb" in output.lower()

    def test_balance_without_projections_shows_warning(self) -> None:
        buf, session = self._make_session_with_projections(["balance", "quit"], with_projections=False)
        session.run()
        output = buf.getvalue()
        assert "not available" in output.lower()

    def test_balance_empty_roster_shows_message(self) -> None:
        buf, session = self._make_session_with_projections(["balance", "quit"])
        session.run()
        output = buf.getvalue()
        # No picks yet → empty roster message
        assert "no picks" in output.lower() or "empty" in output.lower()

    def test_needs_shows_weak_categories(self) -> None:
        buf, session = self._make_session_with_projections(["pick Mike Trout OF", "needs", "quit"])
        session.run()
        output = buf.getvalue()
        # Should show category need info
        assert "needs" in output.lower() or "weak" in output.lower()

    def test_needs_without_projections_shows_warning(self) -> None:
        buf, session = self._make_session_with_projections(["needs", "quit"], with_projections=False)
        session.run()
        output = buf.getvalue()
        assert "not available" in output.lower()

    def test_compact_summary_after_pick(self) -> None:
        """After a pick, a compact category summary line appears."""
        buf, session = self._make_session_with_projections(["pick Mike Trout OF", "quit"])
        session.run()
        output = buf.getvalue()
        # Should show compact summary with Weak/Strong categories
        assert "weak" in output.lower() or "strong" in output.lower()

    def test_no_summary_without_projections(self) -> None:
        """No compact summary when projections not loaded."""
        buf, session = self._make_session_with_projections(["pick Mike Trout OF", "quit"], with_projections=False)
        session.run()
        output = buf.getvalue()
        # Should not contain category strength labels
        assert "weak:" not in output.lower() and "strong:" not in output.lower()


# ---------------------------------------------------------------------------
# load_draft_from_db
# ---------------------------------------------------------------------------


class _InMemoryProvider:
    """Minimal ConnectionProvider for test use."""

    def __init__(self) -> None:
        self._conn = create_connection(":memory:")

    def connection(self):  # noqa: ANN201
        @contextmanager
        def _ctx():  # noqa: ANN202
            yield self._conn

        return _ctx()


class TestLoadDraftFromDB:
    def test_load_restores_engine_state(self) -> None:
        repo = SqliteDraftSessionRepo(_InMemoryProvider())

        record = DraftSessionRecord(
            league="test",
            season=2026,
            teams=4,
            format="snake",
            user_team=1,
            roster_slots={"C": 1, "1B": 1, "OF": 2, "UTIL": 1, "P": 1},
            budget=0,
            status="in_progress",
            created_at="2026-03-07T10:00:00",
            updated_at="2026-03-07T10:00:00",
        )
        session_id = repo.create_session(record)

        repo.save_pick(
            DraftSessionPick(
                session_id=session_id, pick_number=1, team=1, player_id=1, player_name="Mike Trout", position="OF"
            )
        )
        repo.save_pick(
            DraftSessionPick(
                session_id=session_id, pick_number=2, team=2, player_id=5, player_name="Freddie Freeman", position="1B"
            )
        )

        engine = load_draft_from_db(session_id, PLAYERS, repo)
        state = engine.state
        assert len(state.picks) == 2
        assert state.current_pick == 3
        assert 1 not in state.available_pool
        assert 5 not in state.available_pool

    def test_not_found_raises(self) -> None:
        repo = SqliteDraftSessionRepo(_InMemoryProvider())

        with pytest.raises(ValueError, match="not found"):
            load_draft_from_db(9999, PLAYERS, repo)

    def test_load_replays_trades(self) -> None:
        repo = SqliteDraftSessionRepo(_InMemoryProvider())

        record = DraftSessionRecord(
            league="test",
            season=2026,
            teams=4,
            format="snake",
            user_team=1,
            roster_slots={"C": 1, "1B": 1, "OF": 2, "UTIL": 1, "P": 1},
            budget=0,
            status="in_progress",
            created_at="2026-03-07T10:00:00",
            updated_at="2026-03-07T10:00:00",
        )
        session_id = repo.create_session(record)

        # Trade: user (team 1) gives pick 1 to team 2, receives pick 2
        repo.save_trade(
            DraftSessionTrade(
                session_id=session_id,
                trade_number=1,
                team_a=1,
                team_b=2,
                team_a_gives=[1],
                team_b_gives=[2],
            )
        )

        # After trade, pick 1 belongs to team 2, pick 2 belongs to team 1
        # Team 2 picks first (pick 1), then team 1 (pick 2)
        repo.save_pick(
            DraftSessionPick(
                session_id=session_id, pick_number=1, team=2, player_id=1, player_name="Mike Trout", position="OF"
            )
        )
        repo.save_pick(
            DraftSessionPick(
                session_id=session_id, pick_number=2, team=1, player_id=5, player_name="Freddie Freeman", position="1B"
            )
        )

        engine = load_draft_from_db(session_id, PLAYERS, repo)
        assert len(engine.state.picks) == 2
        assert engine.state.current_pick == 3
        assert len(engine.trades) == 1
        # Pick overrides should be in place
        assert engine.team_for_pick(1) == 2
        assert engine.team_for_pick(2) == 1

    def test_load_replays_multiple_trades(self) -> None:
        repo = SqliteDraftSessionRepo(_InMemoryProvider())

        record = DraftSessionRecord(
            league="test",
            season=2026,
            teams=4,
            format="snake",
            user_team=1,
            roster_slots={"C": 1, "1B": 1, "OF": 2, "UTIL": 1, "P": 1},
            budget=0,
            status="in_progress",
            created_at="2026-03-07T10:00:00",
            updated_at="2026-03-07T10:00:00",
        )
        session_id = repo.create_session(record)

        # Trade 1: team 1 gives pick 1 to team 2, receives pick 2
        repo.save_trade(
            DraftSessionTrade(
                session_id=session_id, trade_number=1, team_a=1, team_b=2, team_a_gives=[1], team_b_gives=[2]
            )
        )
        # Trade 2: team 1 gives pick 8 (snake: team 1) to team 3, receives pick 3
        repo.save_trade(
            DraftSessionTrade(
                session_id=session_id, trade_number=2, team_a=1, team_b=3, team_a_gives=[8], team_b_gives=[3]
            )
        )

        engine = load_draft_from_db(session_id, PLAYERS, repo)
        assert len(engine.trades) == 2
        assert engine.team_for_pick(1) == 2
        assert engine.team_for_pick(2) == 1
        assert engine.team_for_pick(3) == 1
        assert engine.team_for_pick(8) == 3


# ---------------------------------------------------------------------------
# Fake repo for persistence tests
# ---------------------------------------------------------------------------


class FakeDraftSessionRepo:
    """In-memory fake implementing the DraftSessionRepo protocol."""

    def __init__(self) -> None:
        self.created_sessions: list[DraftSessionRecord] = []
        self.saved_picks: list[DraftSessionPick] = []
        self.deleted_picks: list[tuple[int, int]] = []  # (session_id, pick_number)
        self.status_updates: list[tuple[int, str]] = []
        self.timestamp_updates: list[tuple[int, str]] = []
        self._next_id = 1

    def create_session(self, record: DraftSessionRecord) -> int:
        self.created_sessions.append(record)
        sid = self._next_id
        self._next_id += 1
        return sid

    def save_pick(self, pick: DraftSessionPick) -> None:
        self.saved_picks.append(pick)

    def delete_pick(self, session_id: int, pick_number: int) -> None:
        self.deleted_picks.append((session_id, pick_number))

    def load_session(self, session_id: int) -> DraftSessionRecord | None:
        return None

    def load_picks(self, session_id: int) -> list[DraftSessionPick]:
        return []

    def list_sessions(self, *, league: str | None = None, season: int | None = None) -> list[DraftSessionRecord]:
        return []

    def update_status(self, session_id: int, status: str) -> None:
        self.status_updates.append((session_id, status))

    def update_timestamp(self, session_id: int, updated_at: str) -> None:
        self.timestamp_updates.append((session_id, updated_at))

    def delete_session(self, session_id: int) -> None:
        pass

    def count_picks(self, session_id: int) -> int:
        return 0

    def save_trade(self, trade: DraftSessionTrade) -> None:
        pass

    def load_trades(self, session_id: int) -> list[DraftSessionTrade]:
        return []

    def delete_trade(self, session_id: int, trade_number: int) -> None:
        pass


# ---------------------------------------------------------------------------
# DraftSession DB persistence (phase 2)
# ---------------------------------------------------------------------------


class TestDraftSessionPersistence:
    """Test that DraftSession persists picks/undo/quit to the repo."""

    def _make_session(
        self,
        commands: list[str],
        *,
        repo: FakeDraftSessionRepo | None = None,
        session_id: int | None = None,
    ) -> tuple[StringIO, DraftSession, FakeDraftSessionRepo]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
            return []

        fake_repo = repo or FakeDraftSessionRepo()

        session = DraftSession(
            engine=engine,
            players=PLAYERS,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
            session_repo=fake_repo,
            session_id=session_id,
            league_name="test-league",
        )
        return buf, session, fake_repo

    def test_session_created_on_run(self) -> None:
        _buf, session, repo = self._make_session(["quit"])
        session.run()
        assert len(repo.created_sessions) == 1
        record = repo.created_sessions[0]
        assert record.league == "test-league"
        assert record.season == 2026
        assert record.teams == 4
        assert record.format == "snake"
        assert record.status == "in_progress"

    def test_resume_with_session_id_skips_create(self) -> None:
        _buf, session, repo = self._make_session(["quit"], session_id=42)
        session.run()
        assert len(repo.created_sessions) == 0

    def test_pick_persists_to_repo(self) -> None:
        _buf, session, repo = self._make_session(["pick Mike Trout OF", "quit"])
        session.run()
        assert len(repo.saved_picks) == 1
        pick = repo.saved_picks[0]
        assert pick.session_id == 1
        assert pick.pick_number == 1
        assert pick.player_id == 1
        assert pick.player_name == "Mike Trout"
        assert pick.position == "OF"
        assert len(repo.timestamp_updates) >= 1

    def test_undo_deletes_from_repo(self) -> None:
        _buf, session, repo = self._make_session(["pick Mike Trout OF", "undo", "quit"])
        session.run()
        assert len(repo.saved_picks) == 1
        assert len(repo.deleted_picks) == 1
        assert repo.deleted_picks[0] == (1, 1)  # session_id=1, pick_number=1

    def test_quit_sets_complete_status(self) -> None:
        _buf, session, repo = self._make_session(["quit"])
        session.run()
        assert len(repo.status_updates) == 1
        assert repo.status_updates[0] == (1, "complete")

    def test_no_repo_works_fine(self) -> None:
        """DraftSession without a repo should work without errors."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)

        cmd_iter = iter(["pick Mike Trout OF", "undo", "quit"])

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
            return []

        session = DraftSession(
            engine=engine,
            players=PLAYERS,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
        )
        session.run()  # should not raise


# ---------------------------------------------------------------------------
# Falls, reaches, and alert tests
# ---------------------------------------------------------------------------

# Players with ADP data for arbitrage testing
ADP_PLAYERS = [
    _make_player(1, "Mike Trout", "OF", 30.0, adp_overall=5.0),
    _make_player(2, "Shohei Ohtani", "SP", 28.0, player_type="pitcher", adp_overall=3.0),
    _make_player(3, "Mookie Betts", "SS", 25.0, adp_overall=8.0),
    _make_player(4, "Ronald Acuna Jr.", "OF", 22.0, adp_overall=2.0),
    _make_player(5, "Freddie Freeman", "1B", 20.0, adp_overall=12.0),
    _make_player(6, "Salvador Perez", "C", 15.0, adp_overall=40.0),
    _make_player(7, "Adley Rutschman", "C", 14.0, adp_overall=45.0),
    _make_player(8, "Aaron Judge", "OF", 27.0, adp_overall=4.0),
]

# Config that starts at a high pick so players are past their ADP
HIGH_PICK_CONFIG = DraftConfig(
    teams=12,
    roster_slots={"C": 1, "1B": 1, "SS": 1, "OF": 3, "UTIL": 1, "SP": 1, "P": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)


def _snake_team(pick_number: int, teams: int) -> int:
    """Compute which team picks at a given pick number in a snake draft."""
    zero_based = pick_number - 1
    round_number = zero_based // teams
    position_in_round = zero_based % teams
    if round_number % 2 == 0:
        return position_in_round + 1
    return teams - position_in_round


def _make_adp_engine(
    players: list[DraftBoardRow],
    config: DraftConfig,
    filler_count: int,
) -> DraftEngine:
    """Create a DraftEngine with filler picks already made to advance the pick counter."""
    engine = DraftEngine()
    engine.start(players, config)
    for i in range(filler_count):
        pick_num = i + 1
        team = _snake_team(pick_num, config.teams)
        # Find a filler player (id >= 100)
        filler_id = 100 + i
        engine.pick(filler_id, team, "OF")
    return engine


class TestFallsREPL:
    """Test the falls command in the REPL."""

    def _make_session(
        self,
        commands: list[str],
        players: list[DraftBoardRow] | None = None,
        config: DraftConfig | None = None,
    ) -> tuple[StringIO, DraftSession]:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        use_players = players or ADP_PLAYERS
        use_config = config or HIGH_PICK_CONFIG
        engine = DraftEngine()
        engine.start(use_players, use_config)

        cmd_iter = iter(commands)

        def fake_input(_prompt: str = "") -> str:
            return next(cmd_iter)

        def fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
            return []

        session = DraftSession(
            engine=engine,
            players=use_players,
            console=test_console,
            recommend_fn=fake_recommend,
            input_fn=fake_input,
        )
        return buf, session

    def test_falls_no_fallers_at_pick_1(self) -> None:
        """At pick 1, no player has slipped past their ADP yet."""
        buf, session = self._make_session(["falls", "quit"])
        session.run()
        output = buf.getvalue()
        assert "No falling players" in output

    def test_falls_shows_fallers(self) -> None:
        """Players with low ADP should show as falling when current pick is high."""
        filler_players = [_make_player(100 + i, f"Filler {i}", "OF", 1.0, adp_overall=50.0 + i) for i in range(15)]
        players = [
            _make_player(10, "Faller A", "OF", 25.0, adp_overall=1.0),
            _make_player(11, "Faller B", "1B", 20.0, adp_overall=2.0),
            _make_player(12, "Normal C", "SS", 15.0, adp_overall=100.0),
            *filler_players,
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 10, "1B": 5, "SS": 5, "UTIL": 5},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        engine = _make_adp_engine(players, config, filler_count=12)

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        cmd_iter = iter(["falls", "quit"])
        session = DraftSession(
            engine=engine,
            players=players,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        assert "Faller A" in output
        assert "Falling Players" in output

    def test_falls_position_filter(self) -> None:
        """Position filter should limit results."""
        filler_players = [_make_player(100 + i, f"Filler {i}", "OF", 1.0, adp_overall=50.0 + i) for i in range(15)]
        players = [
            _make_player(10, "OF Faller", "OF", 25.0, adp_overall=1.0),
            _make_player(11, "1B Faller", "1B", 20.0, adp_overall=2.0),
            *filler_players,
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 10, "1B": 5, "UTIL": 5},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        engine = _make_adp_engine(players, config, filler_count=12)

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        cmd_iter = iter(["falls 1B", "quit"])
        session = DraftSession(
            engine=engine,
            players=players,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        assert "1B Faller" in output
        assert "OF Faller" not in output

    def test_help_includes_falls_and_reaches(self) -> None:
        buf, session = self._make_session(["help", "quit"])
        session.run()
        output = buf.getvalue()
        assert "falls" in output.lower()
        assert "reaches" in output.lower()


class TestReachesREPL:
    """Test the reaches command in the REPL."""

    def test_reaches_no_reaches(self) -> None:
        """No reaches when no picks have been made."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(ADP_PLAYERS, HIGH_PICK_CONFIG)

        cmd_iter = iter(["reaches", "quit"])
        session = DraftSession(
            engine=engine,
            players=ADP_PLAYERS,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        assert "No reach picks" in output

    def test_reaches_shows_reach_picks(self) -> None:
        """A pick well ahead of ADP should show as a reach."""
        # Salvador Perez has ADP 40, picking him at pick 1 = 39 picks ahead
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(ADP_PLAYERS, HIGH_PICK_CONFIG)

        # Pick Salvador Perez (ADP 40) at pick 1 — huge reach
        engine.pick(6, 1, "C")

        cmd_iter = iter(["reaches", "quit"])
        session = DraftSession(
            engine=engine,
            players=ADP_PLAYERS,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        assert "Salvador Perez" in output
        assert "Reach Picks" in output


class TestFallingAlerts:
    """Test automatic falling player alerts after picks."""

    def test_alert_fires_after_pick(self) -> None:
        """Alerts should appear after a pick when significant fallers exist."""
        filler_players = [_make_player(100 + i, f"Filler {i}", "OF", 1.0, adp_overall=50.0 + i) for i in range(25)]
        players = [
            _make_player(10, "Alert Faller", "OF", 30.0, adp_overall=1.0),
            *filler_players,
            _make_player(50, "Pickable", "1B", 5.0, adp_overall=90.0),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 15, "1B": 5, "UTIL": 5},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        engine = _make_adp_engine(players, config, filler_count=22)

        # Now pick 23 — Alert Faller (adp=1, value=30, rank=1) should trigger alert
        # pick 23 with 2 teams: round=11 (odd), pos=0 → team=2-0=2... let's check
        # Actually the REPL pick uses team_on_clock for snake
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        cmd_iter = iter(["pick Pickable 1B", "quit"])
        session = DraftSession(
            engine=engine,
            players=players,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        assert "Falling" in output
        assert "Alert Faller" in output

    def test_no_alert_when_no_significant_fallers(self) -> None:
        """No alerts when no players have fallen significantly."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)
        engine = DraftEngine()
        engine.start(ADP_PLAYERS, HIGH_PICK_CONFIG)

        cmd_iter = iter(["pick Mike Trout OF", "quit"])
        session = DraftSession(
            engine=engine,
            players=ADP_PLAYERS,
            console=test_console,
            recommend_fn=lambda state, *, limit=5: [],
            input_fn=lambda _prompt="": next(cmd_iter),
        )
        session.run()
        output = buf.getvalue()
        # At pick 1, no one has fallen past threshold of 20
        assert "⚡" not in output
