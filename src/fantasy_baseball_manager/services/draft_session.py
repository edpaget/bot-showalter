import json
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftError,
    DraftFormat,
    DraftState,
)
from fantasy_baseball_manager.services.draft_translation import ingest_yahoo_pick
from fantasy_baseball_manager.services.player_resolver import resolve_player

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from fantasy_baseball_manager.domain import (
        DraftBoardRow,
        DraftReport,
        Recommendation,
        YahooDraftPick,
    )
# ---------------------------------------------------------------------------
# Command types (tagged union)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickCommand:
    query: str
    position: str | None = None
    price: int | None = None


@dataclass(frozen=True)
class UndoCommand: ...


@dataclass(frozen=True)
class BestCommand:
    position: str | None = None


@dataclass(frozen=True)
class NeedCommand: ...


@dataclass(frozen=True)
class RosterCommand: ...


@dataclass(frozen=True)
class PoolCommand:
    position: str | None = None


@dataclass(frozen=True)
class StatusCommand: ...


@dataclass(frozen=True)
class SaveCommand: ...


@dataclass(frozen=True)
class HelpCommand: ...


@dataclass(frozen=True)
class ReportCommand: ...


@dataclass(frozen=True)
class QuitCommand: ...


Command = (
    PickCommand
    | UndoCommand
    | BestCommand
    | NeedCommand
    | RosterCommand
    | PoolCommand
    | StatusCommand
    | SaveCommand
    | ReportCommand
    | HelpCommand
    | QuitCommand
)


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


def parse_command(
    line: str,
    fmt: DraftFormat,
    valid_positions: set[str],
) -> Command | None:
    """Parse a REPL input line into a Command, or None if invalid."""
    stripped = line.strip()
    if not stripped:
        return None

    tokens = stripped.split()
    verb = tokens[0].lower()
    args = tokens[1:]

    if verb == "pick":
        return _parse_pick(args, fmt, valid_positions)
    if verb == "undo":
        return UndoCommand()
    if verb == "best":
        pos = _match_position(args[0], valid_positions) if args else None
        return BestCommand(position=pos)
    if verb == "need":
        return NeedCommand()
    if verb == "roster":
        return RosterCommand()
    if verb == "pool":
        pos = _match_position(args[0], valid_positions) if args else None
        return PoolCommand(position=pos)
    if verb == "status":
        return StatusCommand()
    if verb == "save":
        return SaveCommand()
    if verb == "report":
        return ReportCommand()
    if verb == "help":
        return HelpCommand()
    if verb in ("quit", "exit"):
        return QuitCommand()

    return None


def _match_position(token: str, valid_positions: set[str]) -> str | None:
    """Return the canonical position if token matches (case-insensitive)."""
    upper = token.upper()
    if upper in valid_positions:
        return upper
    return None


def _parse_pick(
    args: list[str],
    fmt: DraftFormat,
    valid_positions: set[str],
) -> PickCommand | None:
    """Parse the arguments after 'pick', extracting player query, position, and price.

    Parse from the end:
    - Last token is price if numeric and format is auction
    - Next-to-last (or last after removing price) is position if in valid_positions
    - Remaining tokens form the player query
    """
    if not args:
        return None

    remaining = list(args)
    price: int | None = None
    position: str | None = None

    # Check for price at end (auction only)
    if fmt == DraftFormat.AUCTION and remaining and remaining[-1].isdigit():
        price = int(remaining.pop())

    # Check for position at end
    if remaining:
        pos_candidate = _match_position(remaining[-1], valid_positions)
        if pos_candidate is not None:
            position = pos_candidate
            remaining.pop()

    if not remaining:
        return None

    query = " ".join(remaining)
    return PickCommand(query=query, position=position, price=price)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_draft(state: DraftState, path: Path) -> None:
    """Save draft state to a JSON file."""
    data = {
        "config": {
            "teams": state.config.teams,
            "roster_slots": state.config.roster_slots,
            "format": state.config.format.value,
            "user_team": state.config.user_team,
            "season": state.config.season,
            "budget": state.config.budget,
        },
        "picks": [
            {
                "pick_number": p.pick_number,
                "team": p.team,
                "player_id": p.player_id,
                "player_name": p.player_name,
                "position": p.position,
                "price": p.price,
            }
            for p in state.picks
        ],
    }
    path.write_text(json.dumps(data, indent=2))


def load_draft(path: Path, players: list[DraftBoardRow]) -> DraftEngine:
    """Load a draft from a JSON file by replaying picks."""
    if not path.exists():
        msg = f"Draft file not found: {path}"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text())
    config_data = data["config"]
    config = DraftConfig(
        teams=config_data["teams"],
        roster_slots=config_data["roster_slots"],
        format=DraftFormat(config_data["format"]),
        user_team=config_data["user_team"],
        season=config_data["season"],
        budget=config_data.get("budget", 0),
    )

    engine = DraftEngine()
    engine.start(players, config)

    for pick_data in data["picks"]:
        engine.pick(
            player_id=pick_data["player_id"],
            team=pick_data["team"],
            position=pick_data["position"],
            price=pick_data.get("price"),
        )

    return engine


# ---------------------------------------------------------------------------
# Position auto-detection
# ---------------------------------------------------------------------------


def auto_detect_position(
    player: DraftBoardRow,
    needs: dict[str, int],
    roster_slots: dict[str, int],
) -> str | None:
    """Determine the roster position for a player, or None if ambiguous.

    Logic:
    1. If primary position has an open slot, return it.
    2. If not, check flex (UTIL for batters, P for pitchers).
    3. If exactly one option, return it. Otherwise None.
    """
    candidates: list[str] = []

    # Primary position
    if player.position in needs:
        candidates.append(player.position)

    # Flex slots
    if player.player_type == "batter" and "UTIL" in needs:
        candidates.append("UTIL")
    elif player.player_type == "pitcher" and "P" in needs:
        candidates.append("P")

    if not candidates:
        return None

    # Prefer primary position
    if player.position in candidates:
        return player.position

    if len(candidates) == 1:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Recommend function protocol
# ---------------------------------------------------------------------------


class RecommendFn(Protocol):
    def __call__(self, state: DraftState, *, limit: int = 10) -> list[Recommendation]: ...


class ReportFn(Protocol):
    def __call__(self, state: DraftState, full_pool: list[DraftBoardRow]) -> DraftReport: ...


# ---------------------------------------------------------------------------
# REPL output helpers
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Commands:
  pick <player> [position] [price]  — Record a pick
  undo                              — Reverse the last pick
  best [position]                   — Show top recommendations
  need                              — Show unfilled roster slots
  roster                            — Show your roster
  pool [position]                   — Show available players
  status                            — Show current draft status
  save                              — Save draft to file
  report                            — Show post-draft analysis
  help                              — Show this help
  quit                              — Exit the session
"""


# ---------------------------------------------------------------------------
# DraftSession REPL
# ---------------------------------------------------------------------------


class DraftSession:
    def __init__(
        self,
        engine: DraftEngine,
        players: list[DraftBoardRow],
        *,
        console: Console,
        recommend_fn: RecommendFn,
        report_fn: ReportFn | None = None,
        input_fn: Callable[[str], str] | None = None,
        save_path: Path | None = None,
        yahoo_pick_queue: queue.Queue[YahooDraftPick] | None = None,
        team_map: dict[str, int] | None = None,
    ) -> None:
        self.engine = engine
        self.players = players
        self.console = console
        self.recommend_fn = recommend_fn
        self.report_fn = report_fn
        self.input_fn = input_fn or (lambda prompt: input(prompt))
        self.save_path = save_path
        self._unsaved = False
        self._valid_positions = set(engine.state.config.roster_slots.keys())
        self._yahoo_pick_queue = yahoo_pick_queue
        self._team_map = team_map or {}

    def run(self) -> None:
        """Main REPL loop."""
        state = self.engine.state
        self.console.print(
            f"[bold]Draft session started[/bold] — {state.config.teams} teams, {state.config.format.value} format"
        )
        self._show_status()
        self.console.print("Type [bold]help[/bold] for commands.\n")

        _input_errors = (EOFError, StopIteration)
        while True:
            self._drain_yahoo_picks()
            try:
                line = self.input_fn("draft> ")
            except _input_errors:
                break

            cmd = parse_command(line, state.config.format, self._valid_positions)
            if cmd is None:
                if line.strip():
                    self.console.print("[yellow]Unknown command.[/yellow] Type [bold]help[/bold] for commands.")
                continue

            if not self._handle_command(cmd):
                break

    def _drain_yahoo_picks(self) -> None:
        if self._yahoo_pick_queue is None:
            return

        ingested = False
        while True:
            try:
                yahoo_pick = self._yahoo_pick_queue.get_nowait()
            except queue.Empty:
                break

            result = ingest_yahoo_pick(
                self.engine.pick, set(self.engine.state.available_pool), yahoo_pick, self._team_map
            )
            if result is not None:
                self._unsaved = True
                price_str = f" for ${result.price}" if result.price is not None else ""
                self.console.print(
                    f"[cyan]Yahoo pick: {result.player_name} ({result.position}) "
                    f"— team {result.team}, pick #{result.pick_number}{price_str}[/cyan]"
                )
                ingested = True
            else:
                self.console.print(
                    f"[dim]Yahoo pick skipped: {yahoo_pick.player_name} ({yahoo_pick.yahoo_player_key})[/dim]"
                )

        if ingested:
            self._show_recommendations()

    def _handle_command(self, cmd: Command) -> bool:
        """Dispatch a command. Returns False to quit."""
        if isinstance(cmd, QuitCommand):
            return self._handle_quit()
        if isinstance(cmd, HelpCommand):
            self.console.print(_HELP_TEXT)
        elif isinstance(cmd, PickCommand):
            self._handle_pick(cmd)
        elif isinstance(cmd, UndoCommand):
            self._handle_undo()
        elif isinstance(cmd, BestCommand):
            self._handle_best(cmd)
        elif isinstance(cmd, NeedCommand):
            self._handle_need()
        elif isinstance(cmd, RosterCommand):
            self._handle_roster()
        elif isinstance(cmd, PoolCommand):
            self._handle_pool(cmd)
        elif isinstance(cmd, StatusCommand):
            self._show_status()
        elif isinstance(cmd, SaveCommand):
            self._handle_save()
        elif isinstance(cmd, ReportCommand):
            self._handle_report()
        return True

    # --- Command handlers ---

    def _handle_pick(self, cmd: PickCommand) -> None:
        state = self.engine.state
        pool = list(state.available_pool.values())

        # Resolve player name
        matches = resolve_player(cmd.query, pool)
        if not matches:
            self.console.print(f"[red]No player found matching '{cmd.query}'[/red]")
            return

        if len(matches) > 1:
            self.console.print(f"[yellow]Multiple matches for '{cmd.query}':[/yellow]")
            for i, m in enumerate(matches[:10], 1):
                self.console.print(f"  {i}. {m.player_name} ({m.position}) — ${m.value:.1f}")
            self.console.print("Please be more specific.")
            return

        player = matches[0]

        # Determine position
        position = cmd.position
        if position is None:
            position = auto_detect_position(player, self.engine.my_needs(), state.config.roster_slots)

        if position is None:
            self.console.print(
                f"[yellow]Cannot auto-detect position for {player.player_name}. "
                f"Please specify: pick {player.player_name} <position>[/yellow]"
            )
            return

        # Determine team
        team = self.engine.team_on_clock() if state.config.format == DraftFormat.SNAKE else state.config.user_team

        try:
            pick = self.engine.pick(player.player_id, team, position, price=cmd.price)
        except DraftError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        self._unsaved = True
        price_str = f" for ${pick.price}" if pick.price is not None else ""
        self.console.print(
            f"[green]Picked {pick.player_name} ({pick.position}) "
            f"— team {pick.team}, pick #{pick.pick_number}{price_str}[/green]"
        )

        # Auto-show recommendations after pick
        self._show_recommendations()

    def _handle_undo(self) -> None:
        try:
            pick = self.engine.undo()
        except DraftError as e:
            self.console.print(f"[red]{e}[/red]")
            return
        self._unsaved = True
        self.console.print(f"[yellow]Undid pick #{pick.pick_number}: {pick.player_name}[/yellow]")

    def _handle_best(self, cmd: BestCommand) -> None:
        self._show_recommendations(position=cmd.position)

    def _handle_need(self) -> None:
        needs = self.engine.my_needs()
        if not needs:
            self.console.print("All roster slots filled!")
            return
        self.console.print("[bold]Unfilled slots:[/bold]")
        for pos, count in needs.items():
            self.console.print(f"  {pos}: {count}")

    def _handle_roster(self) -> None:
        roster = self.engine.my_roster()
        if not roster:
            self.console.print("Roster is empty.")
            return
        self.console.print("[bold]Your roster:[/bold]")
        for pick in roster:
            price_str = f" (${pick.price})" if pick.price is not None else ""
            self.console.print(f"  {pick.position}: {pick.player_name}{price_str}")

    def _handle_pool(self, cmd: PoolCommand) -> None:
        available = self.engine.available(cmd.position)
        if not available:
            self.console.print("No players available" + (f" at {cmd.position}" if cmd.position else "") + ".")
            return
        shown = available[:20]
        self.console.print(f"[bold]Available players ({len(available)} total):[/bold]")
        for p in shown:
            self.console.print(f"  {p.player_name} ({p.position}) — ${p.value:.1f}")

    def _show_status(self) -> None:
        state = self.engine.state
        config = state.config
        pick_num = state.current_pick
        current_round = (pick_num - 1) // config.teams + 1

        if config.format == DraftFormat.SNAKE:
            team = self.engine.team_on_clock()
            is_user = " (your pick!)" if team == config.user_team else ""
            self.console.print(
                f"Pick #{pick_num} | Round {current_round} | "
                f"Team {team}{is_user} | {len(state.available_pool)} players available"
            )
        else:
            self.console.print(
                f"Pick #{pick_num} | Round {current_round} | "
                f"Budget: ${state.team_budgets[config.user_team]} | "
                f"{len(state.available_pool)} players available"
            )

    def _show_recommendations(self, position: str | None = None, limit: int = 5) -> None:
        recs = self.recommend_fn(self.engine.state, limit=limit)
        if position:
            recs = [r for r in recs if r.position == position]
        if not recs:
            self.console.print("No recommendations available.")
            return
        self.console.print("[bold]Recommendations:[/bold]")
        for i, rec in enumerate(recs[:limit], 1):
            self.console.print(
                f"  {i}. {rec.player_name} ({rec.position}) — ${rec.value:.1f} | score {rec.score:.2f} | {rec.reason}"
            )

    def _handle_report(self) -> None:
        if self.report_fn is None:
            self.console.print("[yellow]Report not available (no report function configured).[/yellow]")
            return
        report = self.report_fn(self.engine.state, self.players)
        self._print_report(report)

    def _print_report(self, report: DraftReport) -> None:
        c = self.console
        c.print("\n[bold]═══ Draft Report ═══[/bold]\n")

        # Roster value summary
        c.print("[bold]Roster Value[/bold]")
        c.print(f"  Total value:  {report.total_value:.1f}")
        c.print(f"  Optimal:      {report.optimal_value:.1f}")
        c.print(f"  Efficiency:   {report.value_efficiency:.1%}")
        if report.budget is not None:
            c.print(f"  Budget:       ${report.budget}")
            c.print(f"  Spent:        ${report.total_spent}")
        c.print()

        # Category standings
        if report.category_standings:
            c.print("[bold]Category Standings[/bold]")
            for s in report.category_standings:
                c.print(f"  {s.category:<6} z={s.total_z:+.2f}  rank {s.rank}/{s.teams}")
            c.print()

        # Pick grades
        if report.pick_grades:
            c.print(f"[bold]Pick Grades[/bold]  (mean: {report.mean_grade:.2f})")
            for g in report.pick_grades:
                color = "green" if g.grade >= 0.9 else "yellow" if g.grade >= 0.7 else "red"
                c.print(
                    f"  #{g.pick_number:<3} {g.player_name:<20} {g.position:<4} "
                    f"val={g.value:.1f}  best={g.best_available_value:.1f}  "
                    f"[{color}]grade={g.grade:.2f}[/{color}]"
                )
            c.print()

        # Steals
        if report.steals:
            c.print("[bold green]Steals[/bold green]")
            for s in report.steals:
                c.print(f"  #{s.pick_number:<3} {s.player_name:<20} {s.position:<4} delta=+{s.pick_delta}")
            c.print()

        # Reaches
        if report.reaches:
            c.print("[bold red]Reaches[/bold red]")
            for r in report.reaches:
                c.print(f"  #{r.pick_number:<3} {r.player_name:<20} {r.position:<4} delta={r.pick_delta}")
            c.print()

    def _handle_save(self) -> None:
        if self.save_path is None:
            self.console.print("[yellow]No save path configured. Use --resume to set a save file.[/yellow]")
            return
        save_draft(self.engine.state, self.save_path)
        self._unsaved = False
        self.console.print(f"[green]Draft saved to {self.save_path}[/green]")

    def _handle_quit(self) -> bool:
        if self._unsaved and self.save_path is not None:
            self._handle_save()
        self.console.print("Goodbye!")
        return False
