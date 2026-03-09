import json
import queue
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.domain import DraftSessionPick, DraftSessionRecord
from fantasy_baseball_manager.services.adp_arbitrage import detect_falling_players, detect_reaches
from fantasy_baseball_manager.services.category_tracker import analyze_roster, identify_needs
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftError,
    DraftFormat,
    DraftPick,
    DraftState,
)
from fantasy_baseball_manager.services.draft_translation import ingest_yahoo_pick
from fantasy_baseball_manager.services.opponent_model import assess_threats
from fantasy_baseball_manager.services.player_resolver import resolve_player

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from fantasy_baseball_manager.domain import (
        DraftBoardRow,
        DraftReport,
        LeagueSettings,
        Projection,
        Recommendation,
        YahooDraftPick,
    )
    from fantasy_baseball_manager.repos import DraftSessionRepo
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
class BalanceCommand: ...


@dataclass(frozen=True)
class NeedsCommand: ...


@dataclass(frozen=True)
class FallsCommand:
    position: str | None = None
    threshold: int | None = None


@dataclass(frozen=True)
class ReachesCommand: ...


@dataclass(frozen=True)
class ThreatsCommand: ...


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
    | BalanceCommand
    | NeedsCommand
    | FallsCommand
    | ReachesCommand
    | ThreatsCommand
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
    if verb == "needs":
        return NeedsCommand()
    if verb == "balance":
        return BalanceCommand()
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
    if verb == "falls":
        return _parse_falls(args, valid_positions)
    if verb == "reaches":
        return ReachesCommand()
    if verb == "threats":
        return ThreatsCommand()
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


def _parse_falls(
    args: list[str],
    valid_positions: set[str],
) -> FallsCommand:
    """Parse the arguments after 'falls', extracting optional position and threshold."""
    position: str | None = None
    threshold: int | None = None

    i = 0
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args) and args[i + 1].isdigit():
            threshold = int(args[i + 1])
            i += 2
        else:
            pos = _match_position(args[i], valid_positions)
            if pos is not None:
                position = pos
            i += 1

    return FallsCommand(position=position, threshold=threshold)


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


def load_draft_from_db(
    session_id: int,
    players: list[DraftBoardRow],
    repo: DraftSessionRepo,
) -> DraftEngine:
    """Load a draft from the database by replaying picks."""
    record = repo.load_session(session_id)
    if record is None:
        msg = f"Draft session {session_id} not found"
        raise ValueError(msg)

    config = DraftConfig(
        teams=record.teams,
        roster_slots=record.roster_slots,
        format=DraftFormat(record.format),
        user_team=record.user_team,
        season=record.season,
        budget=record.budget,
    )

    engine = DraftEngine()
    engine.start(players, config)

    for pick in repo.load_picks(session_id):
        engine.pick(
            player_id=pick.player_id,
            team=pick.team,
            position=pick.position,
            price=pick.price,
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
  balance                           — Show category projections and strengths
  needs                             — Show category needs and recommendations
  falls [position] [--threshold N]  — Show falling players (ADP arbitrage)
  reaches                           — Show reach picks from the draft log
  threats                           — Show players at risk of being taken
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
        id_aliases: dict[int, int] | None = None,
        projections: list[Projection] | None = None,
        league: LeagueSettings | None = None,
        session_repo: DraftSessionRepo | None = None,
        session_id: int | None = None,
        league_name: str = "",
    ) -> None:
        self.engine = engine
        self.players = players
        self.console = console
        self.recommend_fn = recommend_fn
        self.report_fn = report_fn
        self.input_fn = input_fn or (lambda prompt: input(prompt))
        self.save_path = save_path
        self._unsaved = False
        self._valid_positions = set(engine.state.config.roster_slots.keys()) - {"BN"}
        self._yahoo_pick_queue = yahoo_pick_queue
        self._team_map = team_map or {}
        self._id_aliases = id_aliases
        self._projections = projections
        self._league = league
        self._session_repo = session_repo
        self._session_id = session_id
        self._league_name = league_name

    def run(self) -> None:
        """Main REPL loop."""
        self._maybe_create_session()
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

    def _maybe_create_session(self) -> None:
        """Create a new DB session if a repo is injected and no session_id was provided."""
        if self._session_repo is None or self._session_id is not None:
            return
        config = self.engine.state.config
        now = datetime.now(tz=UTC).isoformat()
        record = DraftSessionRecord(
            league=self._league_name,
            season=config.season,
            teams=config.teams,
            format=config.format.value,
            user_team=config.user_team,
            roster_slots=dict(config.roster_slots),
            budget=config.budget,
            status="in_progress",
            created_at=now,
            updated_at=now,
        )
        self._session_id = self._session_repo.create_session(record)

    def _persist_pick(self, pick: DraftPick) -> None:
        """Write a pick to the DB if a repo is available."""
        if self._session_repo is None or self._session_id is None:
            return
        now = datetime.now(tz=UTC).isoformat()
        db_pick = DraftSessionPick(
            session_id=self._session_id,
            pick_number=pick.pick_number,
            team=pick.team,
            player_id=pick.player_id,
            player_name=pick.player_name,
            position=pick.position,
            price=pick.price,
        )
        self._session_repo.save_pick(db_pick)
        self._session_repo.update_timestamp(self._session_id, now)

    def _persist_undo(self, pick: DraftPick) -> None:
        """Delete an undone pick from the DB if a repo is available."""
        if self._session_repo is None or self._session_id is None:
            return
        now = datetime.now(tz=UTC).isoformat()
        self._session_repo.delete_pick(self._session_id, pick.pick_number)
        self._session_repo.update_timestamp(self._session_id, now)

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
                self.engine.pick,
                set(self.engine.state.available_pool),
                yahoo_pick,
                self._team_map,
                id_aliases=self._id_aliases,
                roster_slots=self.engine.state.config.roster_slots,
                team_rosters=self.engine.state.team_rosters,
            )
            if result is not None:
                self._persist_pick(result)
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
            self._show_falling_alerts()

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
        elif isinstance(cmd, BalanceCommand):
            self._handle_balance()
        elif isinstance(cmd, NeedsCommand):
            self._handle_needs()
        elif isinstance(cmd, FallsCommand):
            self._handle_falls(cmd)
        elif isinstance(cmd, ReachesCommand):
            self._handle_reaches()
        elif isinstance(cmd, ThreatsCommand):
            self._handle_threats()
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

        self._persist_pick(pick)
        self._unsaved = True
        price_str = f" for ${pick.price}" if pick.price is not None else ""
        self.console.print(
            f"[green]Picked {pick.player_name} ({pick.position}) "
            f"— team {pick.team}, pick #{pick.pick_number}{price_str}[/green]"
        )

        # Show compact category summary after pick
        self._show_category_summary()
        # Auto-show recommendations after pick
        self._show_recommendations()
        # Show falling player alerts
        self._show_falling_alerts()

    def _handle_undo(self) -> None:
        try:
            pick = self.engine.undo()
        except DraftError as e:
            self.console.print(f"[red]{e}[/red]")
            return
        self._persist_undo(pick)
        self._unsaved = True
        self.console.print(f"[yellow]Undid pick #{pick.pick_number}: {pick.player_name}[/yellow]")

    def _handle_best(self, cmd: BestCommand) -> None:  # pragma: no cover
        self._show_recommendations(position=cmd.position)

    def _handle_need(self) -> None:  # pragma: no cover
        needs = self.engine.my_needs()
        if not needs:
            self.console.print("All roster slots filled!")
            return
        self.console.print("[bold]Unfilled slots:[/bold]")
        for pos, count in needs.items():
            self.console.print(f"  {pos}: {count}")

    def _handle_roster(self) -> None:  # pragma: no cover
        roster = self.engine.my_roster()
        if not roster:
            self.console.print("Roster is empty.")
            return
        self.console.print("[bold]Your roster:[/bold]")
        for pick in roster:
            price_str = f" (${pick.price})" if pick.price is not None else ""
            self.console.print(f"  {pick.position}: {pick.player_name}{price_str}")

    def _handle_pool(self, cmd: PoolCommand) -> None:  # pragma: no cover
        available = self.engine.available(cmd.position)
        if not available:
            self.console.print("No players available" + (f" at {cmd.position}" if cmd.position else "") + ".")
            return
        shown = available[:20]
        self.console.print(f"[bold]Available players ({len(available)} total):[/bold]")
        for p in shown:
            self.console.print(f"  {p.player_name} ({p.position}) — ${p.value:.1f}")

    def _show_status(self) -> None:  # pragma: no cover
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

    def _show_recommendations(self, position: str | None = None, limit: int = 5) -> None:  # pragma: no cover
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

    @property
    def _has_category_tracking(self) -> bool:
        return self._projections is not None and self._league is not None

    def _handle_balance(self) -> None:  # pragma: no cover
        if not self._has_category_tracking:
            self.console.print("[yellow]Category balance not available (no projections loaded).[/yellow]")
            return

        assert self._projections is not None  # noqa: S101
        assert self._league is not None  # noqa: S101

        roster = self.engine.my_roster()
        if not roster:
            self.console.print("No picks yet — roster is empty.")
            return

        roster_ids = [p.player_id for p in roster]
        analysis = analyze_roster(roster_ids, self._projections, self._league)

        self.console.print("[bold]Category Balance:[/bold]")
        for proj in analysis.projections:
            color = "green" if proj.strength == "strong" else "red" if proj.strength == "weak" else "yellow"
            self.console.print(
                f"  {proj.category:<6} {proj.projected_value:>8.2f}  "
                f"rank {proj.league_rank_estimate}/{self._league.teams}  "
                f"[{color}]{proj.strength}[/{color}]"
            )

    def _handle_needs(self) -> None:  # pragma: no cover
        if not self._has_category_tracking:
            self.console.print("[yellow]Category needs not available (no projections loaded).[/yellow]")
            return

        assert self._projections is not None  # noqa: S101
        assert self._league is not None  # noqa: S101

        roster = self.engine.my_roster()
        if not roster:
            self.console.print("No picks yet — roster is empty.")
            return

        roster_ids = [p.player_id for p in roster]
        available_ids = list(self.engine.state.available_pool.keys())
        player_names = {p.player_id: p.player_name for p in self.players}
        needs = identify_needs(
            roster_ids,
            available_ids,
            self._projections,
            self._league,
            player_names=player_names,
            top_n=3,
        )

        if not needs:
            self.console.print("No weak categories — roster is well-balanced!")
            return

        self.console.print("[bold]Category Needs:[/bold]")
        for need in needs:
            self.console.print(f"  [red]{need.category}[/red] — rank {need.current_rank}/{self._league.teams}")
            for rec in need.best_available:
                tradeoff = (
                    f" [yellow](hurts {', '.join(rec.tradeoff_categories)})[/yellow]" if rec.tradeoff_categories else ""
                )
                self.console.print(f"    {rec.player_name} (+{rec.category_impact:.1f}){tradeoff}")

    def _handle_falls(self, cmd: FallsCommand) -> None:
        state = self.engine.state
        available = list(state.available_pool.values())
        threshold = cmd.threshold if cmd.threshold is not None else 10
        falling = detect_falling_players(state.current_pick, available, threshold=threshold, limit=20)
        if cmd.position:
            falling = [f for f in falling if f.position == cmd.position]
        if not falling:
            pos_msg = f" at {cmd.position}" if cmd.position else ""
            self.console.print(f"No falling players detected{pos_msg}.")
            return
        self.console.print("[bold]Falling Players (ADP Arbitrage):[/bold]")
        for i, f in enumerate(falling, 1):
            self.console.print(
                f"  {i:>2}. {f.player_name:<22} {f.position:<4} "
                f"ADP {f.adp:>5.1f}  slip +{f.picks_past_adp:.0f}  "
                f"val ${f.value:.1f}  score {f.arbitrage_score:.1f}"
            )

    def _handle_reaches(self) -> None:
        state = self.engine.state
        adp_lookup: dict[int, float] = {}
        for p in self.players:
            if p.adp_overall is not None:
                adp_lookup[p.player_id] = p.adp_overall
        reaches = detect_reaches(state.picks, adp_lookup, threshold=10)
        if not reaches:
            self.console.print("No reach picks detected.")
            return
        self.console.print("[bold]Reach Picks:[/bold]")
        for r in reaches:
            self.console.print(
                f"  {r.player_name:<22} {r.position:<4} "
                f"ADP {r.adp:>5.1f}  picked #{r.pick_number}  "
                f"+{r.picks_ahead_of_adp:.0f} ahead  team {r.drafter_team}"
            )

    def _show_falling_alerts(self) -> None:
        """Show brief alerts for significant fallers after a pick."""
        state = self.engine.state
        available = list(state.available_pool.values())
        falling = detect_falling_players(state.current_pick, available, threshold=20, limit=20)
        # Filter to high-value players only (top 50 by value rank)
        alerts = [f for f in falling if f.value_rank <= 50][:3]
        for f in alerts:
            self.console.print(
                f"[bold yellow]⚡ Falling: {f.player_name} ({f.position}) "
                f"— ADP {f.adp:.0f}, now pick {f.current_pick} (+{f.picks_past_adp:.0f})[/bold yellow]"
            )

    def _handle_threats(self) -> None:  # pragma: no cover
        if self._league is None:
            self.console.print("[yellow]Threat assessment not available (no league settings loaded).[/yellow]")
            return

        threats = assess_threats(self.engine.state, self._league)
        if not threats:
            self.console.print("No threats detected — all targets look safe.")
            return

        self.console.print("[bold]Threat Assessment:[/bold]")
        for t in threats:
            color = "red" if t.threat_level == "likely-gone" else "yellow" if t.threat_level == "at-risk" else "green"
            adp_str = f"ADP {t.adp:.0f}" if t.adp is not None else "no ADP"
            self.console.print(
                f"  [{color}]{t.threat_level:>11}[/{color}]  "
                f"{t.player_name} ({t.position}) — ${t.value:.1f} | "
                f"{adp_str} | {t.teams_needing_position} teams need {t.position}"
            )

    def _show_category_summary(self) -> None:  # pragma: no cover
        """Show a one-line compact category summary after a pick."""
        if not self._has_category_tracking:
            return

        assert self._projections is not None  # noqa: S101
        assert self._league is not None  # noqa: S101

        roster = self.engine.my_roster()
        if not roster:
            return

        roster_ids = [p.player_id for p in roster]
        analysis = analyze_roster(roster_ids, self._projections, self._league)

        weak = [p.category.upper() for p in analysis.projections if p.strength == "weak"]
        strong = [p.category.upper() for p in analysis.projections if p.strength == "strong"]

        parts: list[str] = []
        if weak:
            parts.append(f"Weak: {', '.join(weak)}")
        if strong:
            parts.append(f"Strong: {', '.join(strong)}")

        if parts:
            self.console.print(f"[dim]{' | '.join(parts)}[/dim]")

    def _handle_report(self) -> None:  # pragma: no cover
        if self.report_fn is None:
            self.console.print("[yellow]Report not available (no report function configured).[/yellow]")
            return
        report = self.report_fn(self.engine.state, self.players)
        self._print_report(report)

    def _print_report(self, report: DraftReport) -> None:  # pragma: no cover
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

    def _handle_save(self) -> None:  # pragma: no cover
        if self.save_path is None:
            self.console.print("[yellow]No save path configured. Use --resume to set a save file.[/yellow]")
            return
        save_draft(self.engine.state, self.save_path)
        self._unsaved = False
        self.console.print(f"[green]Draft saved to {self.save_path}[/green]")

    def _handle_quit(self) -> bool:  # pragma: no cover
        if self._unsaved and self.save_path is not None:
            self._handle_save()
        if self._session_repo is not None and self._session_id is not None:
            self._session_repo.update_status(self._session_id, "complete")
        self.console.print("Goodbye!")
        return False
