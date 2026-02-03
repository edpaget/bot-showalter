from __future__ import annotations

from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import TYPE_CHECKING, Annotated

import typer
import yaml

from fantasy_baseball_manager.draft.cli import build_projections_and_positions
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue

keeper_app = typer.Typer(help="Keeper analysis commands.")


def _load_keepers_file(path: Path) -> set[str]:
    """Load keepers YAML and return a flat set of all other teams' keeper player IDs."""
    data = yaml.safe_load(path.read_text())
    teams_raw = data.get("teams", {})
    keeper_ids: set[str] = set()
    if isinstance(teams_raw, dict):
        for team_info in teams_raw.values():
            if isinstance(team_info, dict):
                keepers_raw = team_info.get("keepers", [])
                if isinstance(keepers_raw, (list, tuple)):
                    keeper_ids.update(str(k) for k in keepers_raw)
    return keeper_ids


def _build_candidates(
    candidate_ids: list[str],
    all_player_values: list[PlayerValue],
    player_positions: dict[tuple[str, str], tuple[str, ...]],
) -> list[KeeperCandidate]:
    """Build KeeperCandidate list from IDs, matching against player values and positions."""
    pv_by_id: dict[str, PlayerValue] = {}
    for pv in all_player_values:
        # Keep the highest-value entry per player_id (a player may appear as both B and P)
        if pv.player_id not in pv_by_id or pv.total_value > pv_by_id[pv.player_id].total_value:
            pv_by_id[pv.player_id] = pv

    candidates: list[KeeperCandidate] = []
    for cid in candidate_ids:
        pv = pv_by_id.get(cid)
        if pv is None:
            typer.echo(f"Unknown candidate ID: {cid}", err=True)
            raise typer.Exit(code=1)

        # Collect positions from composite keys
        positions: list[str] = []
        for (pid, _), pos in player_positions.items():
            if pid == cid:
                positions.extend(pos)
        eligible = tuple(dict.fromkeys(positions))  # deduplicate, preserve order

        candidates.append(
            KeeperCandidate(
                player_id=cid,
                name=pv.name,
                player_value=pv,
                eligible_positions=eligible,
            )
        )

    return candidates


@keeper_app.command(name="rank")
def keeper_rank(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[
        Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")
    ] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
) -> None:
    """Rank keeper candidates by surplus value over draft replacement level."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    if not candidates.strip():
        typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...'", err=True)
        raise typer.Exit(code=1)

    candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    other_keepers: set[str] = set()
    if keepers_file:
        other_keepers = _load_keepers_file(keepers_file)

    candidate_list = _build_candidates(candidate_ids, all_values, composite_positions)

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    ranked = surplus_calc.rank_candidates(candidate_list, all_values, other_keepers)

    _display_table(ranked, year, "Keeper Candidates Ranked by Surplus Value")


@keeper_app.command(name="optimize")
def keeper_optimize(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[
        Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")
    ] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
) -> None:
    """Find the optimal keeper combination that maximizes total surplus."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    if not candidates.strip():
        typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...'", err=True)
        raise typer.Exit(code=1)

    candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    other_keepers: set[str] = set()
    if keepers_file:
        other_keepers = _load_keepers_file(keepers_file)

    candidate_list = _build_candidates(candidate_ids, all_values, composite_positions)

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    result = surplus_calc.find_optimal_keepers(candidate_list, all_values, other_keepers)

    # Display recommended keepers
    lines: list[str] = []
    lines.append(f"\nOptimal Keepers for {year} (Total Surplus: {result.total_surplus:.1f}):")
    lines.append("")
    _append_table_rows(lines, list(result.keepers))

    # Display all candidates
    lines.append("")
    lines.append("All Candidates:")
    lines.append("")
    _append_table_rows(lines, list(result.all_candidates))

    typer.echo("\n".join(lines))


def _display_table(ranked: list, year: int, title: str) -> None:
    lines: list[str] = []
    lines.append(f"\n{title} ({year}):")
    lines.append("")
    _append_table_rows(lines, ranked)
    typer.echo("\n".join(lines))


def _append_table_rows(lines: list[str], rows: list) -> None:
    header = f"{'Rk':>4} {'Name':<25} {'Pos':<12} {'Value':>7} {'Repl':>7} {'Surplus':>8} {'Slot':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, ks in enumerate(rows, start=1):
        pos_str = "/".join(ks.eligible_positions) if ks.eligible_positions else "-"
        lines.append(
            f"{i:>4} {ks.name:<25} {pos_str:<12}"
            f" {ks.player_value:>7.1f} {ks.replacement_value:>7.1f}"
            f" {ks.surplus_value:>8.1f} {ks.assigned_slot:>5}"
        )
