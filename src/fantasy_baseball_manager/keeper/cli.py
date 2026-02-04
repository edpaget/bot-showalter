from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import Annotated

import typer
from rich.console import Console

from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.keeper.command_helpers import (
    build_candidates,
    load_keepers_file,
    resolve_league_inputs,
    resolve_yahoo_inputs,
)
from fantasy_baseball_manager.keeper.display import display_table, print_keeper_table
from fantasy_baseball_manager.keeper.models import TeamKeeperResult
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator
from fantasy_baseball_manager.services import set_container
from fantasy_baseball_manager.shared.orchestration import build_projections_and_positions

logger = logging.getLogger(__name__)

console = Console()

keeper_app = typer.Typer(help="Keeper analysis commands.")

__all__ = ["keeper_app", "keeper_league", "keeper_optimize", "keeper_rank", "set_container"]


@keeper_app.command(name="rank")
def keeper_rank(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    yahoo: Annotated[bool, typer.Option("--yahoo", help="Fetch candidates from Yahoo roster.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data.")] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Rank keeper candidates by surplus value over draft replacement level."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    yahoo_positions: dict[tuple[str, str], tuple[str, ...]] | None = None
    position_types: list[str] | None = None

    if yahoo:
        candidate_ids, other_keepers, teams, yahoo_positions, position_types = resolve_yahoo_inputs(
            no_cache,
            league_id,
            season,
            candidates,
            keepers_file,
            teams,
        )
    else:
        if not candidates.strip():
            typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...' or --yahoo", err=True)
            raise typer.Exit(code=1)
        candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]
        other_keepers = set()
        if keepers_file:
            other_keepers = load_keepers_file(keepers_file)

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    candidate_list = build_candidates(
        candidate_ids,
        all_values,
        composite_positions,
        yahoo_positions,
        candidate_position_types=position_types,
    )

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    ranked = surplus_calc.rank_candidates(candidate_list, all_values, other_keepers)

    display_table(ranked, year, "Keeper Candidates Ranked by Surplus Value")


@keeper_app.command(name="optimize")
def keeper_optimize(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    yahoo: Annotated[bool, typer.Option("--yahoo", help="Fetch candidates from Yahoo roster.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data.")] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Find the optimal keeper combination that maximizes total surplus."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    yahoo_positions: dict[tuple[str, str], tuple[str, ...]] | None = None
    position_types: list[str] | None = None

    if yahoo:
        candidate_ids, other_keepers, teams, yahoo_positions, position_types = resolve_yahoo_inputs(
            no_cache,
            league_id,
            season,
            candidates,
            keepers_file,
            teams,
        )
    else:
        if not candidates.strip():
            typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...' or --yahoo", err=True)
            raise typer.Exit(code=1)
        candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]
        other_keepers = set()
        if keepers_file:
            other_keepers = load_keepers_file(keepers_file)

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    candidate_list = build_candidates(
        candidate_ids,
        all_values,
        composite_positions,
        yahoo_positions,
        candidate_position_types=position_types,
    )

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    result = surplus_calc.find_optimal_keepers(candidate_list, all_values, other_keepers)

    # Display recommended keepers
    console.print(f"\n[bold]Optimal Keepers for {year} (Total Surplus: {result.total_surplus:.1f}):[/bold]\n")
    print_keeper_table(list(result.keepers))

    # Display all candidates
    console.print("\n[bold]All Candidates:[/bold]\n")
    print_keeper_table(list(result.all_candidates))


@keeper_app.command(name="league")
def keeper_league(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    draft_order: Annotated[
        str | None, typer.Option("--draft-order", help="Comma-separated team keys defining pick order.")
    ] = None,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data.")] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Compute optimal keepers for every team in the league."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    league_data, num_teams = resolve_league_inputs(
        no_cache,
        league_id,
        season,
        teams,
    )

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    # Build draft order mapping: team_key -> pick position (1-based)
    team_pick_order: dict[str, int] = {}
    if draft_order:
        order_keys = [k.strip() for k in draft_order.split(",") if k.strip()]
        for i, key in enumerate(order_keys, start=1):
            team_pick_order[key] = i
    else:
        for i, team_info in enumerate(league_data.teams, start=1):
            team_pick_order[team_info.team_key] = i

    results: list[TeamKeeperResult] = []
    for team_info in league_data.teams:
        # Build candidates for this team
        candidate_ids = list(team_info.candidate_ids)
        yahoo_positions = dict(team_info.candidate_positions)
        team_position_types = list(team_info.candidate_position_types)

        # Skip teams with no candidates
        if not candidate_ids:
            continue

        # Collect all other teams' player IDs as other_keepers
        other_keepers: set[str] = set()
        for other_team in league_data.teams:
            if other_team.team_key != team_info.team_key:
                other_keepers.update(other_team.candidate_ids)

        candidate_list = build_candidates(
            candidate_ids,
            all_values,
            composite_positions,
            yahoo_positions,
            candidate_position_types=team_position_types,
            strict=False,
        )

        # Skip teams where no candidates had projections
        if not candidate_list:
            continue

        pick_position = team_pick_order.get(team_info.team_key, 1)
        calc = DraftPoolReplacementCalculator(user_pick_position=pick_position)
        surplus_calc = SurplusCalculator(calc, num_teams=num_teams, num_keeper_slots=keeper_slots)
        recommendation = surplus_calc.find_optimal_keepers(candidate_list, all_values, other_keepers)

        results.append(
            TeamKeeperResult(
                team_key=team_info.team_key,
                team_name=team_info.team_name,
                recommendation=recommendation,
            )
        )

    # Display results
    for team_result in results:
        pick = team_pick_order.get(team_result.team_key, 0)
        rec = team_result.recommendation
        console.print(
            f"\n[bold]=== {team_result.team_name} (Pick #{pick}) "
            f"— Total Surplus: {rec.total_surplus:.1f} ===[/bold]\n"
        )
        print_keeper_table(list(rec.keepers))
