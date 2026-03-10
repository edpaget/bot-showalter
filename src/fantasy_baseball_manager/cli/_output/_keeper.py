from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console
from fantasy_baseball_manager.cli._output._draft import print_category_needs

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        AdjustedValuation,
        CategoryNeed,
        KeeperDecision,
        KeeperScenario,
        KeeperSolution,
        KeeperTradeImpact,
        LeagueKeeper,
        LeagueKeeperOverview,
        Player,
        RosterAnalysis,
        TradeEvaluation,
        TradePlayerDetail,
    )


def print_keeper_decisions(decisions: list[KeeperDecision]) -> None:
    table = Table(title="Keeper Decisions", show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Yrs", justify="right")
    table.add_column("Rec", justify="left")

    for d in decisions:
        style = "green" if d.surplus >= 0 else "red"
        cost_str = f"Rd {d.original_round} (~${d.cost:.0f})" if d.original_round is not None else f"${d.cost:.0f}"
        table.add_row(
            d.player_name,
            d.position,
            cost_str,
            f"${d.projected_value:.0f}",
            f"${d.surplus:.1f}",
            str(d.years_remaining),
            d.recommendation,
            style=style,
        )

    console.print(table)


def print_adjusted_rankings(rankings: list[AdjustedValuation], *, top: int | None = None) -> None:
    table = Table(title="Keeper-Adjusted Rankings", show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Player", justify="left")
    table.add_column("Type", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Original", justify="right")
    table.add_column("Adjusted", justify="right")
    table.add_column("Change", justify="right")

    display = rankings[:top] if top is not None else rankings
    for i, r in enumerate(display, 1):
        if abs(r.value_change) > 3:
            style = "bold green" if r.value_change > 0 else "bold red"
        elif r.value_change > 0:
            style = "green"
        elif r.value_change < 0:
            style = "red"
        else:
            style = ""
        sign = "+" if r.value_change > 0 else ""
        table.add_row(
            str(i),
            r.player_name,
            r.player_type,
            r.position,
            f"${r.original_value:.1f}",
            f"${r.adjusted_value:.1f}",
            f"{sign}${r.value_change:.1f}",
            style=style,
        )

    console.print(table)


def _trade_player_table(label: str, details: list[TradePlayerDetail]) -> Table:
    table = Table(title=label, show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Yrs", justify="right")

    for d in details:
        style = "green" if d.surplus >= 0 else "red"
        table.add_row(
            d.player_name,
            d.position,
            f"${d.cost:.0f}",
            f"${d.projected_value:.1f}",
            f"${d.surplus:.1f}",
            str(d.years_remaining),
            style=style,
        )
    return table


def print_trade_evaluation(evaluation: TradeEvaluation) -> None:
    console.print()
    console.print(_trade_player_table("You give", evaluation.team_a_gives))
    console.print()
    console.print(_trade_player_table("You receive", evaluation.team_b_gives))
    console.print()

    a_sign = "+" if evaluation.team_a_surplus_delta >= 0 else ""
    b_sign = "+" if evaluation.team_b_surplus_delta >= 0 else ""
    a_style = "green" if evaluation.team_a_surplus_delta >= 0 else "red"
    b_style = "green" if evaluation.team_b_surplus_delta >= 0 else "red"

    console.print(f"Your surplus delta: [{a_style}]{a_sign}${evaluation.team_a_surplus_delta:.1f}[/{a_style}]")
    console.print(f"Their surplus delta: [{b_style}]{b_sign}${evaluation.team_b_surplus_delta:.1f}[/{b_style}]")

    if evaluation.winner == "team_a":
        console.print("[bold green]Verdict: You win this trade[/bold green]")
    elif evaluation.winner == "team_b":
        console.print("[bold red]Verdict: They win this trade[/bold red]")
    else:
        console.print("[bold]Verdict: Even trade[/bold]")


def print_keeper_solution(solution: KeeperSolution) -> None:
    """Print the optimal keeper set with alternatives and sensitivity analysis."""
    optimal = solution.optimal

    # Optimal keeper set table
    table = Table(title="Optimal Keeper Set", show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")

    for p in optimal.players:
        cost_str = f"Rd {p.original_round} (~${p.cost:.0f})" if p.original_round is not None else f"${p.cost:.0f}"
        table.add_row(
            p.player_name,
            p.position,
            cost_str,
            f"${p.projected_value:.0f}",
            f"${p.surplus:.1f}",
        )

    console.print(table)
    console.print(
        f"Total surplus: ${optimal.total_surplus:.1f}  "
        f"Total cost: ${optimal.total_cost:.0f}  "
        f"Score: {optimal.score:.1f}"
    )
    console.print()

    # Alternatives
    if solution.alternatives:
        console.print("[bold]Alternatives[/bold]")
        for i, alt in enumerate(solution.alternatives[:3], 1):
            names = ", ".join(p.player_name for p in alt.players)
            gap = optimal.score - alt.score
            console.print(f"  Alt {i}: {names}  (score: {alt.score:.1f}, gap: -{gap:.1f})")
        console.print()

    # Sensitivity
    if solution.sensitivity:
        sens_table = Table(title="Sensitivity", show_edge=False, pad_edge=False)
        sens_table.add_column("Player", justify="left")
        sens_table.add_column("Surplus Gap", justify="right")

        for entry in solution.sensitivity:
            gap_str = f"{entry.surplus_gap:.1f}" if entry.surplus_gap != float("inf") else "inf"
            sens_table.add_row(entry.player_name, gap_str)

        console.print(sens_table)


def print_keeper_scenarios(scenarios: list[KeeperScenario]) -> None:
    """Print scenario comparison ranked by score."""
    table = Table(title="Scenario Comparison", show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Name", justify="left")
    table.add_column("Score", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Keepers", justify="left")

    for i, s in enumerate(scenarios, 1):
        names = ", ".join(p.player_name for p in s.keeper_set.players)
        delta_str = "-" if s.delta_vs_optimal == 0.0 else f"+${s.delta_vs_optimal:.1f}"
        style = "green" if i == 1 else ""
        table.add_row(
            str(i),
            s.name,
            f"{s.keeper_set.score:.1f}",
            delta_str,
            names,
            style=style,
        )

    console.print(table)


def print_keeper_trade_impact(impact: KeeperTradeImpact) -> None:
    """Print before/after optimal sets and score delta for a trade."""
    before_names = ", ".join(p.player_name for p in impact.before.optimal.players)
    after_names = ", ".join(p.player_name for p in impact.after.optimal.players)

    console.print("[bold]Before[/bold]")
    console.print(f"  Keepers: {before_names}")
    console.print(f"  Score: {impact.before.optimal.score:.1f}")
    console.print()
    console.print("[bold]After[/bold]")
    console.print(f"  Keepers: {after_names}")
    console.print(f"  Score: {impact.after.optimal.score:.1f}")
    console.print()

    color = "green" if impact.score_delta >= 0 else "red"
    delta_str = f"+${impact.score_delta:.1f}" if impact.score_delta >= 0 else f"-${abs(impact.score_delta):.1f}"
    console.print(f"Score delta: [{color}]{delta_str}[/{color}]")


def print_league_keeper_overview(overview: LeagueKeeperOverview, *, top_targets: int = 15) -> None:
    """Render league-wide keeper overview: team rankings, category comparison, and trade targets."""
    # A) Team Rankings
    team_table = Table(title="Team Keeper Rankings", show_edge=False, pad_edge=False)
    team_table.add_column("Rank", justify="right")
    team_table.add_column("Team", justify="left")
    team_table.add_column("Keepers", justify="left")
    team_table.add_column("Total", justify="right")

    for i, tp in enumerate(overview.team_projections, 1):
        keeper_strs = [f"{k.player_name} (${k.value:.0f})" for k in tp.keepers]
        keepers_text = ", ".join(keeper_strs)
        style = "bold" if tp.is_user else ""
        team_table.add_row(
            str(i),
            tp.team_name,
            keepers_text,
            f"${tp.total_value:.1f}",
            style=style,
        )

    console.print(team_table)
    console.print()

    # B) Category Comparison
    user_proj = next((tp for tp in overview.team_projections if tp.is_user), None)
    if user_proj and overview.category_names:
        cat_table = Table(title="Category Comparison", show_edge=False, pad_edge=False)
        cat_table.add_column("Category", justify="left")
        cat_table.add_column("Your Total", justify="right")
        cat_table.add_column("Rank", justify="right")
        cat_table.add_column("Leader", justify="left")

        for cat in overview.category_names:
            user_score = user_proj.category_totals.get(cat, 0.0)
            # Rank teams by this category
            scores = [(tp.team_name, tp.category_totals.get(cat, 0.0)) for tp in overview.team_projections]
            scores.sort(key=lambda x: x[1], reverse=True)
            rank = next(
                (i for i, (name, _) in enumerate(scores, 1) if name == user_proj.team_name),
                len(scores),
            )
            leader_name, leader_score = scores[0]
            n_teams = len(overview.team_projections)
            cat_table.add_row(
                cat,
                f"{user_score:.1f}",
                f"{rank}/{n_teams}",
                f"{leader_name} ({leader_score:.1f})",
            )

        console.print(cat_table)
        console.print()

    # C) Trade Targets
    if overview.trade_targets:
        target_table = Table(title="Trade Targets", show_edge=False, pad_edge=False)
        target_table.add_column("Player", justify="left")
        target_table.add_column("Pos", justify="left")
        target_table.add_column("Value", justify="right")
        target_table.add_column("Owner", justify="left")
        target_table.add_column("Rank", justify="right")

        display = overview.trade_targets[:top_targets]
        for tt in display:
            target_table.add_row(
                tt.player_name,
                tt.position,
                f"${tt.value:.1f}",
                tt.owning_team_name,
                f"#{tt.rank_on_team}",
            )

        console.print(target_table)
    else:
        console.print("[dim]No trade targets found.[/dim]")


def print_keeper_draft_needs(
    overview: LeagueKeeperOverview,
    analysis: RosterAnalysis,
    needs: list[CategoryNeed],
    num_teams: int,
) -> None:
    """Render keeper draft needs: keeper summary, category strengths, and recommendations."""
    # 1. Your Keepers table
    user_proj = next((tp for tp in overview.team_projections if tp.is_user), None)
    if user_proj and user_proj.keepers:
        keeper_table = Table(title="Your Keepers", show_edge=False, pad_edge=False)
        keeper_table.add_column("Player", justify="left")
        keeper_table.add_column("Pos", justify="left")
        keeper_table.add_column("Value", justify="right")

        for k in user_proj.keepers:
            keeper_table.add_row(k.player_name, k.position, f"${k.value:.1f}")

        console.print(keeper_table)
        console.print()
    else:
        console.print("[yellow]No keepers projected.[/yellow]\n")

    # 2. Category Strengths table
    if analysis.projections:
        strength_table = Table(title="Category Strengths", show_edge=False, pad_edge=False)
        strength_table.add_column("Category", justify="left")
        strength_table.add_column("Value", justify="right")
        strength_table.add_column("Rank", justify="right")
        strength_table.add_column("Strength", justify="left")

        strength_colors = {"strong": "green", "average": "yellow", "weak": "red"}

        for proj in analysis.projections:
            color = strength_colors.get(proj.strength, "")
            strength_table.add_row(
                proj.category,
                f"{proj.projected_value:.1f}",
                f"{proj.league_rank_estimate}/{num_teams}",
                f"[{color}]{proj.strength}[/{color}]",
            )

        console.print(strength_table)
        console.print()

    # 3. Delegate to existing category needs renderer
    print_category_needs(needs, num_teams)


def print_league_keepers(keepers: list[LeagueKeeper], players: list[Player]) -> None:
    """Print league keepers grouped by team, showing player name, position, and cost."""
    if not keepers:
        console.print("[dim]No league keepers found.[/dim]")
        return

    player_lookup: dict[int, Player] = {p.id: p for p in players if p.id is not None}

    # Group by team
    by_team: dict[str, list[LeagueKeeper]] = {}
    for k in keepers:
        by_team.setdefault(k.team_name, []).append(k)

    table = Table(title="League Keepers", show_edge=False, pad_edge=False)
    table.add_column("Team", justify="left")
    table.add_column("Player", justify="left")
    table.add_column("Cost", justify="right")

    for team_name in sorted(by_team):
        team_keepers = by_team[team_name]
        for i, k in enumerate(team_keepers):
            player = player_lookup.get(k.player_id)
            name = f"{player.name_first} {player.name_last}" if player else f"ID:{k.player_id}"
            cost_str = f"${k.cost:.0f}" if k.cost is not None else "-"
            # Only show team name on first row for the team
            table.add_row(
                team_name if i == 0 else "",
                name,
                cost_str,
            )

    console.print(table)
