from collections import defaultdict
from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from rich.console import Console

    from fantasy_baseball_manager.domain import (
        ADP,
        CascadeResult,
        CategoryNeed,
        DraftBoard,
        DraftReport,
        LeagueSettings,
        PickTradeEvaluation,
        PickValue,
        PickValueCurve,
        PlayerTier,
        PositionScarcity,
        TierSummaryReport,
    )


def print_draft_board(board: DraftBoard) -> None:
    """Print a draft board as a Rich table."""
    if not board.rows:
        console.print("No players on draft board.")
        return

    has_age = any(r.age is not None for r in board.rows)
    has_bt = any(r.bats_throws is not None for r in board.rows)
    has_tier = any(r.tier is not None for r in board.rows)
    has_adp = any(r.adp_overall is not None for r in board.rows)

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Player")
    if has_age:
        table.add_column("Age", justify="right")
    if has_bt:
        table.add_column("B/T")
    table.add_column("Type")
    table.add_column("Pos")
    table.add_column("Value", justify="right")
    if has_tier:
        table.add_column("Tier", justify="right")
    for cat in board.batting_categories:
        table.add_column(cat, justify="right")
    for cat in board.pitching_categories:
        table.add_column(cat, justify="right")
    if has_adp:
        table.add_column("ADP", justify="right")
        table.add_column("ADPRk", justify="right")
        table.add_column("Delta", justify="right")

    for row in board.rows:
        is_pitcher = row.player_type == "pitcher"
        cells: list[str] = [
            str(row.rank),
            row.player_name,
        ]
        if has_age:
            cells.append(str(row.age) if row.age is not None else "")
        if has_bt:
            cells.append(row.bats_throws if row.bats_throws is not None else "")
        cells.extend(
            [
                row.player_type,
                row.position,
                f"${row.value:.1f}",
            ]
        )
        if has_tier:
            cells.append(str(row.tier) if row.tier is not None else "")
        for cat in board.batting_categories:
            if is_pitcher:
                cells.append("")
            else:
                z = row.category_z_scores.get(cat)
                cells.append(f"{z:.2f}" if z is not None else "")
        for cat in board.pitching_categories:
            if not is_pitcher:
                cells.append("")
            else:
                z = row.category_z_scores.get(cat)
                cells.append(f"{z:.2f}" if z is not None else "")
        if has_adp:
            cells.append(f"{row.adp_overall:.1f}" if row.adp_overall is not None else "")
            cells.append(str(row.adp_rank) if row.adp_rank is not None else "")
            if row.adp_delta is not None:
                if row.adp_delta > 0:
                    cells.append(f"[green]+{row.adp_delta}[/green]")
                elif row.adp_delta < 0:
                    cells.append(f"[red]{row.adp_delta}[/red]")
                else:
                    cells.append("0")
            else:
                cells.append("")
        table.add_row(*cells)

    console.print(table)


def print_draft_tiers(tiers: list[PlayerTier], adp_by_player: dict[int, ADP] | None = None) -> None:
    """Print tier assignments as position-grouped Rich tables with tier separators."""
    if not tiers:
        console.print("No tier data found.")
        return

    # Group by position (already sorted by position then rank from generate_tiers)
    by_position: dict[str, list[PlayerTier]] = defaultdict(list)
    for t in tiers:
        by_position[t.position].append(t)

    has_adp = adp_by_player is not None and len(adp_by_player) > 0

    for position, players in by_position.items():
        console.print(f"[bold]\u2500\u2500 {position} \u2500\u2500[/bold]")

        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Tier", justify="right")
        table.add_column("Rank", justify="right")
        table.add_column("Player")
        table.add_column("Value", justify="right")
        if has_adp:
            table.add_column("ADP", justify="right")

        prev_tier: int | None = None
        for pt in players:
            if prev_tier is not None and pt.tier != prev_tier:
                table.add_section()

            row: list[str] = [
                str(pt.tier),
                str(pt.rank),
                pt.player_name,
                f"${pt.value:.1f}",
            ]
            if has_adp:
                assert adp_by_player is not None  # noqa: S101 - type narrowing
                adp = adp_by_player.get(pt.player_id)
                row.append(f"{adp.overall_pick:.1f}" if adp else "")

            table.add_row(*row)
            prev_tier = pt.tier

        console.print(table)
        console.print()


def print_draft_report(report: DraftReport, out: Console | None = None) -> None:
    """Print a post-draft analysis report."""
    c = out or console
    c.print("\n[bold]═══ Draft Report ═══[/bold]\n")

    # Roster value summary
    c.print("[bold]Roster Value[/bold]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Total value", f"{report.total_value:.1f}")
    table.add_row("Optimal", f"{report.optimal_value:.1f}")
    table.add_row("Efficiency", f"{report.value_efficiency:.1%}")
    if report.budget is not None:
        table.add_row("Budget", f"${report.budget}")
        table.add_row("Spent", f"${report.total_spent}")
    c.print(table)
    c.print()

    # Category standings
    if report.category_standings:
        c.print("[bold]Category Standings[/bold]")
        cat_table = Table(show_edge=False, pad_edge=False)
        cat_table.add_column("Category", justify="left")
        cat_table.add_column("Z-Total", justify="right")
        cat_table.add_column("Rank", justify="right")
        for s in report.category_standings:
            cat_table.add_row(s.category, f"{s.total_z:+.2f}", f"{s.rank}/{s.teams}")
        c.print(cat_table)
        c.print()

    # Pick grades
    if report.pick_grades:
        c.print(f"[bold]Pick Grades[/bold]  (mean: {report.mean_grade:.2f})")
        grade_table = Table(show_edge=False, pad_edge=False)
        grade_table.add_column("Pick#", justify="right")
        grade_table.add_column("Player", justify="left")
        grade_table.add_column("Pos", justify="left")
        grade_table.add_column("Value", justify="right")
        grade_table.add_column("Best Avail", justify="right")
        grade_table.add_column("Grade", justify="right")
        for g in report.pick_grades:
            color = "green" if g.grade >= 0.9 else "yellow" if g.grade >= 0.7 else "red"
            grade_table.add_row(
                str(g.pick_number),
                g.player_name,
                g.position,
                f"{g.value:.1f}",
                f"{g.best_available_value:.1f}",
                f"[{color}]{g.grade:.2f}[/{color}]",
            )
        c.print(grade_table)
        c.print()

    # Steals
    if report.steals:
        c.print("[bold green]Steals[/bold green]")
        steal_table = Table(show_edge=False, pad_edge=False)
        steal_table.add_column("Pick#", justify="right")
        steal_table.add_column("Player", justify="left")
        steal_table.add_column("Pos", justify="left")
        steal_table.add_column("Delta", justify="right")
        for s in report.steals:
            steal_table.add_row(str(s.pick_number), s.player_name, s.position, f"+{s.pick_delta}")
        c.print(steal_table)
        c.print()

    # Reaches
    if report.reaches:
        c.print("[bold red]Reaches[/bold red]")
        reach_table = Table(show_edge=False, pad_edge=False)
        reach_table.add_column("Pick#", justify="right")
        reach_table.add_column("Player", justify="left")
        reach_table.add_column("Pos", justify="left")
        reach_table.add_column("Delta", justify="right")
        for r in report.reaches:
            reach_table.add_row(str(r.pick_number), r.player_name, r.position, str(r.pick_delta))
        c.print(reach_table)
        c.print()


def print_tier_summary(report: TierSummaryReport) -> None:
    """Print a cross-position tier summary matrix."""
    if not report.entries:
        console.print("No tier data found.")
        return

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Position")
    for t in range(1, report.max_tier + 1):
        table.add_column(f"Tier {t}", justify="center")

    # Index entries by (position, tier)
    entry_map: dict[tuple[str, int], str] = {}
    tier_totals: dict[int, int] = defaultdict(int)
    for entry in report.entries:
        cell = f"{entry.count} (avg ${entry.avg_value:.1f})"
        entry_map[(entry.position, entry.tier)] = cell
        tier_totals[entry.tier] += entry.count

    for position in report.positions:
        cells: list[str] = [position]
        for t in range(1, report.max_tier + 1):
            cells.append(entry_map.get((position, t), ""))
        table.add_row(*cells)

    # Totals footer row
    table.add_section()
    footer: list[str] = ["[bold]Total[/bold]"]
    for t in range(1, report.max_tier + 1):
        footer.append(f"[bold]{tier_totals.get(t, 0)}[/bold]")
    table.add_row(*footer)

    console.print(table)


def print_category_needs(needs: list[CategoryNeed], num_teams: int) -> None:
    """Print category needs analysis with recommendations."""
    if not needs:
        console.print("[green]No weak categories identified.[/green]")
        return

    console.print("\n[bold]═══ Category Needs ═══[/bold]\n")

    for need in needs:
        console.print(
            f"[bold red]{need.category.upper()}[/bold red]"
            f"  rank {need.current_rank}/{num_teams} → target {need.target_rank}/{num_teams}"
        )

        if not need.best_available:
            console.print("  No available players found.\n")
            continue

        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Player", justify="left")
        table.add_column("Impact", justify="right")
        table.add_column("Tradeoffs", justify="left")

        for rec in need.best_available:
            impact_color = "green" if rec.category_impact > 0 else "red"
            impact_str = f"[{impact_color}]{rec.category_impact:+.2f}[/{impact_color}]"
            tradeoff_str = ", ".join(rec.tradeoff_categories) if rec.tradeoff_categories else "-"
            if rec.tradeoff_categories:
                tradeoff_str = f"[yellow]{tradeoff_str}[/yellow]"
            table.add_row(rec.player_name or str(rec.player_id), impact_str, tradeoff_str)

        console.print(table)
        console.print()


def print_pick_value_curve(curve: PickValueCurve) -> None:
    """Print pick value curve as a Rich table."""
    table = Table(title=f"Pick Value Curve ({curve.system} — {curve.provider}, {curve.season})")
    table.add_column("Pick", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Player", justify="left")
    table.add_column("Confidence", justify="left")

    confidence_styles = {"high": "green", "medium": "yellow", "low": "dim"}

    for pv in curve.picks:
        style = confidence_styles.get(pv.confidence, "")
        table.add_row(
            str(pv.pick),
            f"{pv.expected_value:.1f}",
            pv.player_name or "—",
            f"[{style}]{pv.confidence}[/{style}]",
        )

    console.print(table)


def _pick_detail_table(label: str, details: list[PickValue]) -> Table:
    table = Table(title=label, show_edge=False, pad_edge=False)
    table.add_column("Pick", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Player", justify="left")

    for pv in details:
        table.add_row(
            str(pv.pick),
            f"{pv.expected_value:.1f}",
            pv.player_name or "—",
        )
    return table


def print_pick_trade_evaluation(evaluation: PickTradeEvaluation) -> None:
    """Print pick trade evaluation with gives/receives tables and verdict."""
    console.print()
    console.print(_pick_detail_table("You give", evaluation.gives_detail))
    console.print()
    console.print(_pick_detail_table("You receive", evaluation.receives_detail))
    console.print()

    sign = "+" if evaluation.net_value >= 0 else ""
    style = "green" if evaluation.net_value >= 0 else "red"
    console.print(f"Net value: [{style}]{sign}{evaluation.net_value:.1f}[/{style}]")

    if evaluation.recommendation == "accept":
        console.print("[bold green]Recommendation: Accept[/bold green]")
    elif evaluation.recommendation == "reject":
        console.print("[bold red]Recommendation: Reject[/bold red]")
    else:
        console.print("[bold]Recommendation: Even[/bold]")


def print_cascade_result(result: CascadeResult) -> None:
    """Print cascade analysis showing before/after roster comparison."""
    before_table = Table(title="Before Trade", show_edge=False, pad_edge=False)
    before_table.add_column("Round", justify="right")
    before_table.add_column("Player", justify="left")
    before_table.add_column("Position", justify="left")
    before_table.add_column("Value", justify="right")

    for pick in result.before.picks:
        before_table.add_row(str(pick.round), pick.player_name, pick.position, f"{pick.value:.1f}")

    after_table = Table(title="After Trade", show_edge=False, pad_edge=False)
    after_table.add_column("Round", justify="right")
    after_table.add_column("Player", justify="left")
    after_table.add_column("Position", justify="left")
    after_table.add_column("Value", justify="right")

    for pick in result.after.picks:
        after_table.add_row(str(pick.round), pick.player_name, pick.position, f"{pick.value:.1f}")

    console.print()
    console.print(before_table)
    console.print(f"  Total: {result.before.total_value:.1f}")
    console.print()
    console.print(after_table)
    console.print(f"  Total: {result.after.total_value:.1f}")
    console.print()

    sign = "+" if result.value_delta >= 0 else ""
    style = "green" if result.value_delta >= 0 else "red"
    console.print(f"Cascade delta: [{style}]{sign}{result.value_delta:.1f}[/{style}]")

    if result.recommendation == "accept":
        console.print("[bold green]Cascade recommendation: Accept[/bold green]")
    elif result.recommendation == "reject":
        console.print("[bold red]Cascade recommendation: Reject[/bold red]")
    else:
        console.print("[bold]Cascade recommendation: Even[/bold]")


def print_scarcity_report(
    scarcities: list[PositionScarcity],
    league: LeagueSettings,
) -> None:
    all_positions: dict[str, int] = dict(league.positions) | dict(league.pitcher_positions)

    console.print(f"\n[bold]Positional Scarcity Report[/bold] — {league.name} ({league.teams} teams)\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Position", style="bold")
    table.add_column("Slots", justify="right")
    table.add_column("Tier1$", justify="right")
    table.add_column("Repl$", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Slope", justify="right")
    table.add_column("Steep Rank", justify="right")

    for ps in scarcities:
        slots = all_positions.get(ps.position, 0)
        slope_abs = abs(ps.dropoff_slope)
        if slope_abs >= 1.5:
            slope_color = "red"
        elif slope_abs >= 0.8:
            slope_color = "yellow"
        else:
            slope_color = "green"

        steep_display = str(ps.steep_rank) if ps.steep_rank is not None else "-"

        table.add_row(
            ps.position.upper(),
            str(slots),
            f"${ps.tier_1_value:.1f}",
            f"${ps.replacement_value:.1f}",
            f"${ps.total_surplus:.1f}",
            f"[{slope_color}]{ps.dropoff_slope:.2f}[/{slope_color}]",
            steep_display,
        )
