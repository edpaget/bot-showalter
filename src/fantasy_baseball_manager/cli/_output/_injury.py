from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ExpectedGamesLost, InjuryProfile


def print_injury_profile(profile: InjuryProfile, player_name: str) -> None:
    """Print a single player's injury profile."""
    console.print(f"[bold]Injury Profile[/bold] — {player_name}")
    console.print()
    console.print(f"  Seasons tracked:      {profile.seasons_tracked}")
    console.print(f"  Total IL stints:      {profile.total_stints}")
    console.print(f"  Total days lost:      {profile.total_days_lost}")
    console.print(f"  Avg days/season:      {profile.avg_days_per_season:.1f}")
    console.print(f"  Max days in season:   {profile.max_days_in_season}")
    console.print(f"  Pct seasons with IL:  {profile.pct_seasons_with_il:.0%}")

    if profile.injury_locations:
        console.print()
        console.print("[bold]Injury Locations:[/bold]")
        for location, count in sorted(profile.injury_locations.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {location}: {count}")

    if profile.recent_stints:
        console.print()
        console.print("[bold]Recent Stints:[/bold]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Season")
        table.add_column("Start")
        table.add_column("Type")
        table.add_column("Days", justify="right")
        table.add_column("Location")
        for stint in profile.recent_stints:
            days_str = str(stint.days) if stint.days is not None else "—"
            table.add_row(
                str(stint.season),
                stint.start_date,
                stint.il_type,
                days_str,
                stint.injury_location or "—",
            )
        console.print(table)


def print_injury_risk_leaderboard(profiles: list[tuple[InjuryProfile, str]]) -> None:
    """Print injury risk leaderboard."""
    if not profiles:
        console.print("No injury-prone players found.")
        return

    console.print(f"[bold]Injury Risk Leaderboard[/bold] — {len(profiles)} players")
    console.print()

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Stints", justify="right")
    table.add_column("Days Lost", justify="right")
    table.add_column("Avg/Season", justify="right")
    table.add_column("Max Season", justify="right")
    table.add_column("% w/ IL", justify="right")
    table.add_column("Top Location")

    for profile, name in profiles:
        top_location = "—"
        if profile.injury_locations:
            top_location = max(profile.injury_locations, key=profile.injury_locations.get)  # type: ignore[arg-type]

        table.add_row(
            name,
            str(profile.total_stints),
            str(profile.total_days_lost),
            f"{profile.avg_days_per_season:.1f}",
            str(profile.max_days_in_season),
            f"{profile.pct_seasons_with_il:.0%}",
            top_location,
        )

    console.print(table)


def print_injury_estimate(estimate: ExpectedGamesLost, profile: InjuryProfile, player_name: str) -> None:
    """Print a single player's injury estimate with profile context."""
    console.print(f"[bold]Injury Estimate[/bold] — {player_name}")
    console.print()
    console.print(f"  Expected days lost: {estimate.expected_days_lost:.1f}")
    console.print(f"  P(full season):     {estimate.p_full_season:.1%}")
    console.print(f"  Confidence:         {estimate.confidence}")
    console.print()
    console.print(f"  Seasons tracked:    {profile.seasons_tracked}")
    console.print(f"  Total IL stints:    {profile.total_stints}")
    console.print(f"  Total days lost:    {profile.total_days_lost}")
    console.print(f"  Avg days/season:    {profile.avg_days_per_season:.1f}")

    if profile.injury_locations:
        console.print()
        console.print("[bold]Injury Locations:[/bold]")
        for location, count in sorted(profile.injury_locations.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {location}: {count}")


def print_games_lost_leaderboard(results: list[tuple[ExpectedGamesLost, InjuryProfile, str]]) -> None:
    """Print games-lost leaderboard table."""
    if not results:
        console.print("No players found.")
        return

    console.print(f"[bold]Expected Games Lost[/bold] — {len(results)} players")
    console.print()

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Exp Days", justify="right")
    table.add_column("P(Full)", justify="right")
    table.add_column("Confidence")
    table.add_column("Stints", justify="right")

    for estimate, profile, name in results:
        table.add_row(
            name,
            f"{estimate.expected_days_lost:.1f}",
            f"{estimate.p_full_season:.0%}",
            estimate.confidence,
            str(profile.total_stints),
        )

    console.print(table)
