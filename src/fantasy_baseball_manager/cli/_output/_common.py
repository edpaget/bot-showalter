from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import StatComparisonRecord

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def print_error(message: str) -> None:
    err_console.print(f"[red bold]Error:[/red bold] {message}")


def _color_delta(value: str, winner: str) -> str:
    if winner == "candidate":
        return f"[green]{value}[/green]"
    if winner == "baseline":
        return f"[red]{value}[/red]"
    return value


def _format_delta(delta: float, pct_delta: float, winner: str) -> tuple[str, str]:
    sign = "+" if delta >= 0 else ""
    delta_str = _color_delta(f"{sign}{delta:.4f}", winner)
    pct_str = _color_delta(f"{sign}{pct_delta:.1f}%", winner)
    return delta_str, pct_str


def _build_two_system_row(rec: StatComparisonRecord) -> list[str]:
    rmse_d, rmse_pct = _format_delta(rec.rmse_delta, rec.rmse_pct_delta, rec.rmse_winner)
    r2_d, r2_pct = _format_delta(rec.r_squared_delta, rec.r_squared_pct_delta, rec.r_squared_winner)
    rc_d, rc_pct = _format_delta(
        rec.rank_correlation_delta, rec.rank_correlation_pct_delta, rec.rank_correlation_winner
    )
    return [
        rec.stat_name,
        f"{rec.baseline_rmse:.4f}",
        f"{rec.candidate_rmse:.4f}",
        rmse_d,
        rmse_pct,
        f"{rec.baseline_r_squared:.4f}",
        f"{rec.candidate_r_squared:.4f}",
        r2_d,
        r2_pct,
        f"{rec.baseline_rank_correlation:.4f}",
        f"{rec.candidate_rank_correlation:.4f}",
        rc_d,
        rc_pct,
    ]
