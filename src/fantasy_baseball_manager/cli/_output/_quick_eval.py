from typing import TYPE_CHECKING

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.services import QuickEvalResult


def print_quick_eval_result(result: QuickEvalResult) -> None:
    console.print(f"\n  Target: [bold]{result.target}[/bold]")
    console.print(f"  RMSE:   {result.rmse:.4f}")
    console.print(f"  R²:     {result.r_squared:.4f}")
    console.print(f"  n:      {result.n}")
    if result.baseline_rmse is not None and result.delta is not None and result.delta_pct is not None:
        color = "green" if result.delta < 0 else "red"
        console.print(f"  Baseline RMSE: {result.baseline_rmse:.4f}")
        console.print(f"  Delta:  [{color}]{result.delta:+.4f}[/{color}] ({result.delta_pct:+.1f}%)")
