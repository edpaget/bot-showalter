"""Grid search over regression constants to find optimal values.

Uses the existing evaluation harness to run multi-year backtests for every
combination of parameters in the search space, then outputs sorted results.

Usage:
    uv run python scripts/grid_search.py --workers 4
    uv run python scripts/grid_search.py --coarse --workers 2
    uv run python scripts/grid_search.py --years 2022,2023 --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore
from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.evaluation.harness import (
    EvaluationConfig,
    evaluate_source,
)
from fantasy_baseball_manager.marcel.data_source import (
    CachedStatsDataSource,
    PybaseballDataSource,
)
from fantasy_baseball_manager.pipeline.presets import build_pipeline
from fantasy_baseball_manager.pipeline.source import PipelineProjectionSource
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

FULL_SEARCH_SPACE: dict[str, list[float]] = {
    "babip_regression_weight": [0.3, 0.4, 0.5, 0.6, 0.7],
    "lob_regression_weight": [0.4, 0.5, 0.6, 0.7, 0.8],
    "batting_hr_pa": [350, 425, 500, 575, 650],
    "batting_singles_pa": [600, 700, 800, 900, 1000],
    "pitching_h_outs": [150, 175, 200, 225, 250],
    "pitching_er_outs": [100, 125, 150, 175, 200],
}

COARSE_SEARCH_SPACE: dict[str, list[float]] = {
    "babip_regression_weight": [0.3, 0.5, 0.7],
    "lob_regression_weight": [0.4, 0.6, 0.8],
    "batting_hr_pa": [350, 500, 650],
    "batting_singles_pa": [600, 800, 1000],
    "pitching_h_outs": [150, 200, 250],
    "pitching_er_outs": [100, 150, 200],
}


@dataclass(frozen=True)
class SearchPoint:
    babip_regression_weight: float
    lob_regression_weight: float
    batting_hr_pa: float
    batting_singles_pa: float
    pitching_h_outs: float
    pitching_er_outs: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_grid(space: dict[str, list[float]]) -> list[SearchPoint]:
    """Generate all combinations from the search space."""
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [SearchPoint(**dict(zip(keys, combo, strict=True))) for combo in itertools.product(*values)]


def search_point_to_config(point: SearchPoint) -> RegressionConfig:
    """Convert a search point into a RegressionConfig with overrides applied."""
    batting = dict(BATTING_REGRESSION_PA)
    batting["hr"] = point.batting_hr_pa
    batting["singles"] = point.batting_singles_pa

    pitching = dict(PITCHING_REGRESSION_OUTS)
    pitching["h"] = point.pitching_h_outs
    pitching["er"] = point.pitching_er_outs

    norm = PitcherNormalizationConfig(
        babip_regression_weight=point.babip_regression_weight,
        lob_regression_weight=point.lob_regression_weight,
    )

    return RegressionConfig(
        batting_regression_pa=batting,
        pitching_regression_outs=pitching,
        pitcher_normalization=norm,
    )


def evaluate_point(
    point: SearchPoint,
    eval_years: list[int],
    pipeline_name: str,
    min_pa: int,
    min_ip: float,
    top_n: int,
    cache_db_path: str | None = None,
) -> dict[str, Any]:
    """Run a multi-year backtest for a single search point.

    Each call creates its own data source so workers don't share state.
    When *cache_db_path* is provided, the data source is wrapped with
    ``CachedStatsDataSource`` so that multiple workers share a single
    SQLite cache (each opens its own connection).
    """
    config = search_point_to_config(point)
    pipeline = build_pipeline(pipeline_name, config=config)
    raw_source = PybaseballDataSource()
    if cache_db_path is not None:
        cache = SqliteCacheStore(Path(cache_db_path))
        data_source: PybaseballDataSource | CachedStatsDataSource = CachedStatsDataSource(raw_source, cache)
    else:
        data_source = raw_source
    league_settings = load_league_settings()

    batting_rhos: list[float] = []
    pitching_rhos: list[float] = []
    batting_rmses: list[float] = []
    pitching_rmses: list[float] = []
    batting_maes: list[float] = []
    pitching_maes: list[float] = []

    for year in eval_years:
        eval_config = EvaluationConfig(
            year=year,
            batting_categories=league_settings.batting_categories,
            pitching_categories=league_settings.pitching_categories,
            min_pa=min_pa,
            min_ip=min_ip,
            top_n=top_n,
        )
        source = PipelineProjectionSource(pipeline, data_source, year)
        evaluation = evaluate_source(source, pipeline_name, data_source, eval_config)

        if evaluation.batting_rank_accuracy:
            batting_rhos.append(evaluation.batting_rank_accuracy.spearman_rho)
        if evaluation.pitching_rank_accuracy:
            pitching_rhos.append(evaluation.pitching_rank_accuracy.spearman_rho)

        for sa in evaluation.batting_stat_accuracy:
            batting_rmses.append(sa.rmse)
            batting_maes.append(sa.mae)
        for sa in evaluation.pitching_stat_accuracy:
            pitching_rmses.append(sa.rmse)
            pitching_maes.append(sa.mae)

    avg_batting_rho = sum(batting_rhos) / len(batting_rhos) if batting_rhos else 0.0
    avg_pitching_rho = sum(pitching_rhos) / len(pitching_rhos) if pitching_rhos else 0.0
    avg_rho = (avg_batting_rho + avg_pitching_rho) / 2.0

    return {
        "params": {
            "babip_regression_weight": point.babip_regression_weight,
            "lob_regression_weight": point.lob_regression_weight,
            "batting_hr_pa": point.batting_hr_pa,
            "batting_singles_pa": point.batting_singles_pa,
            "pitching_h_outs": point.pitching_h_outs,
            "pitching_er_outs": point.pitching_er_outs,
        },
        "metrics": {
            "avg_spearman_rho": round(avg_rho, 5),
            "avg_batting_rho": round(avg_batting_rho, 5),
            "avg_pitching_rho": round(avg_pitching_rho, 5),
            "avg_batting_rmse": round(sum(batting_rmses) / len(batting_rmses), 5) if batting_rmses else None,
            "avg_pitching_rmse": round(sum(pitching_rmses) / len(pitching_rmses), 5) if pitching_rmses else None,
            "avg_batting_mae": round(sum(batting_maes) / len(batting_maes), 5) if batting_maes else None,
            "avg_pitching_mae": round(sum(pitching_maes) / len(pitching_maes), 5) if pitching_maes else None,
        },
    }


# ---------------------------------------------------------------------------
# Worker wrapper for ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _worker(args: tuple[SearchPoint, list[int], str, int, float, int, str | None]) -> dict[str, Any]:
    point, eval_years, pipeline_name, min_pa, min_ip, top_n, cache_db_path = args
    return evaluate_point(point, eval_years, pipeline_name, min_pa, min_ip, top_n, cache_db_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over regression constants.")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use coarse grid (3 values per param, 729 combos) instead of full grid.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument(
        "--years",
        type=str,
        default="2021,2022,2023,2024",
        help="Comma-separated evaluation years.",
    )
    parser.add_argument("--pipeline", type=str, default="marcel_norm", help="Pipeline preset name.")
    parser.add_argument("--min-pa", type=int, default=200, help="Minimum plate appearances.")
    parser.add_argument("--min-ip", type=float, default=50.0, help="Minimum innings pitched.")
    parser.add_argument("--top-n", type=int, default=20, help="N for top-N precision.")
    parser.add_argument(
        "--output",
        type=str,
        default="grid_search_results.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    space = COARSE_SEARCH_SPACE if args.coarse else FULL_SEARCH_SPACE
    grid = generate_grid(space)
    eval_years = [int(y.strip()) for y in args.years.split(",")]

    # Create a shared SQLite cache for all workers
    cache_db_path = str(Path("~/.fantasy_baseball/grid_search_cache.db").expanduser())
    Path(cache_db_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Grid search: {len(grid)} combinations, {len(eval_years)} eval years, {args.workers} worker(s)")
    print(f"Pipeline: {args.pipeline}")
    print(f"Years: {eval_years}")
    print(f"Cache: {cache_db_path}")

    worker_args = [
        (point, eval_years, args.pipeline, args.min_pa, args.min_ip, args.top_n, cache_db_path) for point in grid
    ]

    results: list[dict[str, Any]] = []
    start = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, result in enumerate(executor.map(_worker, worker_args)):
                results.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == len(grid):
                    elapsed = time.time() - start
                    print(f"  [{i + 1}/{len(grid)}] elapsed: {elapsed:.1f}s")
    else:
        for i, wa in enumerate(worker_args):
            results.append(_worker(wa))
            if (i + 1) % 10 == 0 or (i + 1) == len(grid):
                elapsed = time.time() - start
                print(f"  [{i + 1}/{len(grid)}] elapsed: {elapsed:.1f}s")

    # Sort by avg spearman rho descending
    results.sort(key=lambda r: r["metrics"]["avg_spearman_rho"], reverse=True)

    # Write all results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Print top 10
    print("\nTop 10 configurations:")
    print(f"{'Rank':<5} {'Avg Rho':>9} {'Bat Rho':>9} {'Pit Rho':>9}  Params")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        p = r["params"]
        params_str = ", ".join(f"{k}={v}" for k, v in p.items())
        print(
            f"{i + 1:<5} {m['avg_spearman_rho']:>9.5f} {m['avg_batting_rho']:>9.5f} {m['avg_pitching_rho']:>9.5f}  {params_str}"
        )

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
