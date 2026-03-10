import re
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import BinnedValue, CandidateValue, Err, Ok, Result

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from fantasy_baseball_manager.repos import FeatureCandidateRepo

INTERACTION_OPERATIONS: frozenset[str] = frozenset({"product", "ratio", "difference", "sum"})
BINNING_METHODS: frozenset[str] = frozenset({"quantile", "uniform", "custom"})

_DANGEROUS_KEYWORDS = {
    "DROP",
    "INSERT",
    "UPDATE",
    "DELETE",
    "ALTER",
    "EXEC",
    "EXECUTE",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "UNION",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "COMMIT",
    "ROLLBACK",
    "SELECT",
}

_ALLOWED_FUNCTIONS = {
    "AVG",
    "SUM",
    "COUNT",
    "MIN",
    "MAX",
    "CAST",
    "COALESCE",
    "NULLIF",
    "IIF",
    "ABS",
    "ROUND",
    "SUBSTR",
    "LENGTH",
    "TOTAL",
    "GROUP_CONCAT",
}

_VALID_PLAYER_TYPES = {"batter", "pitcher"}


def validate_expression(expression: str) -> Result[str, str]:
    """Validate a SQL aggregation expression fragment.

    Returns Ok(expression) if safe, Err(reason) if dangerous or invalid.
    """
    if not expression or not expression.strip():
        return Err("Expression must not be empty")

    if ";" in expression:
        return Err("Expression must not contain semicolons")

    upper = expression.upper()

    # Check for dangerous keywords (word-boundary match, case-insensitive)
    for keyword in _DANGEROUS_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return Err(f"Expression contains disallowed keyword: {keyword}")

    # Whitelist function calls: tokens followed by '(' must be in the allowed set
    for match in re.finditer(r"\b([A-Z_]+)\s*\(", upper):
        func_name = match.group(1)
        if func_name not in _ALLOWED_FUNCTIONS and func_name not in {
            "FILTER",
            "WHEN",
            "CASE",
            "THEN",
            "ELSE",
            "END",
            "WHERE",
            "AS",
            "AND",
            "OR",
            "NOT",
            "IN",
            "IS",
            "LIKE",
            "BETWEEN",
        }:
            return Err(f"Function not in allowlist: {func_name}")

    return Ok[str](expression)


def aggregate_candidate(
    conn: sqlite3.Connection,
    expression: str,
    seasons: Sequence[int],
    player_type: str,
    *,
    min_pa: int | None = None,
    min_ip: float | None = None,
) -> list[CandidateValue]:
    """Aggregate statcast data into player-season vectors using a SQL expression.

    Raises ValueError on invalid expression or player_type.
    """
    result = validate_expression(expression)
    if isinstance(result, Err):
        msg = f"Invalid expression: {result.error}"
        raise ValueError(msg)

    if player_type not in _VALID_PLAYER_TYPES:
        msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
        raise ValueError(msg)

    player_col = "batter_id" if player_type == "batter" else "pitcher_id"
    placeholders = ",".join("?" for _ in seasons)

    having_clause = ""
    if min_pa is not None and player_type == "batter":
        having_clause = f"HAVING COUNT(*) FILTER (WHERE events IS NOT NULL) >= {min_pa}"
    elif min_ip is not None and player_type == "pitcher":
        min_bf = int(min_ip * 4.3)
        having_clause = f"HAVING COUNT(*) FILTER (WHERE events IS NOT NULL) >= {min_bf}"

    sql = f"""
        SELECT {player_col} AS player_id,
               CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
               {expression} AS value
        FROM statcast_pitch
        WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
        GROUP BY player_id, season
        {having_clause}
    """  # noqa: S608

    rows = conn.execute(sql, list(seasons)).fetchall()
    return [
        CandidateValue(
            player_id=row[0],
            season=row[1],
            value=float(row[2]) if row[2] is not None else None,
        )
        for row in rows
    ]


def _apply_operation(a: float | None, b: float | None, op: str) -> float | None:
    """Apply an arithmetic operation, returning None for any None input or div-by-zero."""
    if a is None or b is None:
        return None
    if op == "product":
        return a * b
    if op == "sum":
        return a + b
    if op == "difference":
        return a - b
    if op == "ratio":
        if b == 0.0:
            return None
        return a / b
    msg = f"Unknown operation: {op}"
    raise ValueError(msg)


def interact_candidates(
    values_a: list[CandidateValue],
    values_b: list[CandidateValue],
    operation: str,
) -> list[CandidateValue]:
    """Compute an interaction between two feature vectors (inner join on player_id, season)."""
    if operation not in INTERACTION_OPERATIONS:
        msg = f"Invalid operation: {operation!r}. Must be one of {sorted(INTERACTION_OPERATIONS)}"
        raise ValueError(msg)

    b_index: dict[tuple[int, int], float | None] = {(v.player_id, v.season): v.value for v in values_b}

    results: list[CandidateValue] = []
    for v in values_a:
        key = (v.player_id, v.season)
        if key not in b_index:
            continue
        result_value = _apply_operation(v.value, b_index[key], operation)
        results.append(CandidateValue(player_id=v.player_id, season=v.season, value=result_value))
    return results


def resolve_feature(
    name_or_expr: str,
    conn: sqlite3.Connection,
    candidate_repo: FeatureCandidateRepo,
    seasons: Sequence[int],
    player_type: str,
) -> list[CandidateValue]:
    """Resolve a name-or-expression to candidate values.

    Checks if *name_or_expr* matches a saved candidate name first;
    if found, uses the stored expression and min_pa/min_ip.
    Otherwise, treats it as a raw SQL expression.
    """
    saved = candidate_repo.get_by_name(name_or_expr)
    if saved is not None:
        return aggregate_candidate(
            conn,
            saved.expression,
            seasons,
            player_type,
            min_pa=saved.min_pa,
            min_ip=saved.min_ip,
        )
    return aggregate_candidate(conn, name_or_expr, seasons, player_type)


def candidate_values_to_dict(values: list[CandidateValue]) -> dict[tuple[int, int], float]:
    """Convert candidate values to a dict keyed by (player_id, season), dropping NULLs."""
    return {(v.player_id, v.season): v.value for v in values if v.value is not None}


def inject_candidate_values(
    rows_by_season: dict[int, list[dict[str, Any]]],
    column_name: str,
    values: dict[tuple[int, int], float],
) -> None:
    """Inject resolved candidate values into row dicts in-place.

    Matches on (player_id, season) keys.  Rows without a match get NaN
    (the feature genuinely has no value for that player-season).
    """
    for season, season_rows in rows_by_season.items():
        for row in season_rows:
            key = (row["player_id"], season)
            row[column_name] = values.get(key, float("nan"))


def inject_candidate_values_lagged(
    rows_by_season: dict[int, list[dict[str, Any]]],
    column_name: str,
    values: dict[tuple[int, int], float],
    lags: tuple[int, ...],
    weights: tuple[float, ...],
) -> None:
    """Inject lag-adjusted candidate values into row dicts in-place.

    For each target season, look up the candidate value from prior seasons
    (``target_season - lag``) and blend with *weights*.  This matches the
    weighted-lag feature materialization pipeline.
    """
    nan = float("nan")
    for target_season, season_rows in rows_by_season.items():
        for row in season_rows:
            pid = row["player_id"]
            blended = 0.0
            total_weight = 0.0
            for lag, weight in zip(lags, weights, strict=True):
                source_season = target_season - lag
                val = values.get((pid, source_season))
                if val is not None:
                    blended += weight * val
                    total_weight += weight
            row[column_name] = blended / total_weight if total_weight > 0 else nan


def source_seasons_for_lags(
    target_seasons: list[int],
    lags: tuple[int, ...],
) -> list[int]:
    """Compute the set of source seasons needed to resolve lagged candidates."""
    sources: set[int] = set()
    for ts in target_seasons:
        for lag in lags:
            sources.add(ts - lag)
    return sorted(sources)


def resolve_and_inject_candidates(
    rows_by_season: dict[int, list[dict[str, Any]]],
    candidate_names: list[str],
    statcast_conn: sqlite3.Connection,
    candidate_repo: FeatureCandidateRepo,
    mlbam_to_internal: dict[int, int],
    player_type: str,
    lags: tuple[int, ...],
    weights: tuple[float, ...],
) -> None:
    """Resolve candidate features from statcast and inject lag-aligned values.

    This is the correct entry point for experiment commands.  It handles:
    1. Computing which source seasons to query (shifted by *lags*).
    2. Resolving candidate values from statcast data.
    3. Remapping mlbam → internal IDs.
    4. Blending multi-lag values with *weights* and injecting into rows.
    """
    target_seasons = list(rows_by_season.keys())
    query_seasons = source_seasons_for_lags(target_seasons, lags)

    for cand_name in candidate_names:
        cv = resolve_feature(cand_name, statcast_conn, candidate_repo, query_seasons, player_type)
        values_dict = candidate_values_to_dict(cv)
        remapped = remap_candidate_keys(values_dict, mlbam_to_internal)
        inject_candidate_values_lagged(rows_by_season, cand_name, remapped, lags, weights)


def remap_candidate_keys(
    values: dict[tuple[int, int], float],
    mlbam_to_internal: dict[int, int],
) -> dict[tuple[int, int], float]:
    """Re-key candidate values from (mlbam_id, season) to (internal_id, season).

    Entries whose mlbam_id has no mapping are silently dropped.
    """
    return {(mlbam_to_internal[mid], season): val for (mid, season), val in values.items() if mid in mlbam_to_internal}


_BIN_LABEL_PREFIX = {"quantile": "Q", "uniform": "B", "custom": "C"}


def _compute_breakpoints(data: list[float], method: str, n_bins: int, breakpoints: list[float] | None) -> list[float]:
    """Compute bin breakpoints for a single season's data."""
    if method == "custom":
        assert breakpoints is not None  # noqa: S101
        return sorted(breakpoints)

    if len(data) < 2:
        return []

    if method == "quantile":
        return statistics.quantiles(data, n=n_bins)

    # uniform: equal-width intervals
    lo, hi = min(data), max(data)
    if lo == hi:
        return []
    width = (hi - lo) / n_bins
    return [lo + width * i for i in range(1, n_bins)]


def _assign_bin(value: float, breaks: list[float], prefix: str, n_bins: int) -> str:
    """Assign a value to its bin label given breakpoints."""
    if not breaks:
        return f"{prefix}1"
    for i, bp in enumerate(breaks):
        if value < bp:
            return f"{prefix}{i + 1}"
    return f"{prefix}{n_bins}"


def bin_candidate(
    values: list[CandidateValue],
    method: str,
    n_bins: int,
    breakpoints: list[float] | None = None,
) -> list[BinnedValue]:
    """Bin continuous candidate values into discrete categories.

    Breakpoints are computed per-season to avoid lookahead.
    """
    if method not in BINNING_METHODS:
        msg = f"Invalid method: {method!r}. Must be one of {sorted(BINNING_METHODS)}"
        raise ValueError(msg)

    if method == "custom" and breakpoints is None:
        msg = "breakpoints required when method='custom'"
        raise ValueError(msg)

    prefix = _BIN_LABEL_PREFIX[method]

    # Group non-null values by season
    by_season: dict[int, list[CandidateValue]] = defaultdict(list)
    for v in values:
        if v.value is not None:
            by_season[v.season].append(v)

    results: list[BinnedValue] = []
    for season, season_values in sorted(by_season.items()):
        data: list[float] = [v.value for v in season_values if v.value is not None]
        breaks = _compute_breakpoints(data, method, n_bins, breakpoints)
        for v in season_values:
            val = v.value
            if val is None:
                continue
            label = _assign_bin(val, breaks, prefix, n_bins)
            results.append(BinnedValue(player_id=v.player_id, season=season, bin_label=label, value=val))

    return results


def cross_bin_candidates(bins_a: list[BinnedValue], bins_b: list[BinnedValue]) -> list[BinnedValue]:
    """Cross-product two binned feature lists on (player_id, season)."""
    b_index: dict[tuple[int, int], BinnedValue] = {(b.player_id, b.season): b for b in bins_b}

    results: list[BinnedValue] = []
    for a in bins_a:
        key = (a.player_id, a.season)
        b = b_index.get(key)
        if b is None:
            continue
        label = f"{a.bin_label}__{b.bin_label}"
        results.append(BinnedValue(player_id=a.player_id, season=a.season, bin_label=label, value=a.value))
    return results
