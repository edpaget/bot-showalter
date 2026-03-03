import re
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import CandidateValue, Err, Ok, Result

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from fantasy_baseball_manager.repos import FeatureCandidateRepo

INTERACTION_OPERATIONS: frozenset[str] = frozenset({"product", "ratio", "difference", "sum"})

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
