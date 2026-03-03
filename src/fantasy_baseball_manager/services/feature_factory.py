import re
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import CandidateValue, Err, Ok, Result

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

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
