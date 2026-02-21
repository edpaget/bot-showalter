from __future__ import annotations

from fantasy_baseball_manager.domain.player_bio import PlayerSummary


def format_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Format data as a plain-text table with aligned columns.

    ``alignments`` is a list of ``"l"`` or ``"r"`` per column. Defaults to
    left-aligned for all columns.
    """
    if not rows:
        return ""
    if alignments is None:
        alignments = ["l"] * len(headers)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad(text: str, width: int, align: str) -> str:
        if align == "r":
            return text.rjust(width)
        return text.ljust(width)

    header_line = "  ".join(_pad(h, col_widths[i], alignments[i]) for i, h in enumerate(headers))
    separator = "  ".join("-" * w for w in col_widths)
    body_lines: list[str] = []
    for row in rows:
        body_lines.append("  ".join(_pad(row[i], col_widths[i], alignments[i]) for i in range(len(headers))))

    return "\n".join([header_line, separator, *body_lines])


def format_player_summary_table(summaries: list[PlayerSummary]) -> str:
    """Format a list of PlayerSummary as a plain-text table."""
    headers = ["Name", "Team", "Age", "Pos", "Exp"]
    alignments = ["l", "l", "r", "l", "r"]
    rows: list[list[str]] = []
    for s in summaries:
        rows.append(
            [
                s.name,
                s.team,
                str(s.age) if s.age is not None else "",
                s.primary_position,
                str(s.experience),
            ]
        )
    return format_table(headers, rows, alignments)


def format_no_results(entity: str, **filters: object) -> str:
    """Return a human-readable 'no results' message with non-None filters."""
    parts: list[str] = []
    for key, value in filters.items():
        if value is not None:
            parts.append(f"{key}={value!r}")
    filter_str = f" matching {', '.join(parts)}" if parts else ""
    return f"No {entity} found{filter_str}."
