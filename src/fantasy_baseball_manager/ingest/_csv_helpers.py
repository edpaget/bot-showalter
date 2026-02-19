from typing import Any


def strip_bom(text: str) -> str:
    """Strip UTF-8 BOM from the start of text (Baseball Savant CSVs include it)."""
    return text.removeprefix("\ufeff")


def nullify_empty_strings(row: dict[str, str]) -> dict[str, Any]:
    return {k: (None if v == "" else v) for k, v in row.items()}
