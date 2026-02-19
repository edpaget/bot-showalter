from typing import Any


def nullify_empty_strings(row: dict[str, str]) -> dict[str, Any]:
    return {k: (None if v == "" else v) for k, v in row.items()}
