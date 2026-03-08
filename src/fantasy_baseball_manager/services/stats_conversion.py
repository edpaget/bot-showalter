"""Shared utilities for converting stats dataclass instances to float dicts."""

import dataclasses

METADATA_FIELDS = frozenset({"id", "player_id", "season", "source", "team_id", "loaded_at"})


def stats_to_dict(obj: object) -> dict[str, float]:
    """Extract all numeric fields from a BattingStats or PitchingStats instance.

    Skips metadata fields (id, player_id, season, source, team_id, loaded_at).
    """
    result: dict[str, float] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if field.name in METADATA_FIELDS:
            continue
        value = getattr(obj, field.name)
        if isinstance(value, int | float):
            result[field.name] = float(value)
    return result
