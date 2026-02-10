"""Player name normalization utilities for cross-source matching."""

from __future__ import annotations

import re
import unicodedata


def normalize_name(name: str) -> str:
    """Normalize a player name for cross-source matching.

    - Strips provider-specific suffixes like (Batter)/(Pitcher)
    - Removes accents/diacritics via NFD decomposition
    - Converts to lowercase
    - Removes periods (for Jr./Sr./J.D.)

    Args:
        name: Raw player name from any data source.

    Returns:
        Normalized lowercase name suitable for dictionary-key matching.
    """
    name = re.sub(r"\s*\((Batter|Pitcher)\)\s*$", "", name)
    normalized = unicodedata.normalize("NFD", name)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = normalized.lower()
    normalized = re.sub(r"\.", "", normalized)
    return normalized
