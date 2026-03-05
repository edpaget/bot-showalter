"""Shared Lahman-to-modern team abbreviation mappings."""

# Lahman abbreviation -> modern abbreviation
TEAM_ALIASES: dict[str, str] = {
    "KCA": "KC",
    "TBA": "TB",
    "SFN": "SF",
    "SDN": "SD",
    "SLN": "STL",
    "CHN": "CHC",
    "CHA": "CWS",
    "LAN": "LAD",
    "NYA": "NYY",
    "NYN": "NYM",
    "WAS": "WSH",
    "ANA": "LAA",
}

# Modern abbreviation -> Lahman abbreviation (auto-built reverse mapping)
REVERSE_ALIASES: dict[str, str] = {v: k for k, v in TEAM_ALIASES.items()}


def to_modern(abbrev: str) -> str:
    """Convert a Lahman abbreviation to its modern form, or pass through if no alias."""
    return TEAM_ALIASES.get(abbrev, abbrev)


def to_lahman(abbrev: str) -> str:
    """Convert a modern abbreviation to its Lahman form, or pass through if no alias."""
    return REVERSE_ALIASES.get(abbrev, abbrev)
