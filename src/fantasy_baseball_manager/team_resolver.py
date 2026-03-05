import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import TeamRepo

# Lahman → modern abbreviation aliases
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


def _split_name(full_name: str) -> tuple[str, str]:
    """Split a team name into (city, nickname).

    Uses last word as nickname, everything before as city.
    Example: "New York Yankees" → ("new york", "yankees")
    """
    parts = full_name.strip().rsplit(maxsplit=1)
    if len(parts) == 2:
        return parts[0].lower(), parts[1].lower()
    return "", parts[0].lower() if parts else ""


class TeamResolver:
    """Resolves free-text team queries into canonical abbreviations.

    Uses tiered matching: exact → substring → fuzzy.
    """

    def __init__(self, team_repo: TeamRepo) -> None:
        self._team_repo = team_repo
        self._lookup: dict[str, set[str]] | None = None

    def _build_lookup(self) -> dict[str, set[str]]:
        lookup: dict[str, set[str]] = {}
        teams = self._team_repo.all()

        def _add(key: str, abbrev: str) -> None:
            key = key.lower()
            if not key:
                return
            lookup.setdefault(key, set()).add(abbrev)

        for team in teams:
            abbrev = team.abbreviation
            # Map abbreviation
            _add(abbrev, abbrev)
            # Map full name
            _add(team.name, abbrev)
            # Split into city and nickname (last word = nickname, rest = city)
            city, nickname = _split_name(team.name)
            _add(city, abbrev)
            _add(nickname, abbrev)
            # Also map each individual word for partial matching
            for word in team.name.split():
                _add(word, abbrev)

        # Lahman aliases → modern abbreviations
        for lahman, modern in TEAM_ALIASES.items():
            _add(lahman, modern)

        return lookup

    def _get_lookup(self) -> dict[str, set[str]]:
        if self._lookup is None:
            self._lookup = self._build_lookup()
        return self._lookup

    def resolve(self, query: str) -> list[str]:
        """Resolve a free-text team query into canonical abbreviations.

        Returns a sorted list of matching abbreviations (empty if no match).
        """
        if not query or not query.strip():
            return []

        lookup = self._get_lookup()
        query_lower = query.strip().lower()

        # 1. Exact match in lookup dict
        if query_lower in lookup:
            return sorted(lookup[query_lower])

        # 2. Substring match against all keys
        substring_hits: set[str] = set()
        for key, abbrevs in lookup.items():
            if query_lower in key:
                substring_hits.update(abbrevs)
        if substring_hits:
            return sorted(substring_hits)

        # 3. Fuzzy match via difflib
        all_keys = list(lookup.keys())
        close = difflib.get_close_matches(query_lower, all_keys, n=3, cutoff=0.6)
        fuzzy_hits: set[str] = set()
        for match in close:
            fuzzy_hits.update(lookup[match])
        if fuzzy_hits:
            return sorted(fuzzy_hits)

        return []
