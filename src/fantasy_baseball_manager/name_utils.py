import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player
    from fantasy_baseball_manager.repos import PlayerRepo

_SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)\s*$", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\s*\((?:Batter|Pitcher)\)\s*$", re.IGNORECASE)
_INITIAL_DOT_RE = re.compile(r"(?<!\w)([A-Za-z])\.")
_ADJACENT_INITIALS_RE = re.compile(r"(?<=\b[A-Za-z]) (?=[A-Za-z]\b)")

# Formal name → common short form used in baseball databases.
# Applied during normalization so "Matthew Boyd" matches "Matt Boyd".
NICK_ALIASES: dict[str, str] = {
    "matthew": "matt",
    "michael": "mike",
    "christopher": "chris",
    "nicholas": "nick",
    "alexander": "alex",
    "benjamin": "ben",
    "gregory": "greg",
    "timothy": "tim",
    "stephen": "steve",
    "steven": "steve",
    "jeffrey": "jeff",
    "zachary": "zach",
    "frederick": "fred",
    "nathaniel": "nate",
    "jonathan": "jon",
    "abraham": "abe",
}


def strip_accents(text: str) -> str:
    """Remove accent/diacritical marks from text via NFKD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def strip_name_decorations(name: str) -> str:
    """Strip parentheticals (Batter/Pitcher) and suffixes (Jr., II, etc.) from a name."""
    name = _PARENTHETICAL_RE.sub("", name).strip()
    return _SUFFIX_RE.sub("", name).strip()


def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Strips parentheticals (Batter/Pitcher), suffixes (Jr., II, etc.),
    accent marks, initial dots, and applies nickname aliases.
    """
    name = _PARENTHETICAL_RE.sub("", name)
    name = _SUFFIX_RE.sub("", name)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Strip periods from initials: "J. T." -> "J  T " / "J.T." -> "JT "
    stripped = _INITIAL_DOT_RE.sub(r"\1", stripped)
    # Collapse whitespace, then merge adjacent single-letter tokens: "J T" -> "JT"
    stripped = " ".join(stripped.split())
    stripped = _ADJACENT_INITIALS_RE.sub("", stripped)
    lowered = stripped.strip().lower()
    # Apply nickname aliases to each token
    tokens = lowered.split()
    tokens = [NICK_ALIASES.get(t, t) for t in tokens]
    return " ".join(tokens)


def resolve_players(player_repo: PlayerRepo, name: str) -> list[Player]:
    """Resolve a player name query to matching Player records.

    Handles these input formats:
    - "Last, First" — comma-separated
    - "First Last" — space-separated (last token is last name)
    - "Last" — single word

    Uses accent-stripped, nickname-normalized matching so
    "Cristopher Sanchez" finds "Cristopher Sánchez".
    """
    if "," in name:
        last, _, first = name.partition(",")
        last = last.strip()
        first = first.strip() or None
    else:
        parts = name.strip().split()
        if len(parts) >= 2:
            first = " ".join(parts[:-1])
            last = parts[-1]
        else:
            first = None
            last = parts[0] if parts else name.strip()

    players = player_repo.search_by_last_name_normalized(strip_accents(last))

    if first:
        norm_first = normalize_name(first)
        players = [p for p in players if p.name_first and normalize_name(p.name_first) == norm_first]

    return players
