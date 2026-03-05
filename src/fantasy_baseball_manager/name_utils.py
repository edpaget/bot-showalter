import re
import unicodedata

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
