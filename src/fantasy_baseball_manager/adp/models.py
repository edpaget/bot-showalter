"""ADP data models."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ADPEntry:
    """A single player's ADP entry.

    Attributes:
        name: The player's name as listed in the ADP source.
        adp: Average draft position.
        positions: Tuple of position eligibilities (e.g., ("OF",), ("DH", "SP")).
        percent_drafted: Percentage of drafts where this player was selected.
    """

    name: str
    adp: float
    positions: tuple[str, ...]
    percent_drafted: float | None = None


@dataclass(frozen=True)
class ADPData:
    """Collection of ADP entries from a source.

    Attributes:
        entries: Tuple of ADP entries.
        fetched_at: When this data was fetched.
        source: The data source identifier (default: "yahoo").
    """

    entries: tuple[ADPEntry, ...]
    fetched_at: datetime
    source: str = "yahoo"
