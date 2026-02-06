from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

DEFAULT_DATA_DIR: Path = Path.home() / ".fantasy_baseball" / "statcast"


@dataclass(frozen=True)
class DownloadConfig:
    seasons: tuple[int, ...]
    data_dir: Path = DEFAULT_DATA_DIR
    max_retries: int = 5
    base_delay: float = 3.0
    max_delay: float = 120.0
    force: bool = False


@dataclass(frozen=True)
class DateChunk:
    date: date
    season: int


@dataclass(frozen=True)
class ChunkResult:
    date: date
    season: int
    row_count: int
    success: bool
    error: str | None = None


class StatcastDownloadError(Exception):
    pass


@dataclass
class SeasonManifest:
    season: int
    fetched_dates: set[str] = field(default_factory=set)
    total_rows: int = 0
