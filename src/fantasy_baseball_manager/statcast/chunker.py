from __future__ import annotations

from fantasy_baseball_manager.statcast.calendar import game_dates
from fantasy_baseball_manager.statcast.models import DateChunk, SeasonManifest


def pending_chunks(
    season: int,
    manifest: SeasonManifest,
    *,
    force: bool = False,
) -> list[DateChunk]:
    """Return DateChunks for dates not yet recorded in the manifest."""
    all_dates = game_dates(season)
    if force:
        return [DateChunk(date=d, season=season) for d in all_dates]
    return [DateChunk(date=d, season=season) for d in all_dates if d.isoformat() not in manifest.fetched_dates]
