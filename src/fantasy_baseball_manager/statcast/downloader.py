from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from fantasy_baseball_manager.result import Ok
from fantasy_baseball_manager.statcast.chunker import pending_chunks
from fantasy_baseball_manager.statcast.models import (
    ChunkResult,
    DownloadConfig,
    SeasonManifest,
    StatcastDownloadError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

    from fantasy_baseball_manager.result import Result
    from fantasy_baseball_manager.statcast.fetcher import StatcastFetcher
    from fantasy_baseball_manager.statcast.store import StatcastStore

logger = logging.getLogger(__name__)


class StatcastDownloader:
    def __init__(
        self,
        fetcher: StatcastFetcher,
        store: StatcastStore,
        config: DownloadConfig,
        progress_callback: Callable[[ChunkResult], None] | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._store = store
        self._config = config
        self._progress_callback = progress_callback

    def download_season(self, season: int) -> Result[SeasonManifest, StatcastDownloadError]:
        manifest = self._store.load_manifest(season)
        chunks = pending_chunks(season, manifest, force=self._config.force)

        if not chunks:
            logger.info("Season %d: all dates already fetched", season)
            return Ok(manifest)

        month_buffers: dict[int, list[pd.DataFrame]] = defaultdict(list)

        for chunk in chunks:
            chunk_result, df = self._fetch_with_retry(chunk.date, chunk.season)

            if chunk_result.success and df is not None and len(df) > 0:
                month_buffers[chunk.date.month].append(df)

            if self._progress_callback:
                self._progress_callback(chunk_result)

            if chunk_result.success:
                manifest.fetched_dates.add(chunk.date.isoformat())
                manifest.total_rows += chunk_result.row_count

        for month, frames in month_buffers.items():
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                self._store.flush_month(season, month, combined)

        self._store.save_manifest(manifest)
        return Ok(manifest)

    def _fetch_with_retry(self, day: date, season: int) -> tuple[ChunkResult, pd.DataFrame | None]:
        for attempt in range(self._config.max_retries):
            try:
                df = self._fetcher.fetch_day(day)
                row_count = len(df)
                return (
                    ChunkResult(date=day, season=season, row_count=row_count, success=True),
                    df if row_count > 0 else None,
                )
            except Exception as e:
                if attempt < self._config.max_retries - 1:
                    delay = min(
                        self._config.base_delay * (2**attempt) + random.random(),
                        self._config.max_delay,
                    )
                    logger.warning(
                        "Fetch failed for %s (attempt %d/%d), retrying in %.1fs: %s",
                        day.isoformat(),
                        attempt + 1,
                        self._config.max_retries,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Permanent failure for %s after %d attempts: %s",
                        day.isoformat(),
                        self._config.max_retries,
                        e,
                    )
                    return (
                        ChunkResult(
                            date=day,
                            season=season,
                            row_count=0,
                            success=False,
                            error=str(e),
                        ),
                        None,
                    )
        # Unreachable when max_retries > 0
        return (
            ChunkResult(date=day, season=season, row_count=0, success=False, error="max_retries=0"),
            None,
        )

    def download_all(self) -> dict[int, Result[SeasonManifest, StatcastDownloadError]]:
        results: dict[int, Result[SeasonManifest, StatcastDownloadError]] = {}
        for season in self._config.seasons:
            results[season] = self.download_season(season)
        return results
