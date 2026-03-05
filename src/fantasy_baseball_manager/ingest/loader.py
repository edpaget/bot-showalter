from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import (
    Err,
    IngestError,
    LoadLog,
    Ok,
    Result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.ingest.protocols import DataSource
    from fantasy_baseball_manager.repos import ConnectionProvider, LoadLogRepo
logger = logging.getLogger(__name__)


class Loader:
    def __init__(
        self,
        source: DataSource,
        repo: Any,
        load_log_repo: LoadLogRepo,
        row_mapper: Callable[[dict[str, Any]], Any | None],
        target_table: str,
        *,
        provider: ConnectionProvider,
        log_provider: ConnectionProvider | None = None,
        post_upsert: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self._source = source
        self._repo = repo
        self._load_log_repo = load_log_repo
        self._row_mapper = row_mapper
        self._target_table = target_table
        self._provider = provider
        self._log_provider = log_provider if log_provider is not None else provider
        self._post_upsert = post_upsert

    def load(self, **fetch_params: Any) -> Result[LoadLog, IngestError]:
        with self._provider.connection() as conn, self._log_provider.connection() as log_conn:
            started_at = datetime.now(UTC).isoformat()
            t0 = time.perf_counter()
            logger.info("Loading %s from %s", self._target_table, self._source.source_detail)

            try:
                rows = self._source.fetch(**fetch_params)
            except Exception as exc:
                logger.error("Fetch failed for %s: %s", self._target_table, exc)
                finished_at = datetime.now(UTC).isoformat()
                log = LoadLog(
                    source_type=self._source.source_type,
                    source_detail=self._source.source_detail,
                    target_table=self._target_table,
                    rows_loaded=0,
                    started_at=started_at,
                    finished_at=finished_at,
                    status="error",
                    error_message=str(exc),
                )
                self._load_log_repo.insert(log)
                log_conn.commit()
                return Err(
                    IngestError(
                        message=str(exc),
                        source_type=self._source.source_type,
                        source_detail=self._source.source_detail,
                        target_table=self._target_table,
                    )
                )

            logger.debug("Fetched %d rows from %s", len(rows), self._source.source_detail)

            rows_loaded = 0
            try:
                for row in rows:
                    mapped = self._row_mapper(row)
                    if mapped is not None:
                        upsert_result = self._repo.upsert(mapped)
                        if self._post_upsert is not None:
                            self._post_upsert(upsert_result, mapped)
                        rows_loaded += 1
                conn.commit()
            except Exception as exc:
                logger.error("Processing failed for %s after %d rows: %s", self._target_table, rows_loaded, exc)
                conn.rollback()
                finished_at = datetime.now(UTC).isoformat()
                log = LoadLog(
                    source_type=self._source.source_type,
                    source_detail=self._source.source_detail,
                    target_table=self._target_table,
                    rows_loaded=0,
                    started_at=started_at,
                    finished_at=finished_at,
                    status="error",
                    error_message=str(exc),
                )
                self._load_log_repo.insert(log)
                log_conn.commit()
                return Err(
                    IngestError(
                        message=str(exc),
                        source_type=self._source.source_type,
                        source_detail=self._source.source_detail,
                        target_table=self._target_table,
                    )
                )

            finished_at = datetime.now(UTC).isoformat()
            log = LoadLog(
                source_type=self._source.source_type,
                source_detail=self._source.source_detail,
                target_table=self._target_table,
                rows_loaded=rows_loaded,
                started_at=started_at,
                finished_at=finished_at,
                status="success",
            )
            self._load_log_repo.insert(log)
            log_conn.commit()
            logger.info("Loaded %d rows into %s in %.1fs", rows_loaded, self._target_table, time.perf_counter() - t0)
            return Ok(log)
