import logging
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from fantasy_baseball_manager.domain.errors import IngestError
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.result import Err, Ok, Result
from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.repos.protocols import LoadLogRepo

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
        conn: sqlite3.Connection,
        log_conn: sqlite3.Connection | None = None,
        post_upsert: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self._source = source
        self._repo = repo
        self._load_log_repo = load_log_repo
        self._row_mapper = row_mapper
        self._target_table = target_table
        self._conn = conn
        self._log_conn = log_conn if log_conn is not None else conn
        self._post_upsert = post_upsert

    def load(self, **fetch_params: Any) -> Result[LoadLog, IngestError]:
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        logger.info("Loading %s from %s", self._target_table, self._source.source_detail)

        try:
            rows = self._source.fetch(**fetch_params)
        except Exception as exc:
            logger.error("Fetch failed for %s: %s", self._target_table, exc)
            finished_at = datetime.now(timezone.utc).isoformat()
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
            self._log_conn.commit()
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
            self._conn.commit()
        except Exception as exc:
            logger.error("Processing failed for %s after %d rows: %s", self._target_table, rows_loaded, exc)
            self._conn.rollback()
            finished_at = datetime.now(timezone.utc).isoformat()
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
            self._log_conn.commit()
            return Err(
                IngestError(
                    message=str(exc),
                    source_type=self._source.source_type,
                    source_detail=self._source.source_detail,
                    target_table=self._target_table,
                )
            )

        finished_at = datetime.now(timezone.utc).isoformat()
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
        self._log_conn.commit()
        logger.info("Loaded %d rows into %s in %.1fs", rows_loaded, self._target_table, time.perf_counter() - t0)
        return Ok(log)
