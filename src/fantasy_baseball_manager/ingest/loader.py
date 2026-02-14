import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.repos.protocols import LoadLogRepo, PlayerRepo


class StatsLoader:
    def __init__(
        self,
        source: DataSource,
        repo: Any,
        load_log_repo: LoadLogRepo,
        row_mapper: Callable[[pd.Series], Any | None],
        target_table: str,
        *,
        conn: sqlite3.Connection,
        log_conn: sqlite3.Connection | None = None,
    ) -> None:
        self._source = source
        self._repo = repo
        self._load_log_repo = load_log_repo
        self._row_mapper = row_mapper
        self._target_table = target_table
        self._conn = conn
        self._log_conn = log_conn if log_conn is not None else conn

    def load(self, **fetch_params: Any) -> LoadLog:
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            df = self._source.fetch(**fetch_params)
        except Exception as exc:
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
            raise

        rows_loaded = 0
        try:
            for _, row in df.iterrows():
                mapped = self._row_mapper(row)
                if mapped is not None:
                    self._repo.upsert(mapped)
                    rows_loaded += 1
            self._conn.commit()
        except Exception as exc:
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
            raise

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
        return log


class PlayerLoader:
    def __init__(
        self,
        source: DataSource,
        player_repo: PlayerRepo,
        load_log_repo: LoadLogRepo,
        row_mapper: Callable[[pd.Series], Player | None],
        *,
        conn: sqlite3.Connection,
    ) -> None:
        self._source = source
        self._player_repo = player_repo
        self._load_log_repo = load_log_repo
        self._row_mapper = row_mapper
        self._conn = conn

    def load(self, **fetch_params: Any) -> LoadLog:
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            df = self._source.fetch(**fetch_params)
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            log = LoadLog(
                source_type=self._source.source_type,
                source_detail=self._source.source_detail,
                target_table="player",
                rows_loaded=0,
                started_at=started_at,
                finished_at=finished_at,
                status="error",
                error_message=str(exc),
            )
            self._load_log_repo.insert(log)
            self._conn.commit()
            raise

        rows_loaded = 0
        try:
            for _, row in df.iterrows():
                player = self._row_mapper(row)
                if player is not None:
                    self._player_repo.upsert(player)
                    rows_loaded += 1
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            finished_at = datetime.now(timezone.utc).isoformat()
            log = LoadLog(
                source_type=self._source.source_type,
                source_detail=self._source.source_detail,
                target_table="player",
                rows_loaded=0,
                started_at=started_at,
                finished_at=finished_at,
                status="error",
                error_message=str(exc),
            )
            self._load_log_repo.insert(log)
            self._conn.commit()
            raise

        finished_at = datetime.now(timezone.utc).isoformat()
        log = LoadLog(
            source_type=self._source.source_type,
            source_detail=self._source.source_detail,
            target_table="player",
            rows_loaded=rows_loaded,
            started_at=started_at,
            finished_at=finished_at,
            status="success",
        )
        self._load_log_repo.insert(log)
        self._conn.commit()
        return log
