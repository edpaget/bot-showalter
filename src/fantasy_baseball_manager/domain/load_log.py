from dataclasses import dataclass


@dataclass(frozen=True)
class LoadLog:
    source_type: str
    source_detail: str
    target_table: str
    rows_loaded: int
    started_at: str
    finished_at: str
    status: str
    id: int | None = None
    error_message: str | None = None
