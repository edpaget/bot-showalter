import pytest

from fantasy_baseball_manager.domain.load_log import LoadLog


class TestLoadLog:
    def test_construct_with_required_fields(self) -> None:
        log = LoadLog(
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
            rows_loaded=150,
            started_at="2025-01-15T10:00:00",
            finished_at="2025-01-15T10:00:05",
            status="success",
        )
        assert log.source_type == "pybaseball"
        assert log.target_table == "batting_stats"
        assert log.rows_loaded == 150
        assert log.status == "success"

    def test_optional_fields(self) -> None:
        log = LoadLog(
            source_type="csv",
            source_detail="projections.csv",
            target_table="projection",
            rows_loaded=0,
            started_at="2025-01-15T10:00:00",
            finished_at="2025-01-15T10:00:01",
            status="error",
            error_message="File not found",
        )
        assert log.error_message == "File not found"

    def test_error_message_defaults_to_none(self) -> None:
        log = LoadLog(
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
            rows_loaded=150,
            started_at="2025-01-15T10:00:00",
            finished_at="2025-01-15T10:00:05",
            status="success",
        )
        assert log.id is None
        assert log.error_message is None

    def test_frozen(self) -> None:
        log = LoadLog(
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
            rows_loaded=150,
            started_at="2025-01-15T10:00:00",
            finished_at="2025-01-15T10:00:05",
            status="success",
        )
        with pytest.raises(AttributeError):
            log.status = "error"  # type: ignore[misc]
