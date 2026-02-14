import pytest

from fantasy_baseball_manager.ingest.date_utils import chunk_date_range


class TestChunkDateRange:
    def test_single_day(self) -> None:
        result = chunk_date_range("2024-06-15", "2024-06-15")
        assert result == [("2024-06-15", "2024-06-15")]

    def test_exact_one_week(self) -> None:
        result = chunk_date_range("2024-06-01", "2024-06-07")
        assert result == [("2024-06-01", "2024-06-07")]

    def test_two_weeks(self) -> None:
        result = chunk_date_range("2024-06-01", "2024-06-14")
        assert result == [
            ("2024-06-01", "2024-06-07"),
            ("2024-06-08", "2024-06-14"),
        ]

    def test_partial_last_chunk(self) -> None:
        result = chunk_date_range("2024-06-01", "2024-06-10")
        assert result == [
            ("2024-06-01", "2024-06-07"),
            ("2024-06-08", "2024-06-10"),
        ]

    def test_custom_chunk_days(self) -> None:
        result = chunk_date_range("2024-06-01", "2024-06-05", chunk_days=3)
        assert result == [
            ("2024-06-01", "2024-06-03"),
            ("2024-06-04", "2024-06-05"),
        ]

    def test_start_after_end_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            chunk_date_range("2024-06-15", "2024-06-01")
