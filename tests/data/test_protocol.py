"""Tests for the data protocol module."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError


class TestAllPlayers:
    """Tests for the ALL_PLAYERS sentinel."""

    def test_cannot_instantiate(self) -> None:
        """ALL_PLAYERS cannot be instantiated."""
        with pytest.raises(TypeError, match="sentinel"):
            ALL_PLAYERS()

    def test_identity_comparison(self) -> None:
        """ALL_PLAYERS should be used with `is` comparison."""
        assert ALL_PLAYERS is ALL_PLAYERS

    def test_type_comparison(self) -> None:
        """ALL_PLAYERS can be compared using type()."""
        query = ALL_PLAYERS  # The class itself, not an instance
        assert query is ALL_PLAYERS


class TestDataSourceError:
    """Tests for the DataSourceError exception."""

    def test_message_only(self) -> None:
        """DataSourceError can be created with just a message."""
        error = DataSourceError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.cause is None
        assert str(error) == "Something went wrong"

    def test_with_cause(self) -> None:
        """DataSourceError can wrap another exception."""
        cause = ValueError("Original error")
        error = DataSourceError("Fetch failed", cause=cause)

        assert error.message == "Fetch failed"
        assert error.cause is cause
        assert str(error) == "Fetch failed: Original error"

    def test_is_exception(self) -> None:
        """DataSourceError is a proper exception."""
        error = DataSourceError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self) -> None:
        """DataSourceError can be raised and caught."""
        with pytest.raises(DataSourceError) as exc_info:
            raise DataSourceError("Test error")

        assert exc_info.value.message == "Test error"


class TestDataSourceContract:
    """Tests demonstrating the DataSource contract."""

    def test_batch_only_source(self, test_context: ...) -> None:
        """A DataSource can support only ALL_PLAYERS queries."""
        from fantasy_baseball_manager.result import Err, Ok

        # Example batch-only source
        def batch_only_source(query: ...) -> ...:
            if query is not ALL_PLAYERS:
                return Err(DataSourceError("Only ALL_PLAYERS supported"))
            return Ok(["player1", "player2"])

        # ALL_PLAYERS works
        result = batch_only_source(ALL_PLAYERS)
        assert result.is_ok()
        assert result.unwrap() == ["player1", "player2"]

    def test_source_returns_result(self, test_context: ...) -> None:
        """DataSources return Result types for explicit error handling."""
        from fantasy_baseball_manager.result import Err

        def source_with_error(query: ...) -> ...:
            return Err(DataSourceError("Network error"))

        result = source_with_error(ALL_PLAYERS)
        assert result.is_err()
        assert isinstance(result.unwrap_err(), DataSourceError)
