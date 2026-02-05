"""Tests for the result module."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.result import Err, Ok, UnwrapError


class TestOk:
    """Tests for the Ok type."""

    def test_is_ok(self) -> None:
        """Ok.is_ok() returns True."""
        result = Ok(42)
        assert result.is_ok() is True

    def test_is_err(self) -> None:
        """Ok.is_err() returns False."""
        result = Ok(42)
        assert result.is_err() is False

    def test_unwrap(self) -> None:
        """Ok.unwrap() returns the contained value."""
        result = Ok(42)
        assert result.unwrap() == 42

    def test_unwrap_or(self) -> None:
        """Ok.unwrap_or() returns the contained value, ignoring default."""
        result = Ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_err_raises(self) -> None:
        """Ok.unwrap_err() raises UnwrapError."""
        result = Ok(42)
        with pytest.raises(UnwrapError, match="Called unwrap_err on Ok value"):
            result.unwrap_err()

    def test_map(self) -> None:
        """Ok.map() applies function to contained value."""
        result = Ok(21)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 42

    def test_map_err(self) -> None:
        """Ok.map_err() returns self unchanged."""
        result: Ok[int] = Ok(42)
        mapped = result.map_err(lambda e: ValueError(str(e)))
        assert mapped.unwrap() == 42

    def test_and_then(self) -> None:
        """Ok.and_then() applies function returning Result."""
        result = Ok(21)
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.unwrap() == 42

    def test_and_then_with_err(self) -> None:
        """Ok.and_then() can return Err from function."""
        result = Ok(21)
        chained = result.and_then(lambda _: Err(ValueError("oops")))
        assert chained.is_err()

    def test_or_else(self) -> None:
        """Ok.or_else() returns self unchanged."""
        result: Ok[int] = Ok(42)
        recovered = result.or_else(lambda e: Ok(0))
        assert recovered.unwrap() == 42


class TestErr:
    """Tests for the Err type."""

    def test_is_ok(self) -> None:
        """Err.is_ok() returns False."""
        result = Err(ValueError("error"))
        assert result.is_ok() is False

    def test_is_err(self) -> None:
        """Err.is_err() returns True."""
        result = Err(ValueError("error"))
        assert result.is_err() is True

    def test_unwrap_raises(self) -> None:
        """Err.unwrap() raises UnwrapError."""
        result = Err(ValueError("test error"))
        with pytest.raises(UnwrapError, match="Called unwrap on Err value"):
            result.unwrap()

    def test_unwrap_or(self) -> None:
        """Err.unwrap_or() returns the default value."""
        result: Err[ValueError] = Err(ValueError("error"))
        assert result.unwrap_or(42) == 42

    def test_unwrap_err(self) -> None:
        """Err.unwrap_err() returns the contained error."""
        error = ValueError("test error")
        result = Err(error)
        assert result.unwrap_err() is error

    def test_map(self) -> None:
        """Err.map() returns self unchanged."""
        error = ValueError("error")
        result: Err[ValueError] = Err(error)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()
        assert mapped.unwrap_err() is error

    def test_map_err(self) -> None:
        """Err.map_err() applies function to contained error."""
        result = Err(ValueError("original"))
        mapped = result.map_err(lambda e: TypeError(f"wrapped: {e}"))
        assert mapped.is_err()
        assert isinstance(mapped.unwrap_err(), TypeError)

    def test_and_then(self) -> None:
        """Err.and_then() returns self unchanged."""
        error = ValueError("error")
        result: Err[ValueError] = Err(error)
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.is_err()
        assert chained.unwrap_err() is error

    def test_or_else(self) -> None:
        """Err.or_else() applies recovery function."""
        result = Err(ValueError("original"))
        recovered = result.or_else(lambda _: Ok(42))
        assert recovered.is_ok()
        assert recovered.unwrap() == 42

    def test_or_else_with_err(self) -> None:
        """Err.or_else() can return new Err."""
        result = Err(ValueError("original"))
        recovered = result.or_else(lambda e: Err(TypeError(f"wrapped: {e}")))
        assert recovered.is_err()
        assert isinstance(recovered.unwrap_err(), TypeError)


class TestResultChaining:
    """Tests for chaining Result operations."""

    def test_chain_successful_operations(self) -> None:
        """Multiple successful operations can be chained."""

        def parse(s: str) -> Ok[int] | Err[ValueError]:
            try:
                return Ok(int(s))
            except ValueError as e:
                return Err(e)

        def double(n: int) -> Ok[int] | Err[ValueError]:
            return Ok(n * 2)

        result = parse("21").and_then(double)
        assert result.unwrap() == 42

    def test_chain_stops_at_first_error(self) -> None:
        """Chaining stops at first error."""

        def parse(s: str) -> Ok[int] | Err[ValueError]:
            try:
                return Ok(int(s))
            except ValueError as e:
                return Err(e)

        def double(n: int) -> Ok[int] | Err[ValueError]:
            return Ok(n * 2)

        result = parse("not a number").and_then(double)
        assert result.is_err()


class TestResultTypeAnnotations:
    """Tests for Result type annotation usage."""

    def test_ok_with_complex_type(self) -> None:
        """Ok works with complex nested types."""
        result: Ok[list[dict[str, int]]] = Ok([{"a": 1}, {"b": 2}])
        assert result.unwrap() == [{"a": 1}, {"b": 2}]

    def test_err_with_custom_exception(self) -> None:
        """Err works with custom exception types."""

        class CustomError(Exception):
            def __init__(self, code: int, message: str) -> None:
                self.code = code
                self.message = message

        result = Err(CustomError(404, "Not found"))
        error = result.unwrap_err()
        assert error.code == 404
        assert error.message == "Not found"
