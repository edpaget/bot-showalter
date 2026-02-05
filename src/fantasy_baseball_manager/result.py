"""Result type for explicit error handling.

Provides a Result[T, E] type with Ok and Err variants for explicit error handling
in data sources and other operations that can fail.

Usage:
    def fetch_data() -> Result[list[Player], DataSourceError]:
        try:
            data = _fetch_from_api()
            return Ok(data)
        except Exception as e:
            return Err(DataSourceError(str(e)))

    result = fetch_data()
    if result.is_ok():
        players = result.unwrap()
    else:
        error = result.unwrap_err()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast, final

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=Exception)
F = TypeVar("F", bound=Exception)


class UnwrapError(Exception):
    """Raised when unwrap() is called on an Err or unwrap_err() on an Ok."""


@final
@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Represents a successful result containing a value."""

    _value: T

    def is_ok(self) -> bool:
        """Returns True if this is an Ok result."""
        return True

    def is_err(self) -> bool:
        """Returns False for Ok results."""
        return False

    def unwrap(self) -> T:
        """Returns the contained value."""
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Returns the contained value, ignoring the default."""
        return self._value

    def unwrap_err(self) -> Exception:
        """Raises UnwrapError since this is not an Err."""
        raise UnwrapError("Called unwrap_err on Ok value")

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """Applies fn to the contained value, returning Ok(fn(value))."""
        return Ok(fn(self._value))

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """Returns self unchanged since this is Ok."""
        return cast("Result[T, F]", self)

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Applies fn to the contained value, returning its result."""
        return fn(self._value)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Returns self unchanged since this is Ok."""
        return cast("Result[T, F]", self)


@final
@dataclass(frozen=True, slots=True)
class Err[E: Exception]:
    """Represents a failed result containing an error."""

    _error: E

    def is_ok(self) -> bool:
        """Returns False for Err results."""
        return False

    def is_err(self) -> bool:
        """Returns True if this is an Err result."""
        return True

    def unwrap[_T](self) -> _T:  # noqa: UP049 # pyright: ignore[reportInvalidTypeVarUse]
        """Raises UnwrapError with the contained error."""
        raise UnwrapError(f"Called unwrap on Err value: {self._error}")

    def unwrap_or[_T](self, default: _T) -> _T:  # noqa: UP049
        """Returns the default value."""
        return default

    def unwrap_err(self) -> E:
        """Returns the contained error."""
        return self._error

    def map[_T, _U](  # noqa: UP049
        self, fn: Callable[[_T], _U]  # pyright: ignore[reportInvalidTypeVarUse]
    ) -> Result[_U, E]:
        """Returns self unchanged since this is Err."""
        return cast("Result[_U, E]", self)

    def map_err[_F: Exception](self, fn: Callable[[E], _F]) -> Result[object, _F]:  # noqa: UP049
        """Applies fn to the contained error, returning Err(fn(error))."""
        return Err(fn(self._error))

    def and_then[_T, _U](  # noqa: UP049
        self, fn: Callable[[_T], Result[_U, E]]  # pyright: ignore[reportInvalidTypeVarUse]
    ) -> Result[_U, E]:
        """Returns self unchanged since this is Err."""
        return cast("Result[_U, E]", self)

    def or_else[_F: Exception](self, fn: Callable[[E], Result[object, _F]]) -> Result[object, _F]:  # noqa: UP049
        """Applies fn to the contained error, returning its result."""
        return fn(self._error)


# Type alias for Result
Result = Ok[T] | Err[E]
