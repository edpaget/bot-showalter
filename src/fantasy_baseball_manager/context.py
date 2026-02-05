"""Ambient context for configuration bound to current execution scope.

Provides a context system using Python's contextvars for ambient configuration
including year, cache settings, and database path. This allows data sources to
have uniform signatures while supporting multi-year workflows.

Usage:
    # Initialize at application entry
    init_context(year=2025)

    # Access context
    ctx = get_context()
    print(ctx.year)  # 2025

    # Multi-year workflows
    with new_context(year=2024):
        historical = batting_source(ALL_PLAYERS)  # Fetches 2024 data

    current = batting_source(ALL_PLAYERS)  # Fetches 2025 data
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Context:
    """Ambient configuration bound to current execution scope.

    Attributes:
        year: The season year for data operations.
        cache_enabled: Whether caching is enabled.
        cache_invalidated: Skip cache reads, allow writes (refresh mode).
        db_path: Path to the SQLite cache database.
    """

    year: int
    cache_enabled: bool = True
    cache_invalidated: bool = False
    db_path: Path = field(
        default_factory=lambda: Path("~/.config/fbm/cache.db").expanduser()
    )

    def copy(self, **overrides: int | bool | Path) -> Context:
        """Create child context with overrides. No shared references.

        Args:
            **overrides: Fields to override in the child context.

        Returns:
            A new Context with the specified overrides.
        """
        year = overrides.get("year", self.year)
        cache_enabled = overrides.get("cache_enabled", self.cache_enabled)
        cache_invalidated = overrides.get("cache_invalidated", self.cache_invalidated)
        db_path = overrides.get("db_path", self.db_path)
        return Context(
            year=year if isinstance(year, int) else self.year,
            cache_enabled=cache_enabled if isinstance(cache_enabled, bool) else self.cache_enabled,
            cache_invalidated=cache_invalidated if isinstance(cache_invalidated, bool) else self.cache_invalidated,
            db_path=db_path if isinstance(db_path, Path) else self.db_path,
        )


_context: ContextVar[Context] = ContextVar("context")


class ContextNotInitializedError(Exception):
    """Raised when get_context() is called before init_context()."""


def get_context() -> Context:
    """Get the current context.

    Returns:
        The current Context instance.

    Raises:
        ContextNotInitializedError: If context has not been initialized.
    """
    try:
        return _context.get()
    except LookupError:
        raise ContextNotInitializedError(
            "Context not initialized. Call init_context() at application entry."
        ) from None


@contextmanager
def new_context(**overrides: int | bool | Path) -> Generator[Context]:
    """Create child context with overrides. Parent context unchanged.

    Args:
        **overrides: Fields to override in the child context.

    Yields:
        The child context.

    Example:
        with new_context(year=2024):
            # All data operations use year 2024
            ...
    """
    current = get_context()
    child = current.copy(**overrides)
    token = _context.set(child)
    try:
        yield child
    finally:
        _context.reset(token)


def init_context(
    year: int,
    *,
    cache_enabled: bool = True,
    cache_invalidated: bool = False,
    db_path: Path | None = None,
) -> Context:
    """Initialize root context. Call once at application entry.

    Args:
        year: The season year for data operations.
        cache_enabled: Whether caching is enabled (default True).
        cache_invalidated: Skip cache reads, allow writes (default False).
        db_path: Path to the SQLite cache database.

    Returns:
        The initialized Context.
    """
    ctx = Context(
        year=year,
        cache_enabled=cache_enabled,
        cache_invalidated=cache_invalidated,
        db_path=db_path or Path("~/.config/fbm/cache.db").expanduser(),
    )
    _context.set(ctx)
    return ctx


def reset_context() -> None:
    """Reset the context. Primarily for testing."""
    try:
        # Get the token by setting and immediately getting
        # This is a workaround since ContextVar doesn't have a direct reset
        _context.set(Context(year=0))  # Temporary set
    except Exception:
        pass
