"""Generic cache wrapper for DataSource functions.

Provides a functional wrapper that can wrap any DataSource[T] with caching.
Cache key is derived from the context's year.

Usage:
    raw_batting = create_batting_source()
    batting_source = cached(
        raw_batting,
        namespace="batting_stats",
        ttl_seconds=30 * 86400,
        serializer=BattingSerializer(),
    )

    # Use wrapped source - automatically caches ALL_PLAYERS queries
    result = batting_source(ALL_PLAYERS)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore
from fantasy_baseball_manager.context import get_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.cache.protocol import CacheStore
    from fantasy_baseball_manager.cache.serialization import Serializer
    from fantasy_baseball_manager.data.protocol import BatchDataSource, DataSource, Query


logger = logging.getLogger(__name__)

T = TypeVar("T")


@lru_cache(maxsize=4)
def _get_cache_store(db_path: Path) -> CacheStore:
    """Get or create a cache store for the given path.

    Uses LRU cache to reuse stores for the same path across calls.
    """
    return SqliteCacheStore(db_path)


def cached(
    source: DataSource[T],
    namespace: str,
    ttl_seconds: int,
    serializer: Serializer[list[T]],
) -> DataSource[T]:
    """Wrap a DataSource with caching. Cache key derived from context.

    Caches only ALL_PLAYERS queries. Single-player and batch queries
    are passed through to the source without caching.

    Args:
        source: The underlying DataSource to wrap.
        namespace: Cache namespace for this data type (e.g., "batting_stats").
        ttl_seconds: Time-to-live for cached entries in seconds.
        serializer: Serializer for converting Sequence[T] to/from strings.

    Returns:
        A wrapped DataSource that caches ALL_PLAYERS results.

    Example:
        raw_source = create_batting_source()
        cached_source = cached(
            raw_source,
            namespace="batting_stats",
            ttl_seconds=30 * 86400,  # 30 days
            serializer=DataclassListSerializer(BattingStats),
        )
    """

    def wrapper(query: Query) -> Ok[list[T]] | Ok[T] | Err[DataSourceError]:
        ctx = get_context()

        # Check cache settings from context
        if not ctx.cache_enabled:
            return source(query)

        # Only cache ALL_PLAYERS queries
        if query is not ALL_PLAYERS:
            return source(query)

        # Build cache key from context
        cache_key = f"{namespace}:{ctx.year}"
        store = _get_cache_store(ctx.db_path)

        # Check cache (unless invalidated/refresh mode)
        if not ctx.cache_invalidated:
            cached_value = store.get(namespace, cache_key)
            if cached_value is not None:
                try:
                    data = serializer.deserialize(cached_value)
                    logger.debug("Cache hit for %s (year=%d)", namespace, ctx.year)
                    return Ok(data)
                except Exception as e:
                    # Invalid cache data - log and refetch
                    logger.warning("Failed to deserialize cached %s: %s", namespace, e)

        # Fetch from source
        result = source(query)

        # Cache successful results
        if result.is_ok():
            try:
                value = result.unwrap()
                # ALL_PLAYERS returns list[T]
                serialized = serializer.serialize(value)
                store.put(namespace, cache_key, serialized, ttl_seconds)
                logger.debug("Cached %s (year=%d)", namespace, ctx.year)
            except Exception as e:
                # Serialization failure shouldn't break the data flow
                logger.warning("Failed to cache %s: %s", namespace, e)

        return result

    # Inner functions can't satisfy Protocol with overloads
    return wrapper  # type: ignore[return-value]


def cached_batch(
    source: BatchDataSource[T],
    namespace: str,
    ttl_seconds: int,
    serializer: Serializer[list[T]],
) -> BatchDataSource[T]:
    """Wrap a BatchDataSource with caching. Simpler than cached().

    Since BatchDataSource only supports ALL_PLAYERS queries, this wrapper
    provides better type inference - no casts needed on the result.

    Args:
        source: The underlying BatchDataSource to wrap.
        namespace: Cache namespace for this data type (e.g., "batting_stats").
        ttl_seconds: Time-to-live for cached entries in seconds.
        serializer: Serializer for converting Sequence[T] to/from strings.

    Returns:
        A wrapped BatchDataSource that caches results.

    Example:
        raw_source = create_batting_source()
        cached_source = cached_batch(
            raw_source,
            namespace="batting_stats",
            ttl_seconds=30 * 86400,
            serializer=DataclassListSerializer(BattingStats),
        )
        result = cached_source(ALL_PLAYERS)
        if result.is_ok():
            stats = result.unwrap()  # Type: Sequence[BattingStats] - no cast!
    """

    def wrapper(
        query: type[ALL_PLAYERS],
    ) -> Ok[list[T]] | Err[DataSourceError]:
        ctx = get_context()

        # Check cache settings from context
        if not ctx.cache_enabled:
            return source(query)

        # Build cache key from context
        cache_key = f"{namespace}:{ctx.year}"
        store = _get_cache_store(ctx.db_path)

        # Check cache (unless invalidated/refresh mode)
        if not ctx.cache_invalidated:
            cached_value = store.get(namespace, cache_key)
            if cached_value is not None:
                try:
                    data = serializer.deserialize(cached_value)
                    logger.debug("Cache hit for %s (year=%d)", namespace, ctx.year)
                    return Ok(data)
                except Exception as e:
                    logger.warning("Failed to deserialize cached %s: %s", namespace, e)

        # Fetch from source
        result = source(query)

        # Cache successful results
        if result.is_ok():
            try:
                value = result.unwrap()
                serialized = serializer.serialize(value)
                store.put(namespace, cache_key, serialized, ttl_seconds)
                logger.debug("Cached %s (year=%d)", namespace, ctx.year)
            except Exception as e:
                logger.warning("Failed to cache %s: %s", namespace, e)

        return result

    return wrapper


def cached_single(
    source: DataSource[T],
    namespace: str,
    ttl_seconds: int,
    serializer: Serializer[T],
    key_fn: ...,  # Callable[[Player], str]
) -> DataSource[T]:
    """Wrap a DataSource with per-player caching.

    Caches single-player queries by player ID. ALL_PLAYERS queries
    are passed through without caching (use `cached()` for batch caching).

    Args:
        source: The underlying DataSource to wrap.
        namespace: Cache namespace for this data type.
        ttl_seconds: Time-to-live for cached entries in seconds.
        serializer: Serializer for converting T to/from strings.
        key_fn: Function to extract cache key from a Player.

    Returns:
        A wrapped DataSource that caches single-player results.
    """

    def wrapper(query: Query) -> Ok[list[T]] | Ok[T] | Err[DataSourceError]:
        ctx = get_context()

        # Check cache settings
        if not ctx.cache_enabled:
            return source(query)

        # ALL_PLAYERS not cached here
        if query is ALL_PLAYERS:
            return source(query)

        # Handle single player
        from fantasy_baseball_manager.player.identity import Player

        if isinstance(query, Player):
            player_key = key_fn(query)
            cache_key = f"{namespace}:{ctx.year}:{player_key}"
            store = _get_cache_store(ctx.db_path)

            if not ctx.cache_invalidated:
                cached_value = store.get(namespace, cache_key)
                if cached_value is not None:
                    try:
                        data = serializer.deserialize(cached_value)
                        return Ok(data)
                    except Exception:
                        pass

            result = source(query)
            if result.is_ok():
                try:
                    serialized = serializer.serialize(result.unwrap())
                    store.put(namespace, cache_key, serialized, ttl_seconds)
                except Exception:
                    pass

            return result

        # List of players - fetch individually
        results: list[object] = []
        for player in query:
            single_result = wrapper(player)
            if single_result.is_err():
                return single_result
            results.append(single_result.unwrap())
        return Ok(results)  # type: ignore[return-value]

    # Inner functions can't satisfy Protocol with overloads
    return wrapper  # type: ignore[return-value]
