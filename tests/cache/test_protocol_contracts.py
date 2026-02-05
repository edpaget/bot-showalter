"""Contract tests for cache protocol implementations.

These tests verify that all implementations of the CacheStore protocol
satisfy the protocol's contract correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fantasy_baseball_manager.cache.protocol import CacheStore
from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore

# =============================================================================
# CacheStore Protocol Contract Tests
# =============================================================================


@pytest.fixture
def temp_cache_path() -> Path:
    """Create a temporary file path for cache testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def sqlite_cache_store(temp_cache_path: Path) -> SqliteCacheStore:
    """Create a SqliteCacheStore for testing."""
    return SqliteCacheStore(temp_cache_path)


# All CacheStore implementations
CACHE_STORES: list[str] = ["sqlite"]


@pytest.fixture(params=CACHE_STORES)
def cache_store(request: pytest.FixtureRequest, temp_cache_path: Path) -> CacheStore:
    """Parametrized fixture that yields each CacheStore implementation."""
    if request.param == "sqlite":
        return SqliteCacheStore(temp_cache_path)
    raise ValueError(f"Unknown cache store type: {request.param}")


class TestCacheStoreContract:
    """Contract tests for CacheStore protocol implementations."""

    def test_has_get_method(self, cache_store: CacheStore) -> None:
        """Verify the implementation has a get method."""
        assert hasattr(cache_store, "get")
        assert callable(cache_store.get)

    def test_has_put_method(self, cache_store: CacheStore) -> None:
        """Verify the implementation has a put method."""
        assert hasattr(cache_store, "put")
        assert callable(cache_store.put)

    def test_has_invalidate_method(self, cache_store: CacheStore) -> None:
        """Verify the implementation has an invalidate method."""
        assert hasattr(cache_store, "invalidate")
        assert callable(cache_store.invalidate)

    def test_get_returns_none_for_missing_key(self, cache_store: CacheStore) -> None:
        """Verify get returns None for a key that doesn't exist."""
        result = cache_store.get("test_namespace", "nonexistent_key")
        assert result is None

    def test_put_and_get_roundtrip(self, cache_store: CacheStore) -> None:
        """Verify put stores a value that can be retrieved with get."""
        namespace = "test_namespace"
        key = "test_key"
        value = "test_value"

        cache_store.put(namespace, key, value, ttl_seconds=3600)
        result = cache_store.get(namespace, key)

        assert result == value

    def test_put_overwrites_existing_value(self, cache_store: CacheStore) -> None:
        """Verify put overwrites an existing value."""
        namespace = "test_namespace"
        key = "test_key"

        cache_store.put(namespace, key, "first_value", ttl_seconds=3600)
        cache_store.put(namespace, key, "second_value", ttl_seconds=3600)
        result = cache_store.get(namespace, key)

        assert result == "second_value"

    def test_different_namespaces_are_isolated(self, cache_store: CacheStore) -> None:
        """Verify different namespaces don't interfere with each other."""
        key = "same_key"

        cache_store.put("namespace_a", key, "value_a", ttl_seconds=3600)
        cache_store.put("namespace_b", key, "value_b", ttl_seconds=3600)

        assert cache_store.get("namespace_a", key) == "value_a"
        assert cache_store.get("namespace_b", key) == "value_b"

    def test_invalidate_removes_specific_key(self, cache_store: CacheStore) -> None:
        """Verify invalidate removes a specific key."""
        namespace = "test_namespace"
        key = "test_key"

        cache_store.put(namespace, key, "test_value", ttl_seconds=3600)
        cache_store.invalidate(namespace, key)
        result = cache_store.get(namespace, key)

        assert result is None

    def test_invalidate_removes_all_keys_in_namespace(self, cache_store: CacheStore) -> None:
        """Verify invalidate with no key removes all keys in namespace."""
        namespace = "test_namespace"

        cache_store.put(namespace, "key1", "value1", ttl_seconds=3600)
        cache_store.put(namespace, "key2", "value2", ttl_seconds=3600)
        cache_store.invalidate(namespace)

        assert cache_store.get(namespace, "key1") is None
        assert cache_store.get(namespace, "key2") is None

    def test_invalidate_namespace_preserves_other_namespaces(self, cache_store: CacheStore) -> None:
        """Verify invalidating a namespace doesn't affect other namespaces."""
        cache_store.put("namespace_a", "key", "value_a", ttl_seconds=3600)
        cache_store.put("namespace_b", "key", "value_b", ttl_seconds=3600)

        cache_store.invalidate("namespace_a")

        assert cache_store.get("namespace_a", "key") is None
        assert cache_store.get("namespace_b", "key") == "value_b"

    def test_put_returns_none(self, cache_store: CacheStore) -> None:
        """Verify put returns None."""
        result = cache_store.put("namespace", "key", "value", ttl_seconds=3600)
        assert result is None

    def test_invalidate_returns_none(self, cache_store: CacheStore) -> None:
        """Verify invalidate returns None."""
        result = cache_store.invalidate("namespace", "key")
        assert result is None

    def test_handles_empty_string_value(self, cache_store: CacheStore) -> None:
        """Verify cache handles empty string values correctly."""
        namespace = "test_namespace"
        key = "test_key"

        cache_store.put(namespace, key, "", ttl_seconds=3600)
        result = cache_store.get(namespace, key)

        assert result == ""

    def test_handles_long_value(self, cache_store: CacheStore) -> None:
        """Verify cache handles long values correctly."""
        namespace = "test_namespace"
        key = "test_key"
        long_value = "x" * 100000  # 100KB of data

        cache_store.put(namespace, key, long_value, ttl_seconds=3600)
        result = cache_store.get(namespace, key)

        assert result == long_value

    def test_handles_special_characters_in_value(self, cache_store: CacheStore) -> None:
        """Verify cache handles special characters in values correctly."""
        namespace = "test_namespace"
        key = "test_key"
        special_value = '{"key": "value", "unicode": "日本語", "emoji": "🎉"}'

        cache_store.put(namespace, key, special_value, ttl_seconds=3600)
        result = cache_store.get(namespace, key)

        assert result == special_value
