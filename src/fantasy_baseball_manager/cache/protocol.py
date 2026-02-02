from __future__ import annotations

from typing import Protocol


class CacheStore(Protocol):
    def get(self, namespace: str, key: str) -> str | None: ...

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None: ...

    def invalidate(self, namespace: str, key: str | None = None) -> None: ...
