"""Serialization protocols and implementations for cache storage.

Provides a Serializer protocol and implementations for converting data to/from
strings for cache storage.

Usage:
    serializer = DataclassListSerializer(BattingStats)
    cached_str = serializer.serialize(stats_list)
    stats_list = serializer.deserialize(cached_str)
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class Serializer(Protocol[T]):
    """Protocol for serializing and deserializing values for cache storage."""

    def serialize(self, value: T) -> str:
        """Convert a value to a string for cache storage."""
        ...

    def deserialize(self, data: str) -> T:
        """Convert a cached string back to the original value."""
        ...


class DataclassListSerializer[T]:
    """Serializer for lists of frozen dataclasses.

    Uses JSON for serialization. Supports nested dataclasses and enums.

    Args:
        dataclass_type: The dataclass type to serialize/deserialize.
    """

    def __init__(self, dataclass_type: type[T]) -> None:
        self._dataclass_type = dataclass_type
        # Runtime check for dataclass, but type checker can't narrow generic T
        if not is_dataclass(dataclass_type):  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError(f"{dataclass_type} is not a dataclass")
        self._field_types = {f.name: f.type for f in fields(dataclass_type)}

    def serialize(self, value: Sequence[T]) -> str:
        """Convert a list of dataclasses to JSON string."""
        return json.dumps([self._to_dict(item) for item in value])

    def deserialize(self, data: str) -> list[T]:
        """Convert a JSON string back to a list of dataclasses."""
        raw_list = json.loads(data)
        return [self._from_dict(item) for item in raw_list]

    def _to_dict(self, obj: T) -> dict[str, Any]:
        """Convert a dataclass to a dictionary, handling nested types."""
        if not is_dataclass(obj):
            raise TypeError(f"{obj} is not a dataclass instance")
        return asdict(obj)  # type: ignore[arg-type]

    def _from_dict(self, data: dict[str, Any]) -> T:
        """Convert a dictionary back to a dataclass instance."""
        return self._dataclass_type(**data)


class JsonSerializer[T]:
    """Generic JSON serializer for simple types.

    Works with any JSON-serializable type (dicts, lists, primitives).
    """

    def serialize(self, value: T) -> str:
        """Convert a value to JSON string."""
        return json.dumps(value)

    def deserialize(self, data: str) -> T:
        """Convert a JSON string back to the original type."""
        return json.loads(data)


class StringSerializer:
    """Passthrough serializer for string values."""

    def serialize(self, value: str) -> str:
        """Return the string unchanged."""
        return value

    def deserialize(self, data: str) -> str:
        """Return the string unchanged."""
        return data
