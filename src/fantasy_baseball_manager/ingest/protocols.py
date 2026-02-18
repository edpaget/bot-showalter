from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    @property
    def source_type(self) -> str: ...

    @property
    def source_detail(self) -> str: ...

    def fetch(self, **params: Any) -> list[dict[str, Any]]: ...
