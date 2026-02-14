from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    @property
    def source_type(self) -> str: ...

    @property
    def source_detail(self) -> str: ...

    def fetch(self, **params: Any) -> pd.DataFrame: ...
