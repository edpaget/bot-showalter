from typing import Any

import pandas as pd
from pybaseball import chadwick_register


class ChadwickSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "chadwick_register"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return chadwick_register()
