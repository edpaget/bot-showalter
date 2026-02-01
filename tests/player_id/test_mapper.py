import math
from unittest.mock import patch

import pandas as pd

from fantasy_baseball_manager.player_id.mapper import ChadwickMapper


def _make_register(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestChadwickMapper:
    def _build(self, rows: list[dict[str, object]]) -> ChadwickMapper:
        with patch("fantasy_baseball_manager.player_id.mapper.pybaseball") as mock_pb:
            mock_pb.chadwick_register.return_value = _make_register(rows)
            return ChadwickMapper()

    def test_yahoo_to_fangraphs_known_id(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": 5678.0}])
        assert mapper.yahoo_to_fangraphs("1234") == "5678"

    def test_fangraphs_to_yahoo_known_id(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": 5678.0}])
        assert mapper.fangraphs_to_yahoo("5678") == "1234"

    def test_unknown_yahoo_id_returns_none(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": 5678.0}])
        assert mapper.yahoo_to_fangraphs("9999") is None

    def test_unknown_fangraphs_id_returns_none(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": 5678.0}])
        assert mapper.fangraphs_to_yahoo("9999") is None

    def test_nan_yahoo_id_skipped(self) -> None:
        mapper = self._build([{"key_yahoo": math.nan, "key_fangraphs": 5678.0}])
        assert mapper.fangraphs_to_yahoo("5678") is None

    def test_nan_fangraphs_id_skipped(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": math.nan}])
        assert mapper.yahoo_to_fangraphs("1234") is None

    def test_multiple_players(self) -> None:
        mapper = self._build(
            [
                {"key_yahoo": 100.0, "key_fangraphs": 200.0},
                {"key_yahoo": 300.0, "key_fangraphs": 400.0},
            ]
        )
        assert mapper.yahoo_to_fangraphs("100") == "200"
        assert mapper.yahoo_to_fangraphs("300") == "400"
        assert mapper.fangraphs_to_yahoo("200") == "100"
        assert mapper.fangraphs_to_yahoo("400") == "300"

    def test_ids_stored_as_strings(self) -> None:
        mapper = self._build([{"key_yahoo": 1234.0, "key_fangraphs": 5678.0}])
        # Float IDs should be converted to int strings, not "1234.0"
        assert mapper.yahoo_to_fangraphs("1234") == "5678"
        assert mapper.yahoo_to_fangraphs("1234.0") is None

    def test_string_ids_preserved(self) -> None:
        mapper = self._build([{"key_yahoo": "sa1234", "key_fangraphs": "sa5678"}])
        assert mapper.yahoo_to_fangraphs("sa1234") == "sa5678"
        assert mapper.fangraphs_to_yahoo("sa5678") == "sa1234"

    def test_empty_register(self) -> None:
        mapper = self._build([])
        assert mapper.yahoo_to_fangraphs("1234") is None
        assert mapper.fangraphs_to_yahoo("5678") is None
