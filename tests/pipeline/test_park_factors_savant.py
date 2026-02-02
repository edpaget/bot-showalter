from unittest.mock import patch

import pytest

from fantasy_baseball_manager.pipeline.park_factors_savant import (
    SavantParkFactorProvider,
)

_SAMPLE_HTML = """
<html><head></head><body>
<script>
var defined = true;
var data = [
    {
        "venue_id": "19",
        "venue_name": "Coors Field",
        "name_display_club": "Rockies",
        "n_pa": "55000",
        "index_hr": "109",
        "index_1b": "112",
        "index_2b": "120",
        "index_3b": "180",
        "index_bb": "101",
        "index_so": "93",
        "index_runs": "115",
        "index_woba": "108",
        "index_obp": "106",
        "index_hits": "110",
        "index_wobacon": "105",
        "index_xwobacon": "101"
    },
    {
        "venue_id": "2680",
        "venue_name": "Petco Park",
        "name_display_club": "Padres",
        "n_pa": "58000",
        "index_hr": "96",
        "index_1b": "97",
        "index_2b": "90",
        "index_3b": "55",
        "index_bb": "99",
        "index_so": "102",
        "index_runs": "92",
        "index_woba": "96",
        "index_obp": "97",
        "index_hits": "96",
        "index_wobacon": "96",
        "index_xwobacon": "100"
    }
];
</script>
</body></html>
"""


class TestSavantParkFactorProvider:
    def _make_provider(self, **kwargs: object) -> SavantParkFactorProvider:
        return SavantParkFactorProvider(**kwargs)  # type: ignore[arg-type]

    def test_parses_embedded_json(self) -> None:
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=_SAMPLE_HTML):
            factors = provider.park_factors(2024)
        assert "COL" in factors
        assert "SDP" in factors
        assert len(factors) == 2

    def test_values_divided_by_100(self) -> None:
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=_SAMPLE_HTML):
            factors = provider.park_factors(2024)
        assert factors["COL"]["hr"] == pytest.approx(1.09)
        assert factors["COL"]["singles"] == pytest.approx(1.12)
        assert factors["SDP"]["hr"] == pytest.approx(0.96)

    def test_team_name_mapping(self) -> None:
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=_SAMPLE_HTML):
            factors = provider.park_factors(2024)
        # "Rockies" -> "COL", "Padres" -> "SDP"
        assert "COL" in factors
        assert "SDP" in factors

    def test_all_stats_mapped(self) -> None:
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=_SAMPLE_HTML):
            factors = provider.park_factors(2024)
        expected_stats = {
            "hr",
            "singles",
            "doubles",
            "triples",
            "bb",
            "so",
            "runs",
            "woba",
            "obp",
            "hits",
            "wobacon",
            "xwobacon",
        }
        assert set(factors["COL"].keys()) == expected_stats

    def test_unknown_team_skipped(self) -> None:
        html = """
        <script>
        var data = [{"venue_id":"999","venue_name":"Fake Park",
        "name_display_club":"Unicorns","index_hr":"100"}];
        </script>
        """
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=html):
            factors = provider.park_factors(2024)
        assert len(factors) == 0

    def test_no_data_variable_returns_empty(self) -> None:
        html = "<html><body>no data here</body></html>"
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=html):
            factors = provider.park_factors(2024)
        assert factors == {}

    def test_invalid_rolling_years_raises(self) -> None:
        with pytest.raises(ValueError, match="rolling_years"):
            SavantParkFactorProvider(rolling_years=2)

    def test_invalid_bat_side_raises(self) -> None:
        with pytest.raises(ValueError, match="bat_side"):
            SavantParkFactorProvider(bat_side="B")

    def test_coors_extended_stats(self) -> None:
        """Verify Coors advanced stats parse correctly."""
        provider = self._make_provider()
        with patch.object(provider, "_fetch", return_value=_SAMPLE_HTML):
            factors = provider.park_factors(2024)
        assert factors["COL"]["runs"] == pytest.approx(1.15)
        assert factors["COL"]["woba"] == pytest.approx(1.08)
        assert factors["COL"]["xwobacon"] == pytest.approx(1.01)
