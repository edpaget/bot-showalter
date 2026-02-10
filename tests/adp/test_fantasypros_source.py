"""Tests for FantasyPros ADP parser and data source."""

from pathlib import Path

import pytest

from fantasy_baseball_manager.adp.fantasypros_source import (
    FantasyProsADPDataSource,
    FantasyProsADPParser,
)
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "fantasypros_sample.csv"


class TestFantasyProsADPParser:
    """Tests for FantasyProsADPParser."""

    def test_parses_names(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        names = [e.name for e in entries]
        assert "Aaron Judge" in names
        assert "Bobby Witt Jr." in names

    def test_parses_adp_values(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        by_name = {e.name: e for e in entries}
        assert by_name["Aaron Judge"].adp == 1.4
        assert by_name["Bobby Witt Jr."].adp == 2.3

    def test_sorted_by_adp(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        adp_values = [e.adp for e in entries]
        assert adp_values == sorted(adp_values)

    def test_single_position(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        by_name = {e.name: e for e in entries}
        assert by_name["Bobby Witt Jr."].positions == ("SS",)

    def test_multi_position_sorted(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        by_name = {e.name: e for e in entries}
        assert by_name["Aaron Judge"].positions == ("DH", "RF")

    def test_accented_names_preserved(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        by_name = {e.name: e for e in entries}
        assert "José Ramírez" in by_name

    def test_skips_rows_with_empty_avg(self) -> None:
        entries = FantasyProsADPParser().parse(SAMPLE_CSV)
        names = [e.name for e in entries]
        assert "Empty ADP Player" not in names
        assert len(entries) == 5

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            FantasyProsADPParser().parse(Path("/nonexistent/file.csv"))


class TestFantasyProsADPDataSource:
    """Tests for the DataSource adapter."""

    def test_returns_ok_for_all_players(self) -> None:
        source = FantasyProsADPDataSource(SAMPLE_CSV)
        result = source(ALL_PLAYERS)
        assert result.is_ok()
        entries = result.unwrap()
        assert len(entries) == 5

    def test_returns_err_for_missing_file(self) -> None:
        source = FantasyProsADPDataSource(Path("/nonexistent.csv"))
        result = source(ALL_PLAYERS)
        assert result.is_err()
