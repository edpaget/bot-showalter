from pathlib import Path

import pytest

from fantasy_baseball_manager.ingest.csv_source import CsvSource
from fantasy_baseball_manager.ingest.protocols import DataSource


class TestCsvSource:
    def test_satisfies_datasource_protocol(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        assert isinstance(CsvSource(csv_file), DataSource)

    def test_source_type(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        assert CsvSource(csv_file).source_type == "csv"

    def test_source_detail(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        assert CsvSource(csv_file).source_detail == str(csv_file)

    def test_reads_csv_into_list_of_dicts(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        source = CsvSource(csv_file)

        rows = source.fetch()

        assert len(rows) == 2
        assert list(rows[0].keys()) == ["name", "age"]
        assert rows[0]["name"] == "Alice"
        assert rows[1]["age"] == "25"  # csv.DictReader returns strings

    def test_passes_params_to_reader(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.tsv"
        csv_file.write_text("name\tage\nAlice\t30\n")
        source = CsvSource(csv_file)

        rows = source.fetch(sep="\t")

        assert list(rows[0].keys()) == ["name", "age"]
        assert len(rows) == 1

    def test_nonexistent_file_raises(self) -> None:
        source = CsvSource("/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            source.fetch()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        source = CsvSource(str(csv_file))

        rows = source.fetch()
        assert len(rows) == 1
