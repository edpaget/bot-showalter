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

    def test_reads_csv_into_dataframe(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        source = CsvSource(csv_file)

        df = source.fetch()

        assert list(df.columns) == ["name", "age"]
        assert len(df) == 2
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["age"] == 25

    def test_passes_params_to_read_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.tsv"
        csv_file.write_text("name\tage\nAlice\t30\n")
        source = CsvSource(csv_file)

        df = source.fetch(sep="\t")

        assert list(df.columns) == ["name", "age"]
        assert len(df) == 1

    def test_nonexistent_file_raises(self) -> None:
        source = CsvSource("/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            source.fetch()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        source = CsvSource(str(csv_file))

        df = source.fetch()
        assert len(df) == 1
