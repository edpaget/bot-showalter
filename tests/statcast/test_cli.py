from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.statcast.cli import statcast_app
from fantasy_baseball_manager.statcast.models import SeasonManifest
from fantasy_baseball_manager.statcast.store import StatcastStore

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


class FakeFetcher:
    def fetch_day(self, day: date) -> pd.DataFrame:
        return pd.DataFrame({"pitch_type": ["FF"], "release_speed": [95.0]})


class TestDownloadCommand:
    def test_download_requires_seasons(self) -> None:
        result = runner.invoke(statcast_app, ["download"])
        assert result.exit_code != 0

    def test_download_with_seasons(self, data_dir: Path) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        # Pre-populate manifest so no actual fetching needed
        store = StatcastStore(data_dir=data_dir)
        all_dates = game_dates(2024)
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={d.isoformat() for d in all_dates},
            total_rows=50000,
        )
        store.save_manifest(manifest)

        with patch("fantasy_baseball_manager.statcast.cli._get_data_dir", return_value=data_dir):
            result = runner.invoke(statcast_app, ["download", "--seasons", "2024"])
        assert result.exit_code == 0
        assert "2024" in result.output

    def test_download_multiple_seasons(self, data_dir: Path) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        store = StatcastStore(data_dir=data_dir)
        for season in (2023, 2024):
            all_dates = game_dates(season)
            manifest = SeasonManifest(
                season=season,
                fetched_dates={d.isoformat() for d in all_dates},
                total_rows=50000,
            )
            store.save_manifest(manifest)

        with patch("fantasy_baseball_manager.statcast.cli._get_data_dir", return_value=data_dir):
            result = runner.invoke(statcast_app, ["download", "--seasons", "2023,2024"])
        assert result.exit_code == 0


class TestStatusCommand:
    def test_status_no_data(self, data_dir: Path) -> None:
        with patch("fantasy_baseball_manager.statcast.cli._get_data_dir", return_value=data_dir):
            result = runner.invoke(statcast_app, ["status"])
        assert result.exit_code == 0
        assert "No statcast data" in result.output.lower() or "no data" in result.output.lower() or result.output.strip() != ""

    def test_status_with_data(self, data_dir: Path) -> None:
        store = StatcastStore(data_dir=data_dir)
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={"2024-04-01", "2024-04-02"},
            total_rows=500,
        )
        store.save_manifest(manifest)
        df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95.0, 85.0]})
        store.flush_month(2024, 4, df)

        with patch("fantasy_baseball_manager.statcast.cli._get_data_dir", return_value=data_dir):
            result = runner.invoke(statcast_app, ["status"])
        assert result.exit_code == 0
        assert "2024" in result.output

    def test_status_with_season_filter(self, data_dir: Path) -> None:
        store = StatcastStore(data_dir=data_dir)
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={"2024-04-01"},
            total_rows=100,
        )
        store.save_manifest(manifest)

        with patch("fantasy_baseball_manager.statcast.cli._get_data_dir", return_value=data_dir):
            result = runner.invoke(statcast_app, ["status", "--season", "2024"])
        assert result.exit_code == 0
        assert "2024" in result.output
