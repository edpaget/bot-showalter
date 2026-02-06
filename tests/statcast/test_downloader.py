from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball_manager.statcast.downloader import StatcastDownloader
from fantasy_baseball_manager.statcast.models import ChunkResult, DownloadConfig, SeasonManifest
from fantasy_baseball_manager.statcast.store import StatcastStore


class FakeFetcher:
    def __init__(
        self,
        data: dict[date, pd.DataFrame] | None = None,
        errors: set[date] | None = None,
    ) -> None:
        self._data = data or {}
        self._errors = errors or set()
        self.calls: list[date] = []

    def fetch_day(self, day: date) -> pd.DataFrame:
        self.calls.append(day)
        if day in self._errors:
            raise RuntimeError(f"Simulated failure for {day}")
        return self._data.get(day, pd.DataFrame())


def _make_sample_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitch_type": ["FF"] * n,
            "release_speed": [95.0] * n,
        }
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def store(data_dir: Path) -> StatcastStore:
    return StatcastStore(data_dir=data_dir)


class TestStatcastDownloader:
    def test_downloads_all_days_in_season(self, data_dir: Path, store: StatcastStore) -> None:
        # Use a very short custom season via 2020 (fewer days) but still too many.
        # Instead, pre-populate manifest to leave only 2 days pending.
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        # Leave only first 2 days pending
        already_fetched = {d.isoformat() for d in all_dates[2:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        sample = _make_sample_df(5)
        data = {all_dates[0]: sample, all_dates[1]: sample}
        fetcher = FakeFetcher(data=data)
        config = DownloadConfig(seasons=(2024,), data_dir=data_dir)
        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)

        result = downloader.download_season(2024)
        assert result.is_ok()
        manifest = result.unwrap()
        assert all_dates[0].isoformat() in manifest.fetched_dates
        assert all_dates[1].isoformat() in manifest.fetched_dates

    def test_skips_already_fetched_dates(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        already_fetched = {d.isoformat() for d in all_dates}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched, total_rows=100)
        store.save_manifest(manifest)

        fetcher = FakeFetcher()
        config = DownloadConfig(seasons=(2024,), data_dir=data_dir)
        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)

        result = downloader.download_season(2024)
        assert result.is_ok()
        assert fetcher.calls == []

    def test_records_progress_callback(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        already_fetched = {d.isoformat() for d in all_dates[1:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        sample = _make_sample_df(3)
        fetcher = FakeFetcher(data={all_dates[0]: sample})
        config = DownloadConfig(seasons=(2024,), data_dir=data_dir)
        results: list[ChunkResult] = []
        downloader = StatcastDownloader(
            fetcher=fetcher,
            store=store,
            config=config,
            progress_callback=results.append,
        )

        downloader.download_season(2024)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].row_count == 3

    def test_retries_on_failure_then_succeeds(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        already_fetched = {d.isoformat() for d in all_dates[1:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        attempt_count = 0
        sample = _make_sample_df(2)

        class RetryFetcher:
            def fetch_day(self, day: date) -> pd.DataFrame:
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise RuntimeError("Transient error")
                return sample

        fetcher = RetryFetcher()
        config = DownloadConfig(
            seasons=(2024,),
            data_dir=data_dir,
            max_retries=5,
            base_delay=0.0,
            max_delay=0.0,
        )
        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)

        result = downloader.download_season(2024)
        assert result.is_ok()
        assert attempt_count == 3

    def test_permanent_failure_records_error(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        already_fetched = {d.isoformat() for d in all_dates[1:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        fetcher = FakeFetcher(errors={all_dates[0]})
        config = DownloadConfig(
            seasons=(2024,),
            data_dir=data_dir,
            max_retries=2,
            base_delay=0.0,
            max_delay=0.0,
        )
        results: list[ChunkResult] = []
        downloader = StatcastDownloader(
            fetcher=fetcher,
            store=store,
            config=config,
            progress_callback=results.append,
        )

        result = downloader.download_season(2024)
        assert result.is_ok()
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None

    def test_download_all_processes_multiple_seasons(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        # Pre-fetch all dates for both seasons
        for season in (2023, 2024):
            all_dates = game_dates(season)
            already_fetched = {d.isoformat() for d in all_dates}
            manifest = SeasonManifest(season=season, fetched_dates=already_fetched)
            store.save_manifest(manifest)

        fetcher = FakeFetcher()
        config = DownloadConfig(seasons=(2023, 2024), data_dir=data_dir)
        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)

        results = downloader.download_all()
        assert 2023 in results
        assert 2024 in results
        assert results[2023].is_ok()
        assert results[2024].is_ok()

    def test_force_refetches_all_dates(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        # Only leave first 2 days in the season for speed
        already_fetched = {d.isoformat() for d in all_dates[2:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        sample = _make_sample_df(1)
        data = {d: sample for d in all_dates[:3]}
        fetcher = FakeFetcher(data=data)
        config = DownloadConfig(seasons=(2024,), data_dir=data_dir, force=True)

        # Pre-populate to leave only 3 pending with force
        already_fetched = {d.isoformat() for d in all_dates[3:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)
        result = downloader.download_season(2024)
        assert result.is_ok()
        # With force, it should fetch all dates, not skip any
        assert len(fetcher.calls) == len(all_dates)

    def test_empty_dataframe_still_records_date(self, data_dir: Path, store: StatcastStore) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        already_fetched = {d.isoformat() for d in all_dates[1:]}
        manifest = SeasonManifest(season=2024, fetched_dates=already_fetched)
        store.save_manifest(manifest)

        fetcher = FakeFetcher()  # Returns empty DataFrames
        config = DownloadConfig(seasons=(2024,), data_dir=data_dir)
        downloader = StatcastDownloader(fetcher=fetcher, store=store, config=config)

        result = downloader.download_season(2024)
        assert result.is_ok()
        manifest = result.unwrap()
        assert all_dates[0].isoformat() in manifest.fetched_dates
