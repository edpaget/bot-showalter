from datetime import date

from fantasy_baseball_manager.statcast.chunker import pending_chunks
from fantasy_baseball_manager.statcast.models import DateChunk, SeasonManifest


class TestPendingChunks:
    def test_all_dates_pending_when_manifest_empty(self) -> None:
        manifest = SeasonManifest(season=2024)
        chunks = pending_chunks(2024, manifest)
        assert len(chunks) > 0
        assert all(isinstance(c, DateChunk) for c in chunks)
        assert all(c.season == 2024 for c in chunks)

    def test_skips_already_fetched_dates(self) -> None:
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={"2024-03-20", "2024-03-21"},
        )
        chunks = pending_chunks(2024, manifest)
        chunk_dates = {c.date for c in chunks}
        assert date(2024, 3, 20) not in chunk_dates
        assert date(2024, 3, 21) not in chunk_dates
        assert date(2024, 3, 22) in chunk_dates

    def test_returns_empty_when_all_fetched(self) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={d.isoformat() for d in all_dates},
        )
        chunks = pending_chunks(2024, manifest)
        assert chunks == []

    def test_chunks_are_sorted_by_date(self) -> None:
        manifest = SeasonManifest(season=2024)
        chunks = pending_chunks(2024, manifest)
        dates = [c.date for c in chunks]
        assert dates == sorted(dates)

    def test_force_ignores_manifest(self) -> None:
        from fantasy_baseball_manager.statcast.calendar import game_dates

        all_dates = game_dates(2024)
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={d.isoformat() for d in all_dates},
        )
        chunks = pending_chunks(2024, manifest, force=True)
        assert len(chunks) == len(all_dates)
