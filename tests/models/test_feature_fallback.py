from typing import Any

from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.feature_fallback import materialize_with_fallback
from fantasy_baseball_manager.models.statcast_gbm.features import build_batter_preseason_weighted_set


class FakeAssembler:
    """Returns preconfigured rows keyed by feature-set seasons."""

    def __init__(self, rows_by_seasons: dict[tuple[int, ...], list[dict[str, Any]]]) -> None:
        self._rows_by_seasons = rows_by_seasons
        self._counter = 0

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        self._counter += 1
        rows = self._rows_by_seasons.get(feature_set.seasons, [])
        return DatasetHandle(
            dataset_id=self._counter,
            feature_set_id=self._counter,
            table_name="fake",
            row_count=len(rows),
            seasons=feature_set.seasons,
        )

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.materialize(feature_set)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        return list(self._rows_by_seasons.get(handle.seasons, []))

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        raise NotImplementedError


class FakePlayerUniverse:
    def __init__(self, ids_by_season: dict[int, set[int]]) -> None:
        self._ids = ids_by_season

    def get_player_ids(
        self,
        season: int,
        player_type: str,
        *,
        source: str | None = None,
        min_pa: int | None = None,
        min_ip: float | None = None,
    ) -> set[int]:
        return self._ids.get(season, set())


def _make_row(player_id: int, season: int) -> dict[str, Any]:
    return {"player_id": player_id, "season": season, "value": 1.0}


class TestMaterializeWithFallback:
    def test_no_fallback_when_all_seasons_present(self) -> None:
        rows_2023 = [_make_row(1, 2023), _make_row(2, 2023)]
        assembler = FakeAssembler({(2023,): rows_2023})
        fs = build_batter_preseason_weighted_set([2023])
        result = materialize_with_fallback(assembler, fs, [2023], "batter")
        assert len(result) == 2

    def test_no_fallback_when_player_universe_is_none(self) -> None:
        assembler = FakeAssembler({(2026,): []})
        fs = build_batter_preseason_weighted_set([2026])
        result = materialize_with_fallback(assembler, fs, [2026], "batter", player_universe=None)
        assert result == []

    def test_fallback_triggers_for_missing_season(self) -> None:
        """When spine returns no rows for 2026, fallback uses player_universe IDs."""
        rows_2025 = [_make_row(1, 2025)]
        fallback_rows = [_make_row(10, 2026), _make_row(11, 2026)]
        assembler = FakeAssembler(
            {
                (2025, 2026): rows_2025,
                (2026,): fallback_rows,
            }
        )
        universe = FakePlayerUniverse({2026: {10, 11}})
        fs = build_batter_preseason_weighted_set([2025, 2026])
        result = materialize_with_fallback(assembler, fs, [2025, 2026], "batter", universe)
        assert len(result) == 3
        seasons = {r["season"] for r in result}
        assert 2025 in seasons
        assert 2026 in seasons

    def test_fallback_noop_when_universe_returns_no_ids(self) -> None:
        rows_2025 = [_make_row(1, 2025)]
        assembler = FakeAssembler({(2025, 2026): rows_2025})
        universe = FakePlayerUniverse({2026: set()})
        fs = build_batter_preseason_weighted_set([2025, 2026])
        result = materialize_with_fallback(assembler, fs, [2025, 2026], "batter", universe)
        assert len(result) == 1
