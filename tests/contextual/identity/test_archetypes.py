"""Tests for ArchetypeModel and archetype fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fantasy_baseball_manager.context import init_context, reset_context

if TYPE_CHECKING:
    from pathlib import Path
from fantasy_baseball_manager.contextual.identity.archetypes import (
    ArchetypeModel,
    fit_archetypes,
    load_archetype_model,
    save_archetype_model,
)
from fantasy_baseball_manager.contextual.identity.stat_profile import (
    PlayerStatProfile,
    PlayerStatProfileBuilder,
)
from fantasy_baseball_manager.marcel.models import BattingSeasonStats
from fantasy_baseball_manager.result import Ok

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    player_id: str,
    player_type: str = "batter",
    rates_career: dict[str, float] | None = None,
    age: int = 28,
) -> PlayerStatProfile:
    if player_type == "batter":
        default_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
    else:
        default_rates = {"so": 0.25, "h": 0.20, "bb": 0.08, "hr": 0.03}
    rates = rates_career or default_rates
    return PlayerStatProfile(
        player_id=player_id,
        name=f"Player {player_id}",
        year=2024,
        player_type=player_type,
        age=age,
        handedness=None,
        rates_career=rates,
        rates_3yr=None,
        rates_1yr=None,
        rates_30d=None,
        opportunities_career=1000.0,
        opportunities_3yr=None,
        opportunities_1yr=None,
    )


def _make_batter_profiles(n: int, seed: int = 42) -> list[PlayerStatProfile]:
    """Generate n batter profiles with slightly varied rates."""
    rng = np.random.default_rng(seed)
    profiles = []
    base = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
    for i in range(n):
        rates = {k: max(0.0, v + rng.normal(0, 0.02)) for k, v in base.items()}
        profiles.append(_make_profile(f"b{i}", "batter", rates, age=25 + i % 10))
    return profiles


# ---------------------------------------------------------------------------
# Step 6: ArchetypeModel — fit, predict, centroids
# ---------------------------------------------------------------------------

class TestArchetypeModel:
    def test_creation_is_not_fitted(self) -> None:
        model = ArchetypeModel(n_archetypes=4)
        assert model.is_fitted is False

    def test_fit_sets_fitted(self) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=3)
        model.fit(X)
        assert model.is_fitted is True

    def test_predict_returns_valid_labels(self) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=3)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (20,)
        assert all(0 <= lab < 3 for lab in labels)

    def test_predict_single_returns_int(self) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=3)
        model.fit(X)
        label = model.predict_single(X[0])
        assert isinstance(label, (int, np.integer))
        assert 0 <= label < 3

    def test_predict_before_fit_raises(self) -> None:
        model = ArchetypeModel(n_archetypes=3)
        X = np.random.default_rng(42).random((5, 19))
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_single_before_fit_raises(self) -> None:
        model = ArchetypeModel(n_archetypes=3)
        x = np.random.default_rng(42).random(19)
        with pytest.raises(ValueError, match="fitted"):
            model.predict_single(x)

    def test_centroids_shape(self) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=4)
        model.fit(X)
        centroids = model.centroids()
        assert centroids.shape == (4, 19)

    def test_centroids_before_fit_raises(self) -> None:
        model = ArchetypeModel(n_archetypes=3)
        with pytest.raises(ValueError, match="fitted"):
            model.centroids()


# ---------------------------------------------------------------------------
# Step 7: ArchetypeModel serialization + save/load
# ---------------------------------------------------------------------------

class TestArchetypeModelSerialization:
    def test_get_params_contains_required_keys(self) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=3)
        model.fit(X)
        params = model.get_params()
        assert "n_archetypes" in params
        assert "scaler" in params
        assert "kmeans" in params
        assert "is_fitted" in params

    def test_roundtrip_preserves_predictions(self) -> None:
        profiles = _make_batter_profiles(30)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=4)
        model.fit(X)
        original_labels = model.predict(X)

        params = model.get_params()
        restored = ArchetypeModel.from_params(params)
        restored_labels = restored.predict(X)

        np.testing.assert_array_equal(original_labels, restored_labels)

    def test_save_load_file_roundtrip(self, tmp_path: Path) -> None:
        profiles = _make_batter_profiles(20)
        X = np.array([p.to_feature_vector() for p in profiles])
        model = ArchetypeModel(n_archetypes=3)
        model.fit(X)
        original_labels = model.predict(X)

        save_archetype_model(model, "test_model", directory=tmp_path)
        loaded = load_archetype_model("test_model", directory=tmp_path)
        loaded_labels = loaded.predict(X)

        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_archetype_model("nonexistent", directory=tmp_path)


# ---------------------------------------------------------------------------
# Step 8: fit_archetypes() convenience function
# ---------------------------------------------------------------------------

class TestFitArchetypes:
    def test_returns_fitted_model_and_labels(self) -> None:
        profiles = _make_batter_profiles(20)
        model, labels = fit_archetypes(profiles, n_archetypes=3)
        assert model.is_fitted
        assert len(labels) == 20
        assert all(0 <= lab < 3 for lab in labels)

    def test_labels_consistent_with_re_prediction(self) -> None:
        profiles = _make_batter_profiles(30)
        model, labels = fit_archetypes(profiles, n_archetypes=4)
        X = np.array([p.to_feature_vector() for p in profiles])
        re_predicted = model.predict(X)
        np.testing.assert_array_equal(labels, re_predicted)

    def test_mixed_player_types_raises(self) -> None:
        batter = _make_profile("b1", "batter")
        pitcher = _make_profile("p1", "pitcher")
        with pytest.raises(ValueError, match="player_type"):
            fit_archetypes([batter, pitcher], n_archetypes=2)

    def test_fewer_profiles_than_archetypes_caps(self) -> None:
        profiles = _make_batter_profiles(3)
        model, labels = fit_archetypes(profiles, n_archetypes=10)
        # Should cap to number of profiles
        assert model.n_archetypes <= 3
        assert len(labels) == 3


# ---------------------------------------------------------------------------
# Step 9: Integration test
# ---------------------------------------------------------------------------

def _batting_season(
    player_id: str = "p1",
    name: str = "Test Batter",
    year: int = 2023,
    age: int = 28,
    pa: int = 600,
    h: int = 150,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 60,
    so: int = 120,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=pa - bb - 5,
        h=h,
        singles=h - doubles - triples - hr,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=5,
        sf=5,
        sh=0,
        sb=10,
        cs=3,
        r=80,
        rbi=90,
    )


class _FakeBattingSource:
    def __init__(self, data_by_year: dict[int, list[BattingSeasonStats]]) -> None:
        self._data = data_by_year

    def __call__(self, query: object) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(self._data.get(year, []))


class _EmptyPitchingSource:
    def __call__(self, query: object) -> Ok[list[object]]:
        return Ok([])


class TestIntegration:
    """End-to-end: mock data -> build profiles -> fit archetypes -> serialize -> re-predict."""

    def setup_method(self) -> None:
        init_context(year=2024)

    def teardown_method(self) -> None:
        reset_context()

    def test_end_to_end_flow(self, tmp_path: Path) -> None:
        # Create mock data with enough players for clustering
        rng = np.random.default_rng(42)
        batting_data: dict[int, list[BattingSeasonStats]] = {}
        for yr in [2022, 2023]:
            seasons = []
            for i in range(15):
                hr_count = int(10 + rng.integers(0, 30))
                seasons.append(
                    _batting_season(
                        player_id=f"b{i}",
                        name=f"Batter {i}",
                        year=yr,
                        age=24 + (yr - 2022) + i % 8,
                        pa=500 + int(rng.integers(0, 150)),
                        hr=hr_count,
                        so=int(80 + rng.integers(0, 80)),
                        bb=int(30 + rng.integers(0, 50)),
                        h=int(100 + rng.integers(0, 80)),
                        doubles=int(15 + rng.integers(0, 20)),
                        triples=int(rng.integers(0, 8)),
                    )
                )
            batting_data[yr] = seasons

        # Build profiles
        builder = PlayerStatProfileBuilder()
        profiles = builder.build_all_profiles(
            batting_source=_FakeBattingSource(batting_data),
            pitching_source=_EmptyPitchingSource(),
            year=2024,
            history_years=[2022, 2023],
        )
        assert len(profiles) == 15

        # Fit archetypes
        model, labels = fit_archetypes(profiles, n_archetypes=3)
        assert model.is_fitted
        assert len(labels) == 15

        # Serialize
        save_archetype_model(model, "integration_test", directory=tmp_path)

        # Deserialize
        loaded = load_archetype_model("integration_test", directory=tmp_path)

        # Re-predict matches
        X = np.array([p.to_feature_vector() for p in profiles])
        original_labels = model.predict(X)
        loaded_labels = loaded.predict(X)
        np.testing.assert_array_equal(original_labels, loaded_labels)
