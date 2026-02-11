"""Tests for hierarchical fine-tuning dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.identity.archetypes import ArchetypeModel
from fantasy_baseball_manager.contextual.identity.stat_profile import PlayerStatProfile
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    HierarchicalFineTuneConfig,
)
from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
    HierarchicalFineTuneBatch,
    HierarchicalFineTuneDataset,
    build_hierarchical_windows,
    collate_hierarchical_samples,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig

from tests.contextual.model.conftest import make_player_context


def _build_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _make_profile(player_id: int) -> PlayerStatProfile:
    return PlayerStatProfile(
        player_id=str(player_id),
        name=f"Player {player_id}",
        year=2024,
        player_type="batter",
        age=28,
        handedness=None,
        rates_career={"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01},
        rates_3yr=None,
        rates_1yr=None,
        rates_30d=None,
        opportunities_career=1000.0,
        opportunities_3yr=None,
        opportunities_1yr=None,
    )


def _make_fitted_archetype_model(profiles: list[PlayerStatProfile]) -> ArchetypeModel:
    X = np.array([p.to_feature_vector() for p in profiles])
    model = ArchetypeModel(n_archetypes=min(3, len(profiles)))
    model.fit(X)
    return model


class TestHierarchicalFineTuneConfig:
    def test_creation_with_defaults(self) -> None:
        config = HierarchicalFineTuneConfig()
        assert config.identity_learning_rate == 1e-3
        assert config.level3_learning_rate == 5e-4
        assert config.head_learning_rate == 1e-3
        assert config.stat_feature_dropout == 0.0

    def test_validation_target_mode(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="target_mode"):
            HierarchicalFineTuneConfig(target_mode="invalid")

    def test_validation_min_games_rates(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="min_games"):
            HierarchicalFineTuneConfig(
                context_window=10, target_window=5, min_games=10,
            )


class TestBuildHierarchicalWindows:
    def test_samples_have_identity_features(self, small_config: ModelConfig) -> None:
        tensorizer = _build_tensorizer(small_config)
        config = HierarchicalFineTuneConfig(
            context_window=2, target_mode="counts", min_games=3,
        )
        contexts = [make_player_context(n_games=6, pitches_per_game=5, player_id=660271)]
        profiles = [_make_profile(660271)]
        archetype_model = _make_fitted_archetype_model(profiles)
        profile_lookup = {660271: profiles[0]}

        windows = build_hierarchical_windows(
            contexts, tensorizer, config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, stat_input_dim=19,
        )
        assert len(windows) > 0
        _, _, _, identity_features, archetype_id = windows[0]
        assert identity_features.shape == (19,)
        assert isinstance(archetype_id, int)
        assert archetype_id >= 0

    def test_missing_profile_fallback(self, small_config: ModelConfig) -> None:
        tensorizer = _build_tensorizer(small_config)
        config = HierarchicalFineTuneConfig(
            context_window=2, target_mode="counts", min_games=3,
        )
        # Player 999 has no profile
        contexts = [make_player_context(n_games=6, pitches_per_game=5, player_id=999)]
        # Need at least some profiles to create a fitted archetype model
        dummy_profiles = [_make_profile(i) for i in range(5)]
        archetype_model = _make_fitted_archetype_model(dummy_profiles)
        profile_lookup: dict[int, PlayerStatProfile] = {}

        windows = build_hierarchical_windows(
            contexts, tensorizer, config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, stat_input_dim=19,
        )
        assert len(windows) > 0
        _, _, _, identity_features, archetype_id = windows[0]
        assert torch.all(identity_features == 0.0)
        assert archetype_id == 0


class TestCollateHierarchicalSamples:
    def test_produces_correct_batch_shapes(self, small_config: ModelConfig) -> None:
        tensorizer = _build_tensorizer(small_config)
        config = HierarchicalFineTuneConfig(
            context_window=2, target_mode="counts", min_games=3,
        )
        contexts = [
            make_player_context(n_games=6, pitches_per_game=5, player_id=660271 + i)
            for i in range(3)
        ]
        profiles = [_make_profile(660271 + i) for i in range(3)]
        archetype_model = _make_fitted_archetype_model(profiles)
        profile_lookup = {660271 + i: profiles[i] for i in range(3)}

        windows = build_hierarchical_windows(
            contexts, tensorizer, config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, stat_input_dim=19,
        )
        dataset = HierarchicalFineTuneDataset(windows)

        # Collate first 2 samples
        samples = [dataset[0], dataset[1]]
        batch = collate_hierarchical_samples(samples)

        assert isinstance(batch, HierarchicalFineTuneBatch)
        assert batch.targets.shape[0] == 2
        assert batch.targets.shape[1] == len(BATTER_TARGET_STATS)
        assert batch.identity_features.shape == (2, 19)
        assert batch.archetype_ids.shape == (2,)
        assert batch.archetype_ids.dtype == torch.long
