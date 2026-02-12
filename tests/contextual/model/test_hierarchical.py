"""Tests for hierarchical model with identity conditioning (Phase 2a)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
from fantasy_baseball_manager.contextual.model.hierarchical import (
    GamePooler,
    HierarchicalModel,
    HierarchicalPredictionHead,
    IdentityModule,
    Level3Attention,
)
from fantasy_baseball_manager.contextual.model.hierarchical_config import (
    HierarchicalModelConfig,
)
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.persistence import ContextualModelMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.contextual.identity.stat_profile import (
        PlayerStatProfile,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig

from .conftest import make_player_context


def _make_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _small_hier_config(d_model: int = 32) -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
        identity_stat_dim=16,
        identity_archetype_dim=8,
        identity_repr_dim=24,
        n_archetypes=5,
        batter_stat_input_dim=19,
        pitcher_stat_input_dim=13,
        level3_d_model=d_model,
        level3_n_heads=2,
        level3_n_layers=1,
        level3_ff_dim=64,
        level3_dropout=0.0,
        level3_max_games=10,
    )


# ---------------------------------------------------------------------------
# Step 1: HierarchicalModelConfig
# ---------------------------------------------------------------------------


class TestHierarchicalModelConfig:
    def test_creation_with_defaults(self) -> None:
        config = HierarchicalModelConfig()
        assert config.identity_stat_dim == 64
        assert config.identity_archetype_dim == 32
        assert config.identity_repr_dim == 96
        assert config.n_archetypes == 20
        assert config.batter_stat_input_dim == 19
        assert config.pitcher_stat_input_dim == 13
        assert config.level3_d_model == 256
        assert config.level3_n_heads == 4
        assert config.level3_n_layers == 2

    def test_frozen_immutability(self) -> None:
        config = HierarchicalModelConfig()
        with pytest.raises(AttributeError):
            config.identity_stat_dim = 128  # type: ignore[misc]

    def test_identity_repr_dim_matches_components(self) -> None:
        config = HierarchicalModelConfig()
        assert config.identity_repr_dim == config.identity_stat_dim + config.identity_archetype_dim


# ---------------------------------------------------------------------------
# Step 2: IdentityModule
# ---------------------------------------------------------------------------


class TestIdentityModule:
    def test_output_shape_batter(self) -> None:
        hc = _small_hier_config()
        module = IdentityModule(hc, stat_input_dim=hc.batter_stat_input_dim)
        stat_features = torch.randn(4, hc.batter_stat_input_dim)
        archetype_ids = torch.tensor([0, 1, 2, 3])
        out = module(stat_features, archetype_ids)
        assert out.shape == (4, hc.identity_repr_dim)

    def test_output_shape_pitcher(self) -> None:
        hc = _small_hier_config()
        module = IdentityModule(hc, stat_input_dim=hc.pitcher_stat_input_dim)
        stat_features = torch.randn(2, hc.pitcher_stat_input_dim)
        archetype_ids = torch.tensor([0, 1])
        out = module(stat_features, archetype_ids)
        assert out.shape == (2, hc.identity_repr_dim)

    def test_gradient_flow(self) -> None:
        hc = _small_hier_config()
        module = IdentityModule(hc, stat_input_dim=hc.batter_stat_input_dim)
        stat_features = torch.randn(2, hc.batter_stat_input_dim, requires_grad=True)
        archetype_ids = torch.tensor([0, 1])
        out = module(stat_features, archetype_ids)
        # Use pow(2).sum() since sum() alone is zero through LayerNorm (mean=0)
        out.pow(2).sum().backward()
        assert stat_features.grad is not None
        assert stat_features.grad.abs().sum() > 0

    def test_different_archetypes_produce_different_outputs(self) -> None:
        hc = _small_hier_config()
        module = IdentityModule(hc, stat_input_dim=hc.batter_stat_input_dim)
        stat_features = torch.randn(1, hc.batter_stat_input_dim)
        # Same stats, different archetype IDs
        out_a = module(stat_features, torch.tensor([0]))
        out_b = module(stat_features, torch.tensor([1]))
        assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Step 3: GamePooler
# ---------------------------------------------------------------------------


class TestGamePooler:
    def test_correct_mean_pooling(self) -> None:
        """Two games with known values → verify mean is correct."""
        pooler = GamePooler()
        batch = 1
        seq_len = 8  # CLS + PLAYER + 3 pitches game0 + 3 pitches game1
        d_model = 16

        hidden = torch.zeros(batch, seq_len, d_model)
        # CLS at pos 0, PLAYER at pos 1
        # game0 pitches at pos 2,3,4 with value 1.0
        hidden[0, 2:5, :] = 1.0
        # game1 pitches at pos 5,6,7 with value 3.0
        hidden[0, 5:8, :] = 3.0

        game_ids = torch.tensor([[-1, -1, 0, 0, 0, 1, 1, 1]])
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        player_token_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        player_token_mask[0, 1] = True

        game_embs, game_mask = pooler(hidden, game_ids, padding_mask, player_token_mask)

        # Should have 2 games
        assert game_mask[0, 0].item() is True
        assert game_mask[0, 1].item() is True

        # Mean of game0 = 1.0, game1 = 3.0
        assert torch.allclose(game_embs[0, 0], torch.ones(d_model), atol=1e-5)
        assert torch.allclose(game_embs[0, 1], torch.full((d_model,), 3.0), atol=1e-5)

    def test_cls_player_padding_excluded(self) -> None:
        """CLS, PLAYER, and padding tokens must not contribute to pools."""
        pooler = GamePooler()
        batch = 1
        seq_len = 6
        d_model = 8

        hidden = torch.ones(batch, seq_len, d_model) * 99.0
        # Only game0 pitches at pos 2,3 with value 2.0
        hidden[0, 2:4, :] = 2.0

        game_ids = torch.tensor([[-1, -1, 0, 0, -2, -2]])
        padding_mask = torch.tensor([[True, True, True, True, False, False]])
        player_token_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        player_token_mask[0, 1] = True

        game_embs, game_mask = pooler(hidden, game_ids, padding_mask, player_token_mask)

        assert game_mask[0, 0].item() is True
        assert torch.allclose(game_embs[0, 0], torch.full((d_model,), 2.0), atol=1e-5)

    def test_output_shape_with_variable_games(self) -> None:
        """Batch of 2 samples with different game counts."""
        pooler = GamePooler()
        d_model = 8
        # Sample 0: 2 games, sample 1: 1 game
        seq_len = 10
        hidden = torch.randn(2, seq_len, d_model)
        game_ids = torch.tensor([
            [-1, -1, 0, 0, 0, 1, 1, 1, -2, -2],
            [-1, -1, 0, 0, 0, -2, -2, -2, -2, -2],
        ])
        padding_mask = torch.tensor([
            [True, True, True, True, True, True, True, True, False, False],
            [True, True, True, True, True, False, False, False, False, False],
        ])
        player_token_mask = torch.zeros(2, seq_len, dtype=torch.bool)
        player_token_mask[:, 1] = True

        game_embs, game_mask = pooler(hidden, game_ids, padding_mask, player_token_mask)

        # max games = 2
        assert game_embs.shape == (2, 2, d_model)
        assert game_mask.shape == (2, 2)
        # Sample 0 has 2 valid games, sample 1 has 1
        assert game_mask[0].tolist() == [True, True]
        assert game_mask[1].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Step 4: Level3Attention
# ---------------------------------------------------------------------------


class TestLevel3Attention:
    def test_output_shape(self) -> None:
        hc = _small_hier_config()
        level3 = Level3Attention(hc)
        game_embs = torch.randn(2, 5, hc.level3_d_model)
        game_mask = torch.ones(2, 5, dtype=torch.bool)
        identity_repr = torch.randn(2, hc.identity_repr_dim)
        out = level3(game_embs, game_mask, identity_repr)
        assert out.shape == (2, hc.level3_d_model)

    def test_identity_conditioning_changes_output(self) -> None:
        hc = _small_hier_config()
        level3 = Level3Attention(hc)
        game_embs = torch.randn(1, 3, hc.level3_d_model)
        game_mask = torch.ones(1, 3, dtype=torch.bool)

        id_a = torch.randn(1, hc.identity_repr_dim)
        id_b = torch.randn(1, hc.identity_repr_dim)
        out_a = level3(game_embs, game_mask, id_a)
        out_b = level3(game_embs, game_mask, id_b)
        assert not torch.allclose(out_a, out_b)

    def test_single_game_works(self) -> None:
        hc = _small_hier_config()
        level3 = Level3Attention(hc)
        game_embs = torch.randn(1, 1, hc.level3_d_model)
        game_mask = torch.ones(1, 1, dtype=torch.bool)
        identity_repr = torch.randn(1, hc.identity_repr_dim)
        out = level3(game_embs, game_mask, identity_repr)
        assert out.shape == (1, hc.level3_d_model)

    def test_padding_games_dont_affect_output(self) -> None:
        hc = _small_hier_config()
        level3 = Level3Attention(hc)
        torch.manual_seed(42)
        identity_repr = torch.randn(1, hc.identity_repr_dim)
        game_embs_2 = torch.randn(1, 2, hc.level3_d_model)
        mask_2 = torch.tensor([[True, True]])

        # Add a padding game that should be masked out
        game_embs_3 = torch.zeros(1, 3, hc.level3_d_model)
        game_embs_3[:, :2, :] = game_embs_2
        game_embs_3[:, 2, :] = 999.0  # garbage that should be masked
        mask_3 = torch.tensor([[True, True, False]])

        out_2 = level3(game_embs_2, mask_2, identity_repr)
        out_3 = level3(game_embs_3, mask_3, identity_repr)
        assert torch.allclose(out_2, out_3, atol=1e-5)

    def test_gradient_flow(self) -> None:
        hc = _small_hier_config()
        level3 = Level3Attention(hc)
        game_embs = torch.randn(1, 3, hc.level3_d_model, requires_grad=True)
        game_mask = torch.ones(1, 3, dtype=torch.bool)
        identity_repr = torch.randn(1, hc.identity_repr_dim, requires_grad=True)
        out = level3(game_embs, game_mask, identity_repr)
        out.sum().backward()
        assert game_embs.grad is not None
        assert identity_repr.grad is not None


# ---------------------------------------------------------------------------
# Step 5: HierarchicalPredictionHead + HierarchicalModel
# ---------------------------------------------------------------------------


class TestHierarchicalPredictionHead:
    def test_output_shape(self) -> None:
        hc = _small_hier_config()
        head = HierarchicalPredictionHead(
            d_model=hc.level3_d_model,
            identity_repr_dim=hc.identity_repr_dim,
            n_targets=6,
            dropout=0.0,
        )
        x = torch.randn(4, hc.level3_d_model + hc.identity_repr_dim)
        out = head(x)
        assert out.shape == (4, 6)


class TestHierarchicalModel:
    def test_forward_produces_predictions(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        batch, stat_features, archetype_ids = self._make_inputs(small_config, hc, n=2)
        output = model(batch, stat_features, archetype_ids)
        assert output["performance_preds"].shape == (2, small_config.n_batter_targets)

    def test_frozen_backbone_unchanged_after_backward(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        # Snapshot backbone params
        backbone_params_before = {
            name: p.clone() for name, p in model.backbone.named_parameters()
        }

        batch, stat_features, archetype_ids = self._make_inputs(small_config, hc, n=2)
        output = model(batch, stat_features, archetype_ids)
        loss = output["performance_preds"].pow(2).sum()
        loss.backward()

        for name, p in model.backbone.named_parameters():
            assert torch.equal(p, backbone_params_before[name]), f"Backbone param {name} changed"
            assert p.grad is None or p.grad.abs().sum() == 0, f"Backbone param {name} has gradients"

    def test_learnable_params_receive_gradients(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        batch, stat_features, archetype_ids = self._make_inputs(small_config, hc, n=2)
        output = model(batch, stat_features, archetype_ids)
        loss = output["performance_preds"].pow(2).sum()
        loss.backward()

        # Check identity module has gradients
        for name, p in model.identity_module.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"identity_module.{name} has no grad"
                assert p.grad.abs().sum() > 0, f"identity_module.{name} has zero grad"

        # Check level3 has gradients
        for name, p in model.level3.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"level3.{name} has no grad"

        # Check head has gradients
        for name, p in model.head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"head.{name} has no grad"

    def test_no_nan_after_backward(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        batch, stat_features, archetype_ids = self._make_inputs(small_config, hc, n=2)
        output = model(batch, stat_features, archetype_ids)
        preds = output["performance_preds"]
        assert not torch.isnan(preds).any()

        loss = preds.pow(2).sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in grad for {name}"

    def test_pitcher_config(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_pitcher_targets,
            stat_input_dim=hc.pitcher_stat_input_dim,
        )
        batch, _, _ = self._make_inputs(small_config, hc, n=2)
        stat_features = torch.randn(2, hc.pitcher_stat_input_dim)
        archetype_ids = torch.tensor([0, 1])
        output = model(batch, stat_features, archetype_ids)
        assert output["performance_preds"].shape == (2, small_config.n_pitcher_targets)

    def test_forward_precomputed_shape(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        game_embs = torch.randn(2, 5, hc.level3_d_model)
        game_mask = torch.ones(2, 5, dtype=torch.bool)
        stat_features = torch.randn(2, hc.batter_stat_input_dim)
        archetype_ids = torch.randint(0, hc.n_archetypes, (2,))

        output = model.forward_precomputed(game_embs, game_mask, stat_features, archetype_ids)
        assert output["performance_preds"].shape == (2, small_config.n_batter_targets)

    def test_forward_precomputed_gradients_flow(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        game_embs = torch.randn(2, 5, hc.level3_d_model)
        game_mask = torch.ones(2, 5, dtype=torch.bool)
        stat_features = torch.randn(2, hc.batter_stat_input_dim)
        archetype_ids = torch.randint(0, hc.n_archetypes, (2,))

        output = model.forward_precomputed(game_embs, game_mask, stat_features, archetype_ids)
        loss = output["performance_preds"].pow(2).sum()
        loss.backward()

        # Identity/level3/head should get gradients
        for name, p in model.identity_module.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"identity_module.{name} has no grad"
        for name, p in model.level3.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"level3.{name} has no grad"
        for name, p in model.head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"head.{name} has no grad"

        # Backbone should NOT get gradients
        for name, p in model.backbone.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"Backbone param {name} has gradients in precomputed path"
            )

    def test_forward_precomputed_identity_differentiation(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        model.eval()
        game_embs = torch.randn(1, 3, hc.level3_d_model)
        game_mask = torch.ones(1, 3, dtype=torch.bool)

        stat_a = torch.zeros(1, hc.batter_stat_input_dim)
        stat_b = torch.ones(1, hc.batter_stat_input_dim) * 5.0
        arch_a = torch.tensor([0])
        arch_b = torch.tensor([2])

        with torch.no_grad():
            out_a = model.forward_precomputed(game_embs, game_mask, stat_a, arch_a)["performance_preds"]
            out_b = model.forward_precomputed(game_embs, game_mask, stat_b, arch_b)["performance_preds"]

        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_forward_precomputed_no_nan(self, small_config: ModelConfig) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        game_embs = torch.randn(2, 5, hc.level3_d_model)
        game_mask = torch.ones(2, 5, dtype=torch.bool)
        stat_features = torch.randn(2, hc.batter_stat_input_dim)
        archetype_ids = torch.randint(0, hc.n_archetypes, (2,))

        output = model.forward_precomputed(game_embs, game_mask, stat_features, archetype_ids)
        preds = output["performance_preds"]
        assert not torch.isnan(preds).any()

        loss = preds.pow(2).sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in grad for {name}"

    # -- helpers --

    @staticmethod
    def _make_backbone(config: ModelConfig) -> ContextualPerformanceModel:
        head = PerformancePredictionHead(config, config.n_batter_targets)
        return ContextualPerformanceModel(config, head)

    @staticmethod
    def _make_inputs(
        config: ModelConfig,
        hc: HierarchicalModelConfig,
        n: int,
    ) -> tuple:
        tensorizer = _make_tensorizer(config)
        contexts = [make_player_context(n_games=3, pitches_per_game=4, player_id=660271 + i) for i in range(n)]
        singles = [tensorizer.tensorize_context(ctx) for ctx in contexts]
        batch = tensorizer.collate(singles)
        stat_features = torch.randn(n, hc.batter_stat_input_dim)
        archetype_ids = torch.randint(0, hc.n_archetypes, (n,))
        return batch, stat_features, archetype_ids


# ---------------------------------------------------------------------------
# Step 11: Integration test
# ---------------------------------------------------------------------------


def _make_profile(
    player_id: int,
    player_type: str = "batter",
    rates: dict[str, float] | None = None,
    age: int = 28,
) -> PlayerStatProfile:
    from fantasy_baseball_manager.contextual.identity.stat_profile import PlayerStatProfile

    if player_type == "batter":
        default_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
    else:
        default_rates = {"so": 0.25, "h": 0.20, "bb": 0.08, "hr": 0.03}
    actual_rates = rates or default_rates
    return PlayerStatProfile(
        player_id=str(player_id),
        name=f"Player {player_id}",
        year=2024,
        player_type=player_type,
        age=age,
        handedness=None,
        rates_career=actual_rates,
        rates_3yr=None,
        rates_1yr=None,
        rates_30d=None,
        opportunities_career=1000.0,
        opportunities_3yr=None,
        opportunities_1yr=None,
    )


class TestIntegration:
    """End-to-end: profiles → archetypes → hierarchical model → dataset → train → serialize → re-predict."""

    def test_end_to_end_hierarchical(self, small_config: ModelConfig, tmp_path: Path) -> None:
        from fantasy_baseball_manager.contextual.identity.archetypes import (
            fit_archetypes,
        )
        from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
        from fantasy_baseball_manager.contextual.training.config import (
            BATTER_TARGET_STATS,
            HierarchicalFineTuneConfig,
        )
        from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
            HierarchicalFineTuneDataset,
            build_hierarchical_windows,
        )
        from fantasy_baseball_manager.contextual.training.hierarchical_finetune import (
            HierarchicalFineTuneTrainer,
        )

        hc = _small_hier_config(d_model=small_config.d_model)

        # 1. Build divergent profiles: high-K vs low-K
        high_k_profile = _make_profile(
            1, rates={"hr": 0.02, "so": 0.35, "bb": 0.05, "h": 0.18, "2b": 0.03, "3b": 0.005},
        )
        low_k_profile = _make_profile(
            2, rates={"hr": 0.06, "so": 0.08, "bb": 0.15, "h": 0.30, "2b": 0.07, "3b": 0.02},
        )
        profiles = [high_k_profile, low_k_profile]
        # Need more profiles for clustering
        rng = np.random.default_rng(42)
        for i in range(3, 15):
            rates = {
                "hr": max(0.0, 0.04 + rng.normal(0, 0.02)),
                "so": max(0.0, 0.20 + rng.normal(0, 0.05)),
                "bb": max(0.0, 0.10 + rng.normal(0, 0.03)),
                "h": max(0.0, 0.25 + rng.normal(0, 0.03)),
                "2b": max(0.0, 0.05 + rng.normal(0, 0.02)),
                "3b": max(0.0, 0.01 + rng.normal(0, 0.005)),
            }
            profiles.append(_make_profile(i, rates=rates))

        # 2. Fit archetypes
        arch_model, _labels = fit_archetypes(profiles, n_archetypes=min(hc.n_archetypes, len(profiles)))
        profile_lookup = {int(p.player_id): p for p in profiles}

        # 3. Create hierarchical model
        backbone = self._make_backbone(small_config)
        model = HierarchicalModel(
            backbone=backbone,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )

        # 4. Build dataset (using counts mode for synthetic data without PA events)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
            identity_learning_rate=0.01, level3_learning_rate=0.01, head_learning_rate=0.01,
            min_warmup_steps=1,
        )
        tensorizer = _make_tensorizer(small_config)
        # Create contexts for a few players
        player_ids = [1, 2, 3, 4, 5]
        contexts = [
            make_player_context(n_games=6, pitches_per_game=5, player_id=pid)
            for pid in player_ids
        ]

        train_windows = build_hierarchical_windows(
            contexts, tensorizer, ft_config, BATTER_TARGET_STATS,
            profile_lookup, arch_model, stat_input_dim=hc.batter_stat_input_dim,
        )
        val_windows = build_hierarchical_windows(
            contexts[:2], tensorizer, ft_config, BATTER_TARGET_STATS,
            profile_lookup, arch_model, stat_input_dim=hc.batter_stat_input_dim,
        )
        train_ds = HierarchicalFineTuneDataset.from_windows(train_windows)
        val_ds = HierarchicalFineTuneDataset.from_windows(val_windows)

        # 5. Train 2 epochs
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        result = trainer.train(train_ds, val_ds)
        assert result["val_loss"] >= 0

        # 6. Verify predictions differ by identity (key diagnostic)
        model.eval()
        ctx = make_player_context(n_games=3, pitches_per_game=5, player_id=1)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        high_k_feat = torch.tensor(high_k_profile.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        low_k_feat = torch.tensor(low_k_profile.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        high_k_arch = torch.tensor([int(arch_model.predict_single(high_k_profile.to_feature_vector()))])
        low_k_arch = torch.tensor([int(arch_model.predict_single(low_k_profile.to_feature_vector()))])

        with torch.no_grad():
            pred_high_k = model(batch, high_k_feat, high_k_arch)["performance_preds"]
            pred_low_k = model(batch, low_k_feat, low_k_arch)["performance_preds"]

        assert not torch.allclose(pred_high_k, pred_low_k, atol=1e-6), (
            "Predictions should differ for divergent identity profiles"
        )

        # 7. Serialize → deserialize → re-predict matches
        store.save_hierarchical_checkpoint(
            "integration_test", model,
            ContextualModelMetadata(
                name="integration_test", epoch=2, train_loss=0.0, val_loss=0.0,
            ),
        )
        loaded_model = store.load_hierarchical_model(
            "integration_test",
            backbone_config=small_config,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        loaded_model.eval()  # type: ignore[union-attr]

        with torch.no_grad():
            pred_loaded = loaded_model(batch, high_k_feat, high_k_arch)["performance_preds"]  # type: ignore[operator]

        assert torch.allclose(pred_high_k, pred_loaded, atol=1e-5), (
            "Loaded model predictions should match original"
        )

    @staticmethod
    def _make_backbone(config: ModelConfig) -> ContextualPerformanceModel:
        head = PerformancePredictionHead(config, config.n_batter_targets)
        return ContextualPerformanceModel(config, head)
