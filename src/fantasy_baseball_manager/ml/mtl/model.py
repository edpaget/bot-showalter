"""Multi-Task Learning neural network models for stat prediction.

This module provides PyTorch-based neural networks that predict multiple
correlated baseball stats simultaneously using shared hidden layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig


class _MTLArchitectureConfigDict(TypedDict):
    """Serialized MTLArchitectureConfig structure."""

    shared_layers: tuple[int, ...]
    head_hidden_size: int
    dropout_rates: tuple[float, ...]
    use_batch_norm: bool


class _MultiTaskModelParams(TypedDict):
    """Serialized MultiTaskBatterModel/MultiTaskPitcherModel structure."""

    state_dict: dict[str, torch.Tensor] | None
    n_features: int
    config: _MTLArchitectureConfigDict | None
    feature_names: list[str]
    training_years: list[int]
    validation_metrics: dict[str, float]
    is_fitted: bool


logger = logging.getLogger(__name__)

# Target stats for each player type
BATTER_STATS: tuple[str, ...] = ("hr", "so", "bb", "singles", "doubles", "triples", "sb")
PITCHER_STATS: tuple[str, ...] = ("h", "er", "so", "bb", "hr")


class _MTLModule(nn.Module):
    """PyTorch module combining shared trunk and stat-specific heads."""

    def __init__(
        self,
        trunk: nn.Sequential,
        heads: nn.ModuleDict,
        log_vars: nn.ParameterDict,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.log_vars = log_vars

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.trunk(x)
        return {stat: head(shared) for stat, head in self.heads.items()}


class MultiTaskNet:
    """Multi-task neural network predicting raw stat rates.

    Architecture:
        Input Features (n_features)
            │
        [Shared Trunk]
            Linear(n_features, 64) → BatchNorm → ReLU → Dropout(0.3)
            Linear(64, 32) → BatchNorm → ReLU → Dropout(0.2)
            │
        [Stat-Specific Heads] (one per stat)
            Linear(32, 16) → ReLU → Linear(16, 1)

    The shared trunk learns general player representations while
    the stat-specific heads specialize for each target stat.
    """

    def __init__(
        self,
        n_features: int,
        target_stats: tuple[str, ...],
        config: MTLArchitectureConfig | None = None,
    ) -> None:
        """Initialize the multi-task network.

        Args:
            n_features: Number of input features.
            target_stats: Tuple of stat names to predict.
            config: Architecture configuration. Uses defaults if not provided.
        """
        from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig

        self.n_features = n_features
        self.target_stats = target_stats
        self.config = config or MTLArchitectureConfig()

        # Build shared trunk
        trunk_layers: list[nn.Module] = []
        in_size = n_features

        for i, out_size in enumerate(self.config.shared_layers):
            trunk_layers.append(nn.Linear(in_size, out_size))
            if self.config.use_batch_norm:
                trunk_layers.append(nn.BatchNorm1d(out_size))
            trunk_layers.append(nn.ReLU())
            if i < len(self.config.dropout_rates):
                trunk_layers.append(nn.Dropout(self.config.dropout_rates[i]))
            in_size = out_size

        trunk = nn.Sequential(*trunk_layers)

        # Build stat-specific output heads
        trunk_out_size = self.config.shared_layers[-1] if self.config.shared_layers else n_features
        head_hidden = self.config.head_hidden_size

        heads = nn.ModuleDict()
        for stat in target_stats:
            heads[stat] = nn.Sequential(
                nn.Linear(trunk_out_size, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

        # Learnable log-variance for uncertainty-weighted loss
        # Initialize to 0 (variance=1)
        log_vars = nn.ParameterDict({stat: nn.Parameter(torch.zeros(1)) for stat in target_stats})

        # Combine into a single module for training
        self._module = _MTLModule(trunk, heads, log_vars)
        self._log_vars = log_vars

    @property
    def module(self) -> nn.Module:
        """Return the underlying PyTorch module for training."""
        return self._module

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning predictions for each stat.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Dict mapping stat name to predicted rate tensor of shape (batch_size, 1).
        """
        return self._module(x)

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Predict stat rates for a single feature vector.

        Args:
            features: Feature array of shape (n_features,) or (1, n_features).

        Returns:
            Dict mapping stat name to predicted rate.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        self._module.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            outputs = self._module(x)

        return {stat: float(pred.item()) for stat, pred in outputs.items()}

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute uncertainty-weighted multi-task loss.

        Uses homoscedastic uncertainty to automatically learn per-stat weights.
        Loss for stat i: (1/2σ²_i) * MSE_i + log(σ_i)

        Args:
            predictions: Dict mapping stat to predicted values.
            targets: Dict mapping stat to target values.

        Returns:
            Total loss tensor.
        """
        total_loss = torch.tensor(0.0, device=next(self._module.parameters()).device)

        for stat in self.target_stats:
            if stat not in predictions or stat not in targets:
                continue

            pred = predictions[stat]
            target = targets[stat]
            log_var = self._log_vars[stat]

            # Uncertainty-weighted loss: (1/2σ²) * MSE + log(σ)
            # = (1/2) * exp(-log_var) * MSE + 0.5 * log_var
            precision = torch.exp(-log_var)
            mse = F.mse_loss(pred, target)
            stat_loss = 0.5 * precision * mse + 0.5 * log_var

            total_loss = total_loss + stat_loss.squeeze()

        return total_loss

    def get_uncertainty_weights(self) -> dict[str, float]:
        """Return learned uncertainty weights (1/σ²) for each stat."""
        weights = {}
        with torch.no_grad():
            for stat in self.target_stats:
                log_var = self._log_vars[stat]
                weights[stat] = float(torch.exp(-log_var).item())
        return weights


@dataclass
class MultiTaskBatterModel:
    """Wrapper for batter MTL model with metadata.

    Predicts per-PA rates for: HR, SO, BB, singles, doubles, triples, SB
    """

    STATS: tuple[str, ...] = BATTER_STATS

    network: MultiTaskNet | None = None
    feature_names: list[str] = field(default_factory=list)
    training_years: tuple[int, ...] = field(default_factory=tuple)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted and self.network is not None

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Predict stat rates for a feature vector.

        Args:
            features: Feature array of shape (n_features,) or (1, n_features).

        Returns:
            Dict mapping stat name to predicted per-PA rate.

        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_fitted or self.network is None:
            raise ValueError("MultiTaskBatterModel has not been fitted")
        return self.network.predict(features)

    def get_params(self) -> _MultiTaskModelParams:
        """Return model state for serialization."""
        return _MultiTaskModelParams(
            state_dict=self.network.module.state_dict() if self.network else None,
            n_features=self.network.n_features if self.network else 0,
            config=(
                _MTLArchitectureConfigDict(
                    shared_layers=self.network.config.shared_layers,
                    head_hidden_size=self.network.config.head_hidden_size,
                    dropout_rates=self.network.config.dropout_rates,
                    use_batch_norm=self.network.config.use_batch_norm,
                )
                if self.network
                else None
            ),
            feature_names=self.feature_names,
            training_years=list(self.training_years),
            validation_metrics=self.validation_metrics,
            is_fitted=self._is_fitted,
        )

    @classmethod
    def from_params(cls, params: _MultiTaskModelParams) -> MultiTaskBatterModel:
        """Reconstruct model from serialized state."""
        from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig

        model = cls(
            feature_names=params["feature_names"],
            training_years=tuple(params["training_years"]),
            validation_metrics=params.get("validation_metrics", {}),
        )

        state_dict = params["state_dict"]
        config_dict = params["config"]
        if state_dict is not None and config_dict is not None:
            config = MTLArchitectureConfig(**config_dict)
            model.network = MultiTaskNet(
                n_features=params["n_features"],
                target_stats=cls.STATS,
                config=config,
            )
            model.network.module.load_state_dict(state_dict)
            model._is_fitted = params.get("is_fitted", False)

        return model


@dataclass
class MultiTaskPitcherModel:
    """Wrapper for pitcher MTL model with metadata.

    Predicts per-out rates for: H, ER, SO, BB, HR
    """

    STATS: tuple[str, ...] = PITCHER_STATS

    network: MultiTaskNet | None = None
    feature_names: list[str] = field(default_factory=list)
    training_years: tuple[int, ...] = field(default_factory=tuple)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted and self.network is not None

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Predict stat rates for a feature vector.

        Args:
            features: Feature array of shape (n_features,) or (1, n_features).

        Returns:
            Dict mapping stat name to predicted per-out rate.

        Raises:
            ValueError: If model has not been trained.
        """
        if not self.is_fitted or self.network is None:
            raise ValueError("MultiTaskPitcherModel has not been fitted")
        return self.network.predict(features)

    def get_params(self) -> _MultiTaskModelParams:
        """Return model state for serialization."""
        return _MultiTaskModelParams(
            state_dict=self.network.module.state_dict() if self.network else None,
            n_features=self.network.n_features if self.network else 0,
            config=(
                _MTLArchitectureConfigDict(
                    shared_layers=self.network.config.shared_layers,
                    head_hidden_size=self.network.config.head_hidden_size,
                    dropout_rates=self.network.config.dropout_rates,
                    use_batch_norm=self.network.config.use_batch_norm,
                )
                if self.network
                else None
            ),
            feature_names=self.feature_names,
            training_years=list(self.training_years),
            validation_metrics=self.validation_metrics,
            is_fitted=self._is_fitted,
        )

    @classmethod
    def from_params(cls, params: _MultiTaskModelParams) -> MultiTaskPitcherModel:
        """Reconstruct model from serialized state."""
        from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig

        model = cls(
            feature_names=params["feature_names"],
            training_years=tuple(params["training_years"]),
            validation_metrics=params.get("validation_metrics", {}),
        )

        state_dict = params["state_dict"]
        config_dict = params["config"]
        if state_dict is not None and config_dict is not None:
            config = MTLArchitectureConfig(**config_dict)
            model.network = MultiTaskNet(
                n_features=params["n_features"],
                target_stats=cls.STATS,
                config=config,
            )
            model.network.module.load_state_dict(state_dict)
            model._is_fitted = params.get("is_fitted", False)

        return model
