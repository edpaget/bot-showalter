"""Unified model registry providing cross-type operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.registry.base_store import BaseModelStore, ModelMetadata

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore

# Matches versioned names like "default_v2", "experiment_v10"
_VERSION_PATTERN = re.compile(r"^(.+)_v(\d+)$")


@dataclass
class ModelRegistry:
    """Facade over all model stores for cross-type operations.

    Provides unified listing, version management, and comparison
    across GB, MTL, MLE, and contextual model stores.
    """

    gb_store: BaseModelStore
    mtl_store: MTLBaseModelStore
    mle_store: BaseModelStore
    contextual_store: ContextualModelStore

    def list_all(
        self,
        model_type: str | None = None,
        player_type: str | None = None,
    ) -> list[ModelMetadata]:
        """List models across all stores, optionally filtered.

        Args:
            model_type: Filter by "gb_residual", "mtl", "mle", or "contextual".
            player_type: Filter by "batter" or "pitcher".

        Returns:
            Sorted list of model metadata from all matching stores.
        """
        all_models: list[ModelMetadata] = []

        stores: list[tuple[str, BaseModelStore]] = [
            ("gb_residual", self.gb_store),
            ("mtl", self.mtl_store),
            ("mle", self.mle_store),
        ]

        for type_name, store in stores:
            if model_type is not None and model_type != type_name:
                continue
            all_models.extend(store.list_models())

        if model_type is None or model_type == "contextual":
            all_models.extend(self._list_contextual())

        if player_type:
            all_models = [m for m in all_models if m.player_type == player_type]

        return sorted(all_models, key=lambda m: (m.model_type, m.name, m.created_at))

    def get_store(self, model_type: str) -> BaseModelStore:
        """Get the store for a given model type.

        Args:
            model_type: One of "gb_residual", "mtl", "mle".

        Raises:
            ValueError: If model_type is unknown or "contextual" (use
                contextual_store directly for that).
        """
        stores: dict[str, BaseModelStore] = {
            "gb_residual": self.gb_store,
            "mtl": self.mtl_store,
            "mle": self.mle_store,
        }
        if model_type not in stores:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Valid types: {', '.join(stores)}. "
                "For contextual models, use registry.contextual_store directly."
            )
        return stores[model_type]

    def next_version(self, base_name: str, model_type: str, player_type: str) -> int:
        """Return the next version number for a model base name.

        Scans existing models matching the base name pattern and returns
        max(version) + 1, or 1 if no versions exist.

        Args:
            base_name: Base model name (e.g. "default").
            model_type: Store type ("gb_residual", "mtl", "mle").
            player_type: "batter" or "pitcher".
        """
        versions = self.versions_of(base_name, model_type, player_type)
        if not versions:
            return 1
        return max(m.version for m in versions) + 1

    def versions_of(
        self,
        base_name: str,
        model_type: str,
        player_type: str | None = None,
    ) -> list[ModelMetadata]:
        """List all versions of a model, sorted by version number.

        Matches both the base name exactly (treated as version 1) and
        versioned names like "{base_name}_v2", "{base_name}_v3".

        Args:
            base_name: Base model name (e.g. "default").
            model_type: Store type ("gb_residual", "mtl", "mle").
            player_type: Optional filter by player type.
        """
        store = self.get_store(model_type)
        all_models = store.list_models()

        matching: list[ModelMetadata] = []
        for m in all_models:
            if player_type and m.player_type != player_type:
                continue
            parsed_base = _parse_base_name(m.name)
            if parsed_base == base_name:
                matching.append(m)

        return sorted(matching, key=lambda m: m.version)

    def compare(
        self,
        name_a: str,
        name_b: str,
        model_type: str,
        player_type: str,
    ) -> dict[str, Any]:
        """Compare metrics between two model versions.

        Returns a dict with side-by-side metadata and metric diffs.
        """
        store = self.get_store(model_type)
        meta_a = store.get_metadata(name_a, player_type)
        meta_b = store.get_metadata(name_b, player_type)

        if meta_a is None:
            raise FileNotFoundError(f"Model not found: {name_a} ({player_type})")
        if meta_b is None:
            raise FileNotFoundError(f"Model not found: {name_b} ({player_type})")

        return {
            "a": meta_a.to_dict(),
            "b": meta_b.to_dict(),
            "training_years_diff": {
                "a_only": sorted(set(meta_a.training_years) - set(meta_b.training_years)),
                "b_only": sorted(set(meta_b.training_years) - set(meta_a.training_years)),
            },
            "metrics_diff": _diff_metrics(meta_a.metrics, meta_b.metrics),
        }

    def _list_contextual(self) -> list[ModelMetadata]:
        """Adapt contextual metadata to the unified ModelMetadata format."""
        checkpoints = self.contextual_store.list_checkpoints()
        result: list[ModelMetadata] = []
        for cp in checkpoints:
            # Infer player_type from perspective or name
            player_type = ""
            if cp.perspective:
                player_type = cp.perspective
            elif "batter" in cp.name:
                player_type = "batter"
            elif "pitcher" in cp.name:
                player_type = "pitcher"

            metrics: dict[str, Any] = {
                "epoch": cp.epoch,
                "train_loss": cp.train_loss,
                "val_loss": cp.val_loss,
            }
            if cp.pitch_type_accuracy is not None:
                metrics["pitch_type_accuracy"] = cp.pitch_type_accuracy
            if cp.pitch_result_accuracy is not None:
                metrics["pitch_result_accuracy"] = cp.pitch_result_accuracy
            if cp.per_stat_mse is not None:
                metrics["per_stat_mse"] = cp.per_stat_mse

            result.append(
                ModelMetadata(
                    name=cp.name,
                    model_type="contextual",
                    player_type=player_type,
                    version=1,
                    training_years=(),
                    stats=list(cp.target_stats) if cp.target_stats else [],
                    feature_names=[],
                    created_at=cp.created_at or "",
                    metrics=metrics,
                )
            )
        return result


def _parse_base_name(name: str) -> str:
    """Extract base name from a potentially versioned name.

    "default" -> "default"
    "default_v2" -> "default"
    "experiment_v10" -> "experiment"
    """
    match = _VERSION_PATTERN.match(name)
    if match:
        return match.group(1)
    return name


def _diff_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compute differences between two metrics dicts."""
    all_keys = sorted(set(a) | set(b))
    diff: dict[str, Any] = {}
    for key in all_keys:
        val_a = a.get(key)
        val_b = b.get(key)
        if isinstance(val_a, int | float) and isinstance(val_b, int | float):
            diff[key] = {"a": val_a, "b": val_b, "delta": val_b - val_a}
        else:
            diff[key] = {"a": val_a, "b": val_b}
    return diff
