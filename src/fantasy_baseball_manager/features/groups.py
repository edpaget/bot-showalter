from collections.abc import Sequence
from dataclasses import dataclass

from fantasy_baseball_manager.features.types import AnyFeature, FeatureSet, SpineFilter

_REGISTRY: dict[str, "FeatureGroup"] = {}


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    description: str
    player_type: str  # "batter" | "pitcher" | "both"
    features: tuple[AnyFeature, ...]


def register_group(group: FeatureGroup) -> FeatureGroup:
    """Register a feature group by name. Raises ValueError on duplicate."""
    if group.name in _REGISTRY:
        msg = f"Feature group '{group.name}' is already registered"
        raise ValueError(msg)
    _REGISTRY[group.name] = group
    return group


def get_group(name: str) -> FeatureGroup:
    """Return a registered feature group. Raises KeyError if not found."""
    if name not in _REGISTRY:
        raise KeyError(f"'{name}': no feature group registered with this name")
    return _REGISTRY[name]


def list_groups() -> list[str]:
    """Return sorted list of registered feature group names."""
    return sorted(_REGISTRY)


def _clear() -> None:
    """Clear the registry. For testing only."""
    _REGISTRY.clear()


def compose_feature_set(
    name: str,
    groups: Sequence[FeatureGroup],
    seasons: tuple[int, ...],
    source_filter: str | None = None,
    spine_filter: SpineFilter | None = None,
) -> FeatureSet:
    """Compose multiple feature groups into a single FeatureSet.

    Features are concatenated in order; duplicates (by name) are
    deduplicated with first occurrence winning.
    """
    seen: set[str] = set()
    features: list[AnyFeature] = []
    for group in groups:
        for feature in group.features:
            if feature.name not in seen:
                seen.add(feature.name)
                features.append(feature)
    return FeatureSet(
        name=name,
        features=tuple(features),
        seasons=seasons,
        source_filter=source_filter,
        spine_filter=spine_filter if spine_filter is not None else SpineFilter(),
    )
