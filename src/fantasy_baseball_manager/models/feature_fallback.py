"""Shared helper for materializing feature sets with player-universe fallback.

When predicting future seasons, the feature spine may return 0 rows because
no actuals exist yet.  ``materialize_with_fallback`` detects missing seasons
and uses a ``PlayerUniverseProvider`` to inject the expected player IDs so
that features can still be computed from lagged data.
"""

import dataclasses
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.features import DatasetAssembler, FeatureSet, SpineFilter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.models.protocols import PlayerUniverseProvider


def materialize_with_fallback(
    assembler: DatasetAssembler,
    feature_set: FeatureSet,
    seasons: Sequence[int],
    player_type: str,
    player_universe: PlayerUniverseProvider | None = None,
) -> list[dict[str, Any]]:
    """Materialize a feature set, falling back to player_ids for missing seasons.

    If *player_universe* is provided and some requested seasons produce no rows,
    the provider supplies player IDs for those seasons so that a second
    materialization with an explicit player-ID spine filter can fill in the gap.
    """
    handle = assembler.get_or_materialize(feature_set)
    rows = assembler.read(handle)
    if player_universe is None:
        return rows
    present_seasons = {row["season"] for row in rows}
    missing_seasons = [s for s in seasons if s not in present_seasons]
    if not missing_seasons:
        return rows
    all_ids: set[int] = set()
    for season in missing_seasons:
        all_ids |= player_universe.get_player_ids(season, player_type)
    if not all_ids:
        return rows
    fallback_fs = dataclasses.replace(
        feature_set,
        seasons=tuple(missing_seasons),
        spine_filter=SpineFilter(
            player_type=feature_set.spine_filter.player_type,
            player_ids=tuple(sorted(all_ids)),
        ),
    )
    fb_handle = assembler.get_or_materialize(fallback_fs)
    return rows + assembler.read(fb_handle)
