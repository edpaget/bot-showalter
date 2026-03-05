"""Playing-time resolution — shared logic for ensemble and Marcel.

Parses the ``playing_time`` config parameter and returns a
:class:`ConsensusLookup` (or ``None`` for native mode).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain.pt_normalization import (
    ConsensusLookup,
    build_consensus_lookup,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.domain.projection import Projection


def resolve_playing_time(
    pt_spec: str,
    season: int,
    fetch_projections: Callable[..., list[Projection]],
) -> ConsensusLookup | None:
    """Resolve a playing-time specification to a consensus lookup.

    Parameters
    ----------
    pt_spec:
        One of ``"native"``, ``"consensus"``, ``"consensus:sys1,sys2,..."``,
        or a single system name (e.g. ``"steamer"``, ``"playing-time-model"``).
    season:
        The season to fetch projections for.
    fetch_projections:
        Callable matching ``ProjectionRepo.get_by_season(season, system=...)``.

    Returns
    -------
    ``None`` for native mode, otherwise a :class:`ConsensusLookup`.
    """
    if pt_spec == "native":
        return None

    if pt_spec == "consensus":
        return _consensus_from_systems(["steamer", "zips"], season, fetch_projections)

    if pt_spec.startswith("consensus:"):
        systems = [s.strip() for s in pt_spec[len("consensus:") :].split(",")]
        return _consensus_from_systems(systems, season, fetch_projections)

    # Single system name
    return _single_system_lookup(pt_spec, season, fetch_projections)


def _available_pt_system_names(
    season: int,
    fetch_projections: Callable[..., list[Projection]],
) -> list[str]:
    """Return sorted unique system names that have PA or IP projections."""
    all_projs = fetch_projections(season)
    systems: set[str] = set()
    for p in all_projs:
        pa = p.stat_json.get("pa", 0)
        ip = p.stat_json.get("ip", 0)
        if pa > 0 or ip > 0:
            systems.add(p.system)
    return sorted(systems)


def _consensus_from_systems(
    systems: list[str],
    season: int,
    fetch_projections: Callable[..., list[Projection]],
) -> ConsensusLookup:
    available: list[str] | None = None
    proj_lists: list[list[Projection]] = []
    for sys in systems:
        projs = fetch_projections(season, system=sys)
        if not projs:
            if available is None:
                available = _available_pt_system_names(season, fetch_projections)
            avail_str = ", ".join(available) if available else "none"
            warnings.warn(
                f"No projections found for system {sys!r} in consensus list for season {season}. "
                f"Available PT systems: {avail_str}",
                stacklevel=3,
            )
        proj_lists.append(projs)
    return build_consensus_lookup(*proj_lists)


def _single_system_lookup(
    system: str,
    season: int,
    fetch_projections: Callable[..., list[Projection]],
) -> ConsensusLookup:
    projections = fetch_projections(season, system=system)
    if not projections:
        available = _available_pt_system_names(season, fetch_projections)
        avail_str = ", ".join(available) if available else "none"
        warnings.warn(
            f"No projections found for system {system!r} in season {season}; "
            f"PT will fall back to native. Available PT systems: {avail_str}",
            stacklevel=3,
        )
        return ConsensusLookup(batting_pt={}, pitching_pt={})
    return build_consensus_lookup(projections)
