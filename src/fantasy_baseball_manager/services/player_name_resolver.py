from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerAlias, PlayerIdentity, PlayerType
from fantasy_baseball_manager.name_utils import normalize_name

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import PlayerAliasRepo


class PlayerNameResolver:
    """Resolve player names to PlayerIdentity via the alias table."""

    def __init__(self, alias_repo: PlayerAliasRepo) -> None:
        self._alias_repo = alias_repo

    def resolve(
        self,
        name: str,
        *,
        season: int | None = None,
        player_type: PlayerType | None = None,
    ) -> PlayerIdentity | None:
        """Resolve a name to a single PlayerIdentity, or None if ambiguous/missing."""
        candidates = self._lookup_and_filter(name, season=season, player_type=player_type)
        identities = _deduplicate(candidates, player_type)
        if len(identities) == 1:
            return identities[0]
        return None

    def resolve_all(
        self,
        name: str,
        *,
        season: int | None = None,
        player_type: PlayerType | None = None,
    ) -> list[PlayerIdentity]:
        """Resolve a name to all matching PlayerIdentity candidates."""
        candidates = self._lookup_and_filter(name, season=season, player_type=player_type)
        return _deduplicate(candidates, player_type)

    def register_alias(
        self,
        alias_name: str,
        identity: PlayerIdentity,
        source: str,
    ) -> None:
        """Persist a new alias mapping for future lookups."""
        self._alias_repo.upsert(
            PlayerAlias(
                alias_name=normalize_name(alias_name),
                player_id=identity.player_id,
                player_type=identity.player_type,
                source=source,
            )
        )

    def _lookup_and_filter(
        self,
        name: str,
        *,
        season: int | None = None,
        player_type: PlayerType | None = None,
    ) -> list[PlayerAlias]:
        normalized = normalize_name(name)
        matches = self._alias_repo.find_by_name(normalized)
        if not matches:
            return []

        if player_type is not None:
            matches = [m for m in matches if m.player_type is None or m.player_type == player_type]

        if season is not None:
            matches = _filter_by_season(matches, season)

        return matches


def _filter_by_season(aliases: list[PlayerAlias], season: int) -> list[PlayerAlias]:
    """Keep aliases whose active range includes the given season."""
    active = []
    for a in aliases:
        if a.active_from is not None and a.active_from > season:
            continue
        if a.active_to is not None and a.active_to < season:
            continue
        active.append(a)
    return active if active else aliases


def _deduplicate(
    aliases: list[PlayerAlias],
    caller_type: PlayerType | None,
) -> list[PlayerIdentity]:
    """Collapse aliases to unique (player_id, player_type) identities."""
    seen: set[tuple[int, PlayerType]] = set()
    result: list[PlayerIdentity] = []
    for a in aliases:
        ptype = a.player_type or caller_type
        if ptype is None:
            continue
        key = (a.player_id, ptype)
        if key not in seen:
            seen.add(key)
            result.append(PlayerIdentity(a.player_id, ptype))
    return result
