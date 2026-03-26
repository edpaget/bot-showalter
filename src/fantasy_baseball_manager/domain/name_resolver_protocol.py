from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerIdentity, PlayerType


@runtime_checkable
class NameResolver(Protocol):
    def resolve(
        self,
        name: str,
        *,
        season: int | None = None,
        player_type: PlayerType | None = None,
    ) -> PlayerIdentity | None: ...

    def resolve_all(
        self,
        name: str,
        *,
        season: int | None = None,
        player_type: PlayerType | None = None,
    ) -> list[PlayerIdentity]: ...

    def register_alias(
        self,
        alias_name: str,
        identity: PlayerIdentity,
        source: str,
    ) -> None: ...
