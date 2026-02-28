from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerProfile, compute_age

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Valuation
    from fantasy_baseball_manager.repos import PlayerRepo
_PITCHER_POSITIONS = {"SP", "RP"}


class PlayerProfileService:
    """Provides player profile data (bio + position) in a single call."""

    def __init__(self, player_repo: PlayerRepo) -> None:
        self._player_repo = player_repo

    def get_profile(self, player_id: int, season: int) -> PlayerProfile | None:
        """Look up a single player profile."""
        player = self._player_repo.get_by_id(player_id)
        if player is None:
            return None
        return PlayerProfile(
            player_id=player_id,
            name=f"{player.name_first} {player.name_last}",
            age=compute_age(player.birth_date, season),
            bats=player.bats,
            throws=player.throws,
        )

    def get_profiles(
        self,
        player_ids: list[int],
        season: int,
        *,
        positions: dict[int, tuple[str, ...]] | None = None,
    ) -> dict[int, PlayerProfile]:
        """Batch-lookup profiles, optionally enriching with position data."""
        players = self._player_repo.get_by_ids(player_ids)
        result: dict[int, PlayerProfile] = {}
        for player in players:
            if player.id is None:
                continue
            pos = positions.get(player.id, ()) if positions is not None else ()
            result[player.id] = PlayerProfile(
                player_id=player.id,
                name=f"{player.name_first} {player.name_last}",
                age=compute_age(player.birth_date, season),
                bats=player.bats,
                throws=player.throws,
                positions=pos,
                pitcher_type=_derive_pitcher_type(pos),
            )
        return result

    def enrich_valuations(
        self,
        valuations: list[Valuation],
        season: int,
        *,
        positions: dict[int, tuple[str, ...]] | None = None,
    ) -> dict[int, PlayerProfile]:
        """Convenience wrapper that extracts player IDs from valuations."""
        player_ids = [v.player_id for v in valuations]
        return self.get_profiles(player_ids, season, positions=positions)


def _derive_pitcher_type(positions: tuple[str, ...]) -> str | None:
    """Derive pitcher classification from positions tuple."""
    pitcher_pos = [p for p in positions if p in _PITCHER_POSITIONS]
    if not pitcher_pos:
        return None
    has_sp = "SP" in pitcher_pos
    has_rp = "RP" in pitcher_pos
    if has_sp and has_rp:
        return "SP/RP"
    if has_sp:
        return "SP"
    return "RP"
