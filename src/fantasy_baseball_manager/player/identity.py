"""Canonical player identity type.

Provides the Player dataclass as the canonical representation of player identity
across the system. Supports two-way players via yahoo_sub_id.

Usage:
    # Regular player
    player = Player(
        name="Mike Trout",
        yahoo_id="10155",
        fangraphs_id="10155",
    )

    # Two-way player (Ohtani as batter)
    ohtani_batter = Player(
        name="Shohei Ohtani",
        yahoo_id="10835",
        yahoo_sub_id="1000001",  # Yahoo's synthetic batter ID
        fangraphs_id="19755",
    )

    # Two-way player (Ohtani as pitcher)
    ohtani_pitcher = Player(
        name="Shohei Ohtani",
        yahoo_id="10835",
        yahoo_sub_id="1000002",  # Yahoo's synthetic pitcher ID
        fangraphs_id="19755",
    )

    # Check if player is two-way
    if player.is_two_way:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Player:
    """Canonical player identity.

    Attributes:
        name: Player's full name.
        yahoo_id: Yahoo player ID (the primary/real ID).
        yahoo_sub_id: Synthetic Yahoo ID for two-way players (e.g., "1000001").
            When present, this is the ID used in Yahoo's API for the specific
            position type (batter vs pitcher).
        fangraphs_id: FanGraphs player ID (populated by mappers).
        mlbam_id: MLB Advanced Media player ID (populated by mappers).
        team: Player's current team abbreviation.
        eligible_positions: Tuple of position strings the player is eligible for.
        age: Player's age.
    """

    name: str
    yahoo_id: str
    yahoo_sub_id: str | None = None
    fangraphs_id: str | None = None
    mlbam_id: str | None = None
    team: str | None = None
    eligible_positions: tuple[str, ...] = field(default_factory=tuple)
    age: int | None = None

    @property
    def is_two_way(self) -> bool:
        """Returns True if this player is a two-way player with a synthetic ID."""
        return self.yahoo_sub_id is not None

    @property
    def effective_yahoo_id(self) -> str:
        """Returns the effective Yahoo ID for API lookups.

        For two-way players, returns yahoo_sub_id (the synthetic ID).
        For regular players, returns yahoo_id.
        """
        return self.yahoo_sub_id if self.yahoo_sub_id is not None else self.yahoo_id

    def with_ids(
        self,
        *,
        fangraphs_id: str | None = None,
        mlbam_id: str | None = None,
    ) -> Player:
        """Return a new Player with additional IDs populated.

        Only non-None arguments override existing values.
        """
        return Player(
            name=self.name,
            yahoo_id=self.yahoo_id,
            yahoo_sub_id=self.yahoo_sub_id,
            fangraphs_id=fangraphs_id if fangraphs_id is not None else self.fangraphs_id,
            mlbam_id=mlbam_id if mlbam_id is not None else self.mlbam_id,
            team=self.team,
            eligible_positions=self.eligible_positions,
            age=self.age,
        )

    def with_metadata(
        self,
        *,
        team: str | None = None,
        eligible_positions: tuple[str, ...] | None = None,
        age: int | None = None,
    ) -> Player:
        """Return a new Player with additional metadata populated.

        Only non-None arguments override existing values.
        """
        return Player(
            name=self.name,
            yahoo_id=self.yahoo_id,
            yahoo_sub_id=self.yahoo_sub_id,
            fangraphs_id=self.fangraphs_id,
            mlbam_id=self.mlbam_id,
            team=team if team is not None else self.team,
            eligible_positions=eligible_positions if eligible_positions is not None else self.eligible_positions,
            age=age if age is not None else self.age,
        )
