"""Player identity enricher that stamps Player objects onto PlayerRates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.player.identity import Player

if TYPE_CHECKING:
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper


@dataclass
class PlayerIdentityEnricher:
    """Adjuster that creates enriched Player objects and stamps them on PlayerRates.

    Should run first in the adjuster chain so downstream stages can
    read ``player.player.mlbam_id`` instead of calling an id_mapper.

    Pipeline rate computers produce PlayerRates keyed by FanGraphs ID,
    so this enricher maps fangraphs_id → mlbam_id directly (rather than
    going through the yahoo_id-based callable interface).
    """

    mapper: SfbbMapper

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            mlbam_id = self.mapper.fangraphs_to_mlbam(p.player_id)
            enriched_player = Player(
                yahoo_id="",
                fangraphs_id=p.player_id,
                mlbam_id=mlbam_id,
                name=p.name,
            )
            result.append(
                PlayerRates(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    rates=p.rates,
                    opportunities=p.opportunities,
                    metadata=p.metadata,
                    player=enriched_player,
                )
            )
        return result
