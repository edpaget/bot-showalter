from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.statcast_data import (
        StatcastBatterStats,
        StatcastDataSource,
    )
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterBabipConfig:
    adjustment_weight: float = 0.5
    min_pa_for_adjustment: int = 100
    league_hr_per_barrel: float = 0.55


class BatterBabipAdjuster:
    def __init__(
        self,
        statcast_source: StatcastDataSource,
        id_mapper: PlayerIdMapper,
        config: BatterBabipConfig | None = None,
    ) -> None:
        self._statcast_source = statcast_source
        self._id_mapper = id_mapper
        self._config = config or BatterBabipConfig()
        self._statcast_lookup: dict[str, StatcastBatterStats] | None = None

    def _ensure_statcast_data(self, year: int) -> dict[str, StatcastBatterStats]:
        if self._statcast_lookup is None:
            stats = self._statcast_source.batter_expected_stats(year - 1)
            self._statcast_lookup = {s.player_id: s for s in stats}
            logger.debug(
                "BatterBabipAdjuster loaded %d Statcast records for year %d",
                len(self._statcast_lookup),
                year - 1,
            )
        return self._statcast_lookup

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        if not players:
            return []

        year = players[0].year
        lookup = self._ensure_statcast_data(year)
        result: list[PlayerRates] = []

        for p in players:
            if self._is_pitcher(p):
                result.append(p)
                continue

            statcast = self._find_statcast(p.player_id, lookup)
            if statcast is None or statcast.pa < self._config.min_pa_for_adjustment:
                result.append(p)
                continue

            bip_rate = self._bip_rate(p)
            if bip_rate <= 0:
                result.append(p)
                continue

            result.append(self._adjust_player(p, statcast, bip_rate))

        return result

    def _is_pitcher(self, player: PlayerRates) -> bool:
        return "pa_per_year" not in player.metadata

    def _find_statcast(
        self, fg_id: str, lookup: dict[str, StatcastBatterStats]
    ) -> StatcastBatterStats | None:
        mlbam_id = self._id_mapper.fangraphs_to_mlbam(fg_id)
        if mlbam_id is None:
            return None
        return lookup.get(mlbam_id)

    def _bip_rate(self, player: PlayerRates) -> float:
        rates = player.rates
        bb = rates.get("bb", 0.0)
        so = rates.get("so", 0.0)
        hbp = rates.get("hbp", 0.0)
        sf = rates.get("sf", 0.0)
        sh = rates.get("sh", 0.0)
        return 1.0 - (bb + so + hbp + sf + sh)

    def _adjust_player(
        self,
        player: PlayerRates,
        statcast: StatcastBatterStats,
        bip_rate: float,
    ) -> PlayerRates:
        rates = player.rates
        bb = rates.get("bb", 0.0)
        hbp = rates.get("hbp", 0.0)
        sf = rates.get("sf", 0.0)
        sh = rates.get("sh", 0.0)
        ab_per_pa = 1.0 - (bb + hbp + sf + sh)

        # Expected BABIP from Statcast xBA
        x_hr_rate = statcast.barrel_rate * self._config.league_hr_per_barrel * bip_rate
        x_babip = (statcast.xba * ab_per_pa - x_hr_rate) / bip_rate

        # Observed BABIP from current rates
        singles = rates.get("singles", 0.0)
        doubles = rates.get("doubles", 0.0)
        triples = rates.get("triples", 0.0)
        hits_on_bip = singles + doubles + triples
        observed_babip = hits_on_bip / bip_rate

        # Adjust singles toward xBABIP
        babip_delta = x_babip - observed_babip
        w = self._config.adjustment_weight
        singles_adjustment = babip_delta * w * bip_rate
        new_singles = max(0.0, singles + singles_adjustment)

        new_rates = dict(rates)
        new_rates["singles"] = new_singles

        metadata = dict(player.metadata)
        metadata["x_babip"] = x_babip
        metadata["observed_batter_babip"] = observed_babip
        metadata["babip_singles_adjustment"] = singles_adjustment

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=new_rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )
