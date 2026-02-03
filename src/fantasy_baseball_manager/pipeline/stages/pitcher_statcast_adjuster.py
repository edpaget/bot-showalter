from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.statcast_data import (
        PitcherStatcastDataSource,
        StatcastPitcherStats,
    )
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)

BLENDED_STATS = ("h", "er")
REQUIRED_RATES = ("h", "er")


@dataclass(frozen=True)
class PitcherStatcastConfig:
    h_blend_weight: float = 0.0
    er_blend_weight: float = 0.25
    min_pa_for_blend: int = 200
    league_hr_per_barrel: float = 0.55


class PitcherStatcastAdjuster:
    def __init__(
        self,
        statcast_source: PitcherStatcastDataSource,
        id_mapper: PlayerIdMapper,
        config: PitcherStatcastConfig | None = None,
    ) -> None:
        self._statcast_source = statcast_source
        self._id_mapper = id_mapper
        self._config = config or PitcherStatcastConfig()
        self._statcast_lookup: dict[str, StatcastPitcherStats] | None = None

    def _ensure_statcast_data(self, year: int) -> dict[str, StatcastPitcherStats]:
        if self._statcast_lookup is None:
            stats = self._statcast_source.pitcher_expected_stats(year - 1)
            self._statcast_lookup = {s.player_id: s for s in stats}
            logger.debug(
                "PitcherStatcastAdjuster loaded %d records for year %d",
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
            if self._is_batter(p):
                result.append(p)
                continue

            if not self._has_required_rates(p):
                result.append(p)
                continue

            statcast = self._find_statcast(p.player_id, lookup)
            if statcast is None or statcast.pa < self._config.min_pa_for_blend:
                result.append(p)
                continue

            result.append(self._blend_player(p, statcast))

        return result

    def _is_batter(self, player: PlayerRates) -> bool:
        return "pa_per_year" in player.metadata

    def _has_required_rates(self, player: PlayerRates) -> bool:
        return all(stat in player.rates for stat in REQUIRED_RATES)

    def _find_statcast(
        self, fg_id: str, lookup: dict[str, StatcastPitcherStats]
    ) -> StatcastPitcherStats | None:
        mlbam_id = self._id_mapper.fangraphs_to_mlbam(fg_id)
        if mlbam_id is None:
            return None
        return lookup.get(mlbam_id)

    def _blend_player(self, player: PlayerRates, statcast: StatcastPitcherStats) -> PlayerRates:
        weights = {
            "h": self._config.h_blend_weight,
            "er": self._config.er_blend_weight,
        }
        rates = dict(player.rates)

        sc_rates = self._derive_statcast_rates(player, statcast)

        for stat in BLENDED_STATS:
            w = weights[stat]
            marcel_val = rates.get(stat, 0.0)
            statcast_val = sc_rates.get(stat, marcel_val)
            rates[stat] = w * statcast_val + (1.0 - w) * marcel_val

        metadata = dict(player.metadata)
        metadata["pitcher_xera"] = statcast.xera
        metadata["pitcher_xba_against"] = statcast.xba
        metadata["pitcher_statcast_blended"] = True
        metadata["pitcher_h_blend_weight"] = self._config.h_blend_weight
        metadata["pitcher_er_blend_weight"] = self._config.er_blend_weight

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )

    def _derive_statcast_rates(
        self, player: PlayerRates, statcast: StatcastPitcherStats
    ) -> dict[str, float]:
        rates = player.rates
        bb = rates.get("bb", 0.0)
        hbp = rates.get("hbp", 0.0)

        # AB per BF approximation (1 - walks - HBP)
        ab_per_bf = 1.0 - bb - hbp

        # Hit rate from xBA against
        x_h = statcast.xba * ab_per_bf

        # ER rate from xERA (convert from per-9-innings to per-out)
        x_er = statcast.xera / 27.0

        return {
            "h": x_h,
            "er": x_er,
        }
