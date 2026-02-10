from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
    from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats

logger = logging.getLogger(__name__)

BLENDED_STATS = ("hr", "singles", "doubles", "triples")


@dataclass(frozen=True)
class StatcastBlendConfig:
    blend_weight: float = 0.35
    min_pa_for_blend: int = 100
    league_hr_per_barrel: float = 0.55
    default_doubles_share: float = 0.85


class StatcastRateAdjuster:
    def __init__(
        self,
        feature_store: FeatureStore,
        config: StatcastBlendConfig | None = None,
    ) -> None:
        self._feature_store = feature_store
        self._config = config or StatcastBlendConfig()

    def _ensure_statcast_data(self, year: int) -> dict[str, StatcastBatterStats]:
        return self._feature_store.batter_statcast(year - 1)

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

            statcast = self._find_statcast(p, lookup)
            if statcast is None or statcast.pa < self._config.min_pa_for_blend:
                result.append(p)
                continue

            result.append(self._blend_player(p, statcast))

        return result

    def _is_pitcher(self, player: PlayerRates) -> bool:
        return "pa_per_year" not in player.metadata

    def _find_statcast(self, player: PlayerRates, lookup: dict[str, StatcastBatterStats]) -> StatcastBatterStats | None:
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return None
        return lookup.get(mlbam_id)

    def _blend_player(self, player: PlayerRates, statcast: StatcastBatterStats) -> PlayerRates:
        w = self._config.blend_weight
        rates = dict(player.rates)

        # Derive statcast-based rates
        sc_rates = self._derive_statcast_rates(player, statcast)

        # Blend only the hit-type rates
        for stat in BLENDED_STATS:
            marcel_val = rates.get(stat, 0.0)
            statcast_val = sc_rates.get(stat, marcel_val)
            rates[stat] = w * statcast_val + (1.0 - w) * marcel_val

        metadata: PlayerMetadata = {**player.metadata}
        metadata["statcast_blended"] = True
        metadata["statcast_xwoba"] = statcast.xwoba
        metadata["blend_weight_used"] = w

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
            player=player.player,
        )

    def _derive_statcast_rates(self, player: PlayerRates, statcast: StatcastBatterStats) -> dict[str, float]:
        rates = player.rates
        bb = rates.get("bb", 0.0)
        so = rates.get("so", 0.0)
        hbp = rates.get("hbp", 0.0)
        sf = rates.get("sf", 0.0)
        sh = rates.get("sh", 0.0)

        # BIP rate and AB/PA approximation
        bip_rate = 1.0 - (bb + so + hbp + sf + sh)
        ab_per_pa = 1.0 - (bb + hbp + sf + sh)

        # HR from barrel rate
        statcast_hr = statcast.barrel_rate * self._config.league_hr_per_barrel * bip_rate

        # Hit decomposition from xBA / xSLG
        xh_per_pa = statcast.xba * ab_per_pa
        hr_per_ab = statcast_hr / ab_per_pa if ab_per_pa > 0 else 0.0
        non_hr_iso = max(0.0, (statcast.xslg - statcast.xba) - 3.0 * hr_per_ab)

        # Split 2B/3B using player's existing ratio (or default)
        existing_2b = rates.get("doubles", 0.0)
        existing_3b = rates.get("triples", 0.0)
        total_xbh = existing_2b + existing_3b
        if total_xbh > 0:
            doubles_share = existing_2b / total_xbh
            triples_share = existing_3b / total_xbh
        else:
            doubles_share = self._config.default_doubles_share
            triples_share = 1.0 - doubles_share

        denom = doubles_share + 2.0 * triples_share
        xbh_per_ab = non_hr_iso / denom if denom > 0 else 0.0

        x2b = doubles_share * xbh_per_ab * ab_per_pa
        x3b = triples_share * xbh_per_ab * ab_per_pa
        x1b = max(0.0, xh_per_pa - statcast_hr - x2b - x3b)

        return {
            "hr": statcast_hr,
            "singles": x1b,
            "doubles": x2b,
            "triples": x3b,
        }
