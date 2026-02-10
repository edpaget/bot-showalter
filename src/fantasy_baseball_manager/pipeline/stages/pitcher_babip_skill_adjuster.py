from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore

logger = logging.getLogger(__name__)

REQUIRED_RATES = ("h", "hr", "so")


@dataclass(frozen=True)
class PitcherBabipSkillConfig:
    blend_weight: float = 0.40
    min_pa: int = 300
    base_babip: float = 0.300
    gb_coeff: float = -0.10
    iffb_coeff: float = -0.15
    ld_coeff: float = 0.20
    lg_gb_pct: float = 0.43
    lg_iffb_pct: float = 0.10
    lg_ld_pct: float = 0.20
    min_babip: float = 0.250
    max_babip: float = 0.340


class PitcherBabipSkillAdjuster:
    def __init__(
        self,
        feature_store: FeatureStore,
        config: PitcherBabipSkillConfig | None = None,
    ) -> None:
        self._feature_store = feature_store
        self._config = config or PitcherBabipSkillConfig()

    def _ensure_data(self, year: int) -> dict[str, PitcherBattedBallStats]:
        return self._feature_store.pitcher_batted_ball(year - 1)

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        if not players:
            return []

        year = players[0].year
        lookup = self._ensure_data(year)
        result: list[PlayerRates] = []

        for p in players:
            if self._is_batter(p):
                result.append(p)
                continue

            if not self._has_required_rates(p):
                result.append(p)
                continue

            if "expected_babip" not in p.metadata:
                result.append(p)
                continue

            bb_stats = lookup.get(p.player_id)
            if bb_stats is None or bb_stats.pa < self._config.min_pa:
                result.append(p)
                continue

            result.append(self._adjust_pitcher(p, bb_stats))

        return result

    def _is_batter(self, player: PlayerRates) -> bool:
        return "pa_per_year" in player.metadata

    def _has_required_rates(self, player: PlayerRates) -> bool:
        return all(stat in player.rates for stat in REQUIRED_RATES)

    def _adjust_pitcher(self, player: PlayerRates, bb_stats: PitcherBattedBallStats) -> PlayerRates:
        cfg = self._config

        # Compute skill-based xBABIP from batted-ball profile
        x_babip = (
            cfg.base_babip
            + cfg.gb_coeff * (bb_stats.gb_pct - cfg.lg_gb_pct)
            + cfg.iffb_coeff * (bb_stats.iffb_pct - cfg.lg_iffb_pct)
            + cfg.ld_coeff * (bb_stats.ld_pct - cfg.lg_ld_pct)
        )
        x_babip = max(cfg.min_babip, min(cfg.max_babip, x_babip))

        # Blend with normalization's expected BABIP
        expected_babip = player.metadata["expected_babip"]
        w = cfg.blend_weight
        new_babip = w * x_babip + (1.0 - w) * expected_babip

        # Recompute H rate from new BABIP
        rates = dict(player.rates)
        hr = rates["hr"]
        so = rates["so"]
        bb = rates.get("bb", 0.0)
        hbp = rates.get("hbp", 0.0)

        denom = 1.0 - new_babip
        h_new = rates["h"] if abs(denom) < 1e-9 else (hr + new_babip * (1.0 - hr - so)) / denom

        # Recompute ER from new H rate + LOB% from metadata
        expected_lob = player.metadata.get("expected_lob_pct", 0.73)
        baserunners = h_new - hr + bb + hbp
        er_new = baserunners * (1.0 - expected_lob) + hr

        rates["h"] = h_new
        rates["er"] = er_new

        metadata: PlayerMetadata = {**player.metadata}
        metadata["pitcher_x_babip"] = x_babip
        metadata["pitcher_gb_pct"] = bb_stats.gb_pct
        metadata["pitcher_babip_skill_blended"] = new_babip

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
