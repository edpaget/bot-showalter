"""Skill change adjuster that applies projection adjustments based on YoY skill changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.pipeline.skill_data import (
    BatterSkillDelta,
    PitcherSkillDelta,
    SkillDeltaComputer,
    SkillDeltaComputerProtocol,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillChangeConfig:
    """Thresholds and adjustment factors for skill change detection."""

    # Minimum PA required for reliable skill deltas
    min_pa: int = 200

    # Batter thresholds (absolute delta required to trigger adjustment)
    barrel_rate_threshold: float = 0.02  # 2 percentage points
    hard_hit_rate_threshold: float = 0.03  # 3 percentage points
    exit_velo_max_threshold: float = 1.5  # mph
    chase_rate_threshold: float = 0.03  # 3 percentage points
    whiff_rate_threshold: float = 0.03  # 3 percentage points
    sprint_speed_threshold: float = 0.5  # ft/sec

    # Pitcher thresholds
    fastball_velo_threshold: float = 1.0  # mph
    pitcher_whiff_threshold: float = 0.03  # 3 percentage points
    gb_rate_threshold: float = 0.03  # 3 percentage points

    # Batter adjustment factors (how much to adjust rate per unit of skill change)
    barrel_to_hr_factor: float = 0.5  # HR rate += delta * factor
    barrel_to_doubles_factor: float = 0.3
    hard_hit_to_hr_factor: float = 0.2
    exit_velo_to_hr_factor: float = 0.005  # per mph
    chase_to_bb_factor: float = -0.5  # lower chase = more BB (negative factor)
    chase_to_so_factor: float = 0.3  # lower chase = fewer SO
    whiff_to_so_factor: float = 0.7
    sprint_to_sb_factor: float = 0.02  # per ft/sec

    # Pitcher adjustment factors
    velo_to_so_factor: float = 0.003  # per mph
    velo_to_er_factor: float = -0.001  # per mph (higher velo = lower ER)
    pitcher_whiff_to_so_factor: float = 0.5
    gb_to_hr_factor: float = -0.1  # higher GB = fewer HR allowed


@dataclass
class SkillChangeAdjuster:
    """Pipeline adjuster that applies skill-change-based corrections."""

    delta_computer: SkillDeltaComputerProtocol
    config: SkillChangeConfig = field(default_factory=SkillChangeConfig)

    _batter_deltas: dict[str, BatterSkillDelta] | None = field(default=None, init=False)
    _pitcher_deltas: dict[str, PitcherSkillDelta] | None = field(default=None, init=False)
    _cached_year: int | None = field(default=None, init=False)

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        """Implements RateAdjuster protocol."""
        if not players:
            return []

        year = players[0].year
        self._ensure_deltas_loaded(year)

        return [self._adjust_player(p) for p in players]

    def _ensure_deltas_loaded(self, year: int) -> None:
        """Lazy-load skill deltas for the given year."""
        if self._cached_year != year:
            self._batter_deltas = self.delta_computer.compute_batter_deltas(year)
            self._pitcher_deltas = self.delta_computer.compute_pitcher_deltas(year)
            self._cached_year = year
            logger.debug(
                "Loaded skill deltas for year %d: %d batters, %d pitchers",
                year,
                len(self._batter_deltas),
                len(self._pitcher_deltas),
            )

    def _adjust_player(self, player: PlayerRates) -> PlayerRates:
        if self._is_batter(player):
            return self._adjust_batter(player)
        return self._adjust_pitcher(player)

    def _is_batter(self, player: PlayerRates) -> bool:
        """Check if player is a batter based on metadata."""
        return "pa_per_year" in player.metadata

    def _adjust_batter(self, player: PlayerRates) -> PlayerRates:
        """Apply skill-change adjustments to a batter."""
        assert self._batter_deltas is not None

        delta = self._batter_deltas.get(player.player_id)
        if delta is None:
            return player

        if not delta.has_sufficient_sample(self.config.min_pa):
            return player

        rates = dict(player.rates)
        adjustments: dict[str, float] = {}

        # Barrel rate -> HR, doubles
        if delta.barrel_rate_delta is not None and abs(delta.barrel_rate_delta) >= self.config.barrel_rate_threshold:
            hr_adj = delta.barrel_rate_delta * self.config.barrel_to_hr_factor
            rates["hr"] = max(0.0, rates.get("hr", 0.0) + hr_adj)
            adjustments["barrel->hr"] = hr_adj

            dbl_adj = delta.barrel_rate_delta * self.config.barrel_to_doubles_factor
            rates["doubles"] = max(0.0, rates.get("doubles", 0.0) + dbl_adj)
            adjustments["barrel->doubles"] = dbl_adj

        # Hard hit rate -> HR
        if (
            delta.hard_hit_rate_delta is not None
            and abs(delta.hard_hit_rate_delta) >= self.config.hard_hit_rate_threshold
        ):
            hr_adj = delta.hard_hit_rate_delta * self.config.hard_hit_to_hr_factor
            rates["hr"] = max(0.0, rates.get("hr", 0.0) + hr_adj)
            adjustments["hard_hit->hr"] = hr_adj

        # Exit velocity -> HR
        if (
            delta.exit_velo_max_delta is not None
            and abs(delta.exit_velo_max_delta) >= self.config.exit_velo_max_threshold
        ):
            hr_adj = delta.exit_velo_max_delta * self.config.exit_velo_to_hr_factor
            rates["hr"] = max(0.0, rates.get("hr", 0.0) + hr_adj)
            adjustments["ev_max->hr"] = hr_adj

        # Chase rate -> BB, SO (note: lower chase is better for hitter)
        if delta.chase_rate_delta is not None and abs(delta.chase_rate_delta) >= self.config.chase_rate_threshold:
            # Lower chase = more BB (negative delta * negative factor = positive adj)
            bb_adj = delta.chase_rate_delta * self.config.chase_to_bb_factor
            rates["bb"] = max(0.0, rates.get("bb", 0.0) + bb_adj)
            adjustments["chase->bb"] = bb_adj

            # Lower chase = fewer SO
            so_adj = delta.chase_rate_delta * self.config.chase_to_so_factor
            rates["so"] = max(0.0, rates.get("so", 0.0) + so_adj)
            adjustments["chase->so"] = so_adj

        # Whiff rate -> SO
        if delta.whiff_rate_delta is not None and abs(delta.whiff_rate_delta) >= self.config.whiff_rate_threshold:
            so_adj = delta.whiff_rate_delta * self.config.whiff_to_so_factor
            rates["so"] = max(0.0, rates.get("so", 0.0) + so_adj)
            adjustments["whiff->so"] = so_adj

        # Sprint speed -> SB (if sb rate exists)
        if (
            delta.sprint_speed_delta is not None
            and "sb" in rates
            and abs(delta.sprint_speed_delta) >= self.config.sprint_speed_threshold
        ):
            sb_adj = delta.sprint_speed_delta * self.config.sprint_to_sb_factor
            rates["sb"] = max(0.0, rates.get("sb", 0.0) + sb_adj)
            adjustments["sprint->sb"] = sb_adj

        if not adjustments:
            return player

        metadata: PlayerMetadata = {**player.metadata}
        metadata["skill_change_adjustments"] = adjustments

        logger.debug(
            "Applied skill change adjustments to %s: %s",
            player.name,
            adjustments,
        )

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )

    def _adjust_pitcher(self, player: PlayerRates) -> PlayerRates:
        """Apply skill-change adjustments to a pitcher."""
        assert self._pitcher_deltas is not None

        delta = self._pitcher_deltas.get(player.player_id)
        if delta is None:
            return player

        if not delta.has_sufficient_sample(self.config.min_pa):
            return player

        rates = dict(player.rates)
        adjustments: dict[str, float] = {}

        # Fastball velocity -> SO, ER
        if (
            delta.fastball_velo_delta is not None
            and abs(delta.fastball_velo_delta) >= self.config.fastball_velo_threshold
        ):
            so_adj = delta.fastball_velo_delta * self.config.velo_to_so_factor
            rates["so"] = max(0.0, rates.get("so", 0.0) + so_adj)
            adjustments["velo->so"] = so_adj

            if "er" in rates:
                er_adj = delta.fastball_velo_delta * self.config.velo_to_er_factor
                rates["er"] = max(0.0, rates.get("er", 0.0) + er_adj)
                adjustments["velo->er"] = er_adj

        # Whiff rate -> SO
        if delta.whiff_rate_delta is not None and abs(delta.whiff_rate_delta) >= self.config.pitcher_whiff_threshold:
            so_adj = delta.whiff_rate_delta * self.config.pitcher_whiff_to_so_factor
            rates["so"] = max(0.0, rates.get("so", 0.0) + so_adj)
            adjustments["whiff->so"] = so_adj

        # GB rate -> HR allowed
        if delta.gb_rate_delta is not None and abs(delta.gb_rate_delta) >= self.config.gb_rate_threshold:
            hr_adj = delta.gb_rate_delta * self.config.gb_to_hr_factor
            rates["hr"] = max(0.0, rates.get("hr", 0.0) + hr_adj)
            adjustments["gb->hr"] = hr_adj

        if not adjustments:
            return player

        metadata: PlayerMetadata = {**player.metadata}
        metadata["skill_change_adjustments"] = adjustments

        logger.debug(
            "Applied skill change adjustments to pitcher %s: %s",
            player.name,
            adjustments,
        )

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )


def build_skill_change_adjuster(
    skill_source: SkillDataSource,
    config: SkillChangeConfig | None = None,
) -> SkillChangeAdjuster:
    """Factory function to build a SkillChangeAdjuster."""
    delta_computer = SkillDeltaComputer(skill_source)
    return SkillChangeAdjuster(
        delta_computer=delta_computer,
        config=config or SkillChangeConfig(),
    )
