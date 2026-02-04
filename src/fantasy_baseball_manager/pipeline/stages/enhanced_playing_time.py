"""Enhanced playing time projector with injury, age, and volatility adjustments."""

from __future__ import annotations

from fantasy_baseball_manager.marcel.weights import projected_ip, projected_pa
from fantasy_baseball_manager.pipeline.stages.playing_time_config import (
    PlayingTimeConfig,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates


class EnhancedPlayingTimeProjector:
    """Playing time projector with injury, age, and volatility adjustments.

    Applies multiplicative factors to the base Marcel playing time formula:
        projected_pt = base_pt * injury_factor * age_factor * volatility_factor
        projected_pt = min(projected_pt, role_cap)
    """

    def __init__(self, config: PlayingTimeConfig | None = None) -> None:
        self._config = config or PlayingTimeConfig()

    def project(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            pa_per_year = p.metadata.get("pa_per_year")
            ip_per_year = p.metadata.get("ip_per_year")

            if pa_per_year is not None:
                opps, pt_metadata = self._project_batter(p, pa_per_year)
            elif ip_per_year is not None:
                # Handle single float value (convert to list for consistency)
                ip_list = [ip_per_year] if isinstance(ip_per_year, (int, float)) else ip_per_year
                opps, pt_metadata = self._project_pitcher(p, ip_list)
            else:
                opps = p.opportunities
                pt_metadata: PlayerMetadata = {}

            merged_metadata: PlayerMetadata = {**p.metadata, **pt_metadata}

            result.append(
                PlayerRates(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    rates=p.rates,
                    opportunities=opps,
                    metadata=merged_metadata,
                )
            )
        return result

    def _project_batter(
        self,
        player: PlayerRates,
        pa_per_year: list[float],
    ) -> tuple[float, PlayerMetadata]:
        """Project batter playing time with adjustments."""
        cfg = self._config

        # Base Marcel projection
        base_pa = projected_pa(
            pa_y1=pa_per_year[0],
            pa_y2=pa_per_year[1] if len(pa_per_year) > 1 else 0,
        )

        # Calculate factors
        games_per_year = player.metadata.get("games_per_year")
        injury_factor = self._compute_injury_factor_batter(pa_per_year, games_per_year)
        age_factor = self._compute_age_factor(player.age)
        volatility_factor = self._compute_volatility_factor(pa_per_year)

        # Apply factors
        projected_pa_adj = base_pa * injury_factor * age_factor * volatility_factor

        # Apply role cap
        position = player.metadata.get("position")
        cap = cfg.catcher_pa_cap if position == "C" else cfg.batter_pa_cap
        projected_pa_final = min(projected_pa_adj, cap)

        pt_metadata: PlayerMetadata = {
            "injury_factor": injury_factor,
            "age_pt_factor": age_factor,
            "volatility_factor": volatility_factor,
            "base_pa": base_pa,
        }

        return projected_pa_final, pt_metadata

    def _project_pitcher(
        self,
        player: PlayerRates,
        ip_per_year: list[float],
    ) -> tuple[float, PlayerMetadata]:
        """Project pitcher playing time with adjustments."""
        cfg = self._config
        is_starter = player.metadata.get("is_starter", True)

        # Base Marcel projection
        base_ip = projected_ip(
            ip_y1=ip_per_year[0],
            ip_y2=ip_per_year[1] if len(ip_per_year) > 1 else 0,
            is_starter=is_starter,
        )

        # Calculate factors
        games_per_year = player.metadata.get("games_per_year")
        injury_factor = self._compute_injury_factor_pitcher(ip_per_year, games_per_year, is_starter)
        age_factor = self._compute_age_factor(player.age)
        volatility_factor = self._compute_volatility_factor(ip_per_year)

        # Apply factors
        projected_ip_adj = base_ip * injury_factor * age_factor * volatility_factor

        # Apply role cap
        cap = cfg.starter_ip_cap if is_starter else cfg.reliever_ip_cap
        projected_ip_final = min(projected_ip_adj, cap)

        # Convert to outs for rate multiplication
        opps = projected_ip_final * 3

        pt_metadata: PlayerMetadata = {
            "injury_factor": injury_factor,
            "age_pt_factor": age_factor,
            "volatility_factor": volatility_factor,
            "base_ip": base_ip,
        }

        return opps, pt_metadata

    def _compute_injury_factor_batter(
        self,
        pa_per_year: list[float],
        games_per_year: list[float] | None,
    ) -> float:
        """Compute injury factor for batters based on games played proxy."""
        if games_per_year is not None and len(games_per_year) > 0:
            # Use actual games data
            actual_games = games_per_year[0]
            expected_games = 162.0
        else:
            # Infer games from PA (league avg ~4 PA/game)
            actual_games = pa_per_year[0] / 4.0 if pa_per_year else 0.0
            expected_games = 162.0

        return self._interpolate_injury_factor(actual_games, expected_games)

    def _compute_injury_factor_pitcher(
        self,
        ip_per_year: list[float],
        games_per_year: list[float] | None,
        is_starter: bool,
    ) -> float:
        """Compute injury factor for pitchers based on games/starts."""
        if games_per_year is not None and len(games_per_year) > 0:
            actual_games = games_per_year[0]
        elif is_starter:
            # Starters ~6 IP/start, typical season ~32 starts
            actual_games = ip_per_year[0] / 6.0 if ip_per_year else 0.0
        else:
            # Relievers ~1 IP/appearance, typical season ~60-70 appearances
            actual_games = ip_per_year[0] / 1.0 if ip_per_year else 0.0

        # Expected games based on role
        expected_games = 32.0 if is_starter else 65.0

        return self._interpolate_injury_factor(actual_games, expected_games)

    def _interpolate_injury_factor(
        self,
        actual_games: float,
        expected_games: float,
    ) -> float:
        """Interpolate injury factor based on games played ratio."""
        cfg = self._config

        if expected_games <= 0:
            return 1.0

        games_ratio = actual_games / expected_games

        if games_ratio >= 1.0:
            return 1.0
        elif games_ratio <= cfg.min_games_pct:
            return 1.0 - cfg.games_played_weight
        else:
            # Linear interpolation between min_games_pct and 1.0
            range_pct = 1.0 - cfg.min_games_pct
            position_in_range = (games_ratio - cfg.min_games_pct) / range_pct
            return (1.0 - cfg.games_played_weight) + (cfg.games_played_weight * position_in_range)

    def _compute_age_factor(self, age: int) -> float:
        """Compute age-based playing time decline factor."""
        cfg = self._config

        years_past_threshold = max(0, age - cfg.age_decline_start)
        decline = years_past_threshold * cfg.age_decline_rate

        return max(0.0, 1.0 - decline)

    def _compute_volatility_factor(self, values_per_year: list[float]) -> float:
        """Compute volatility penalty based on year-over-year variance."""
        cfg = self._config

        if len(values_per_year) < 2:
            return 1.0

        # Check for volatile year-over-year swings
        for i in range(len(values_per_year) - 1):
            current = values_per_year[i]
            previous = values_per_year[i + 1]

            if previous > 0:
                change_pct = abs(current - previous) / previous
                if change_pct > cfg.volatility_threshold:
                    return 1.0 - cfg.volatility_penalty

        return 1.0
