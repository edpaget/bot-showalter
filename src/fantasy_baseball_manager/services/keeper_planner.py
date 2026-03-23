"""Pre-draft keeper planner service.

Orchestrates the keeper optimizer, adjusted valuations, scarcity analysis,
and category tracking into scenario comparisons for pre-draft planning.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    KeeperConstraints,
    KeeperPlanResult,
    KeeperScenarioResult,
)
from fantasy_baseball_manager.services.category_tracker import analyze_roster, identify_needs
from fantasy_baseball_manager.services.keeper_optimizer import solve_keepers
from fantasy_baseball_manager.services.keeper_service import (
    _best_valuation_for_player,
    compute_adjusted_valuations,
    compute_surplus,
)
from fantasy_baseball_manager.services.positional_scarcity import compute_scarcity

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        KeeperCost,
        KeeperDecision,
        LeagueSettings,
        Player,
        Projection,
        Valuation,
    )

logger = logging.getLogger(__name__)


class KeeperPlannerService:
    """Orchestrates keeper scenario planning for pre-draft analysis."""

    def __init__(
        self,
        *,
        keeper_costs: list[KeeperCost],
        valuations: list[Valuation],
        players: list[Player],
        projections: list[Projection],
        league: LeagueSettings,
        batter_positions: dict[int, list[str]],
        pitcher_positions: dict[int, list[str]],
    ) -> None:
        self._keeper_costs = keeper_costs
        self._valuations = valuations
        self._players = players
        self._projections = projections
        self._league = league
        self._batter_positions = batter_positions
        self._pitcher_positions = pitcher_positions
        self._cache: dict[frozenset[int], KeeperScenarioResult] = {}

    def plan(
        self,
        season: int,
        max_keepers: int,
        *,
        custom_scenarios: list[set[int]] | None = None,
        board_preview_size: int = 20,
    ) -> KeeperPlanResult:
        """Compute keeper scenarios with adjusted boards, scarcity, and category needs."""
        # Build decisions from keeper costs
        decisions = compute_surplus(self._keeper_costs, self._valuations, self._players)
        decision_lookup = {(d.player_id, d.player_type): d for d in decisions}

        # Collect scenario keeper sets
        scenario_sets: list[frozenset[int]] = []

        if decisions:
            try:
                constraints = KeeperConstraints(max_keepers=max_keepers)
                solution = solve_keepers(decisions, constraints)
                # Add optimal
                optimal_ids = frozenset(p.player_id for p in solution.optimal.players)
                scenario_sets.append(optimal_ids)
                # Add alternatives
                for alt in solution.alternatives:
                    alt_ids = frozenset(p.player_id for p in alt.players)
                    if alt_ids not in scenario_sets:
                        scenario_sets.append(alt_ids)
            except ValueError:
                logger.info("No valid keeper sets found from optimizer")

        # Add custom scenarios
        if custom_scenarios:
            for cs in custom_scenarios:
                fs = frozenset(cs)
                if fs not in scenario_sets:
                    scenario_sets.append(fs)

        if not scenario_sets:
            return KeeperPlanResult(scenarios=())

        # Compute each scenario
        results: list[KeeperScenarioResult] = []
        for keeper_ids in scenario_sets:
            if keeper_ids in self._cache:
                results.append(self._cache[keeper_ids])
                continue

            scenario = self._compute_scenario(keeper_ids, decision_lookup, board_preview_size)
            self._cache[keeper_ids] = scenario
            results.append(scenario)

        return KeeperPlanResult(scenarios=tuple(results))

    def _compute_scenario(
        self,
        keeper_ids: frozenset[int],
        decision_lookup: dict[tuple[int, str], KeeperDecision],
        board_preview_size: int,
    ) -> KeeperScenarioResult:
        """Compute a single keeper scenario's adjusted board, scarcity, and needs."""
        # Get keeper decisions for this set — match by player_id across all types
        keeper_decisions = tuple(d for (pid, _), d in decision_lookup.items() if pid in keeper_ids)
        total_surplus = sum(d.surplus for d in keeper_decisions)

        # Compute adjusted valuations
        adjusted = compute_adjusted_valuations(
            {(pid, None) for pid in keeper_ids},
            self._projections,
            self._batter_positions,
            self._pitcher_positions,
            self._league,
            self._valuations,
            self._players,
        )

        # Board preview: top N by adjusted value
        board_preview = tuple(sorted(adjusted, key=lambda a: a.adjusted_value, reverse=True)[:board_preview_size])

        # Convert adjusted valuations to Valuation-like objects for scarcity.
        # Key by (player_id, player_type) to avoid collisions for two-way players.
        val_lookup: dict[tuple[int, str], Valuation] = {}
        for v in self._valuations:
            key = (v.player_id, v.player_type)
            existing = val_lookup.get(key)
            if existing is None or v.value > existing.value:
                val_lookup[key] = v

        adjusted_as_valuations: list[Valuation] = []
        for adj in adjusted:
            orig = _best_valuation_for_player(adj.player_id, val_lookup)
            if orig is not None:
                adjusted_as_valuations.append(replace(orig, value=adj.adjusted_value))

        scarcity = tuple(compute_scarcity(adjusted_as_valuations, self._league))

        # Category analysis
        keeper_id_list = list(keeper_ids)
        analysis = analyze_roster(keeper_id_list, self._projections, self._league)

        # Available pool for needs = all projected players minus keepers
        proj_player_ids = {p.player_id for p in self._projections}
        available_ids = sorted(proj_player_ids - keeper_ids)

        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in self._players if p.id is not None}
        category_needs = tuple(
            identify_needs(keeper_id_list, available_ids, self._projections, self._league, player_names)
        )

        return KeeperScenarioResult(
            keeper_ids=keeper_ids,
            keeper_decisions=keeper_decisions,
            total_surplus=total_surplus,
            board_preview=board_preview,
            scarcity=scarcity,
            category_needs=category_needs,
            strongest_categories=tuple(analysis.strongest_categories),
            weakest_categories=tuple(analysis.weakest_categories),
        )
