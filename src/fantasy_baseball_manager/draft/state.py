from __future__ import annotations

from fantasy_baseball_manager.draft.models import DraftPick, DraftRanking, RosterConfig
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory

FILLED_POSITION_MULTIPLIER: float = 0.7


class DraftState:
    def __init__(
        self,
        roster_config: RosterConfig,
        player_values: list[PlayerValue],
        player_positions: dict[tuple[str, str], tuple[str, ...]],
        category_weights: dict[StatCategory, float],
    ) -> None:
        self._roster_config = roster_config
        self._player_values: dict[tuple[str, str], PlayerValue] = {
            (pv.player_id, pv.position_type): pv for pv in player_values
        }
        self._player_positions = player_positions
        self._category_weights = category_weights
        self._drafted: set[str] = set()
        self._slot_filled: dict[str, int] = {}
        self._slot_capacity: dict[str, int] = {slot.position: slot.count for slot in roster_config.slots}

    def draft_player(
        self,
        player_id: str,
        is_user: bool,
        position: str | None = None,
    ) -> DraftPick:
        name = ""
        for key, pv in self._player_values.items():
            if key[0] == player_id:
                name = pv.name
                break
        self._drafted.add(player_id)
        if is_user and position is not None:
            self._slot_filled[position] = self._slot_filled.get(position, 0) + 1
        return DraftPick(
            player_id=player_id,
            name=name,
            is_user=is_user,
            position=position,
        )

    def get_rankings(self, limit: int | None = None) -> list[DraftRanking]:
        rankings: list[DraftRanking] = []
        for pv in self._player_values.values():
            if pv.player_id in self._drafted:
                continue

            weighted_cats: list[CategoryValue] = []
            weighted_value = 0.0
            raw_value = 0.0
            for cv in pv.category_values:
                weight = self._category_weights.get(cv.category, 1.0)
                weighted_cv = CategoryValue(
                    category=cv.category,
                    raw_stat=cv.raw_stat,
                    value=cv.value * weight,
                )
                weighted_cats.append(weighted_cv)
                weighted_value += cv.value * weight
                raw_value += cv.value

            positions = self._player_positions.get((pv.player_id, pv.position_type), ())
            best_position, multiplier = self._find_best_position(positions)
            adjusted_value = weighted_value * multiplier

            rankings.append(
                DraftRanking(
                    rank=0,
                    player_id=pv.player_id,
                    name=pv.name,
                    eligible_positions=positions,
                    best_position=best_position,
                    position_multiplier=multiplier,
                    raw_value=raw_value,
                    weighted_value=weighted_value,
                    adjusted_value=adjusted_value,
                    category_values=tuple(weighted_cats),
                )
            )

        rankings.sort(key=lambda r: r.adjusted_value, reverse=True)

        result: list[DraftRanking] = []
        for i, r in enumerate(rankings[:limit], start=1):
            result.append(
                DraftRanking(
                    rank=i,
                    player_id=r.player_id,
                    name=r.name,
                    eligible_positions=r.eligible_positions,
                    best_position=r.best_position,
                    position_multiplier=r.position_multiplier,
                    raw_value=r.raw_value,
                    weighted_value=r.weighted_value,
                    adjusted_value=r.adjusted_value,
                    category_values=r.category_values,
                )
            )
        return result

    def position_needs(self) -> dict[str, int]:
        needs: dict[str, int] = {}
        for position, capacity in self._slot_capacity.items():
            filled = self._slot_filled.get(position, 0)
            remaining = max(0, capacity - filled)
            needs[position] = remaining
        return needs

    def _find_best_position(
        self,
        eligible: tuple[str, ...],
    ) -> tuple[str | None, float]:
        if not eligible:
            return None, 1.0

        best_pos: str | None = None
        best_remaining = -1

        for pos in eligible:
            capacity = self._slot_capacity.get(pos, 0)
            if capacity == 0:
                continue
            filled = self._slot_filled.get(pos, 0)
            remaining = capacity - filled
            if remaining > best_remaining:
                best_remaining = remaining
                best_pos = pos

        # Only fall back to Util/BN when all eligible positions are filled
        if best_remaining <= 0:
            for fallback in ("Util", "BN"):
                if fallback in self._slot_capacity:
                    capacity = self._slot_capacity[fallback]
                    filled = self._slot_filled.get(fallback, 0)
                    remaining = capacity - filled
                    if remaining > best_remaining:
                        best_remaining = remaining
                        best_pos = fallback

        if best_pos is None:
            return None, FILLED_POSITION_MULTIPLIER

        if best_remaining <= 0:
            return best_pos, FILLED_POSITION_MULTIPLIER

        return best_pos, 1.0
