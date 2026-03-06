from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow, LeagueSettings


class DraftFormat(StrEnum):
    SNAKE = "snake"
    AUCTION = "auction"
    LIVE = "live"


class DraftError(Exception): ...


@dataclass(frozen=True)
class DraftConfig:
    teams: int
    roster_slots: dict[str, int]
    format: DraftFormat
    user_team: int
    season: int
    budget: int = 0


@dataclass(frozen=True)
class DraftPick:
    pick_number: int
    team: int
    player_id: int
    player_name: str
    position: str
    price: int | None = None


@dataclass
class DraftState:
    config: DraftConfig
    picks: list[DraftPick]
    available_pool: dict[int, DraftBoardRow]
    team_rosters: dict[int, list[DraftPick]]
    team_budgets: dict[int, int]
    current_pick: int = 1


class DraftEngine:
    _state: DraftState | None = None
    _removed_rows: dict[int, DraftBoardRow]

    def __init__(self) -> None:
        self._removed_rows = {}

    @property
    def state(self) -> DraftState:
        return self._require_state()

    def _require_state(self) -> DraftState:
        if self._state is None:
            msg = "Draft not started"
            raise DraftError(msg)
        return self._state

    def start(self, players: list[DraftBoardRow], config: DraftConfig) -> DraftState:
        self._removed_rows = {}
        pool = {p.player_id: p for p in players}
        rosters: dict[int, list[DraftPick]] = {t: [] for t in range(1, config.teams + 1)}
        budgets: dict[int, int] = {
            t: config.budget if config.format == DraftFormat.AUCTION else 0 for t in range(1, config.teams + 1)
        }
        self._state = DraftState(
            config=config,
            picks=[],
            available_pool=pool,
            team_rosters=rosters,
            team_budgets=budgets,
        )
        return self._state

    def pick(
        self,
        player_id: int,
        team: int,
        position: str,
        *,
        price: int | None = None,
    ) -> DraftPick:
        position = position.upper()
        state = self._require_state()
        config = state.config

        # Snake: validate team matches expected
        if config.format == DraftFormat.SNAKE:
            expected = self._snake_team(state.current_pick, config.teams)
            if team != expected:
                msg = f"Wrong team: expected team {expected}, got team {team}"
                raise DraftError(msg)

        # Pool validation
        if player_id not in state.available_pool:
            msg = f"Player {player_id} is not in the available pool"
            raise DraftError(msg)

        # Position slot validation
        if position not in config.roster_slots:
            msg = f"{position!r} is not a valid roster slot"
            raise DraftError(msg)

        filled = sum(1 for p in state.team_rosters[team] if p.position == position)
        if filled >= config.roster_slots[position]:
            msg = f"Team {team} roster slot {position!r} is full ({filled}/{config.roster_slots[position]})"
            raise DraftError(msg)

        # Auction budget validation
        if config.format == DraftFormat.AUCTION:
            if price is None:
                msg = "A price is required for auction picks"
                raise DraftError(msg)
            if price < 1:
                msg = "Price must be at least $1"
                raise DraftError(msg)
            total_slots = sum(config.roster_slots.values())
            filled_slots = len(state.team_rosters[team])
            remaining_after = total_slots - filled_slots - 1
            reserve = remaining_after  # $1 per remaining slot
            if price > state.team_budgets[team]:
                msg = f"Price ${price} exceeds remaining budget ${state.team_budgets[team]}"
                raise DraftError(msg)
            if price > state.team_budgets[team] - reserve:
                msg = (
                    f"Price ${price} violates reserve requirement: "
                    f"must reserve ${reserve} for {remaining_after} remaining slot(s)"
                )
                raise DraftError(msg)

        player = state.available_pool[player_id]
        draft_pick = DraftPick(
            pick_number=state.current_pick,
            team=team,
            player_id=player_id,
            player_name=player.player_name,
            position=position,
            price=price,
        )

        # Mutate state
        self._removed_rows[player_id] = state.available_pool.pop(player_id)
        state.team_rosters[team].append(draft_pick)
        state.picks.append(draft_pick)
        state.current_pick += 1
        if config.format == DraftFormat.AUCTION and price is not None:
            state.team_budgets[team] -= price

        return draft_pick

    def undo(self) -> DraftPick:
        state = self._require_state()
        if not state.picks:
            msg = "There are no picks to undo"
            raise DraftError(msg)

        last = state.picks.pop()
        state.team_rosters[last.team].remove(last)
        state.current_pick -= 1

        # Restore player to pool
        row = self._removed_rows.pop(last.player_id)
        state.available_pool[last.player_id] = row

        # Restore auction budget
        if last.price is not None:
            state.team_budgets[last.team] += last.price

        return last

    def available(self, position: str | None = None) -> list[DraftBoardRow]:
        state = self._require_state()
        rows = list(state.available_pool.values())
        if position is not None:
            rows = [r for r in rows if r.position == position]
        return sorted(rows, key=lambda r: r.value, reverse=True)

    def my_roster(self) -> list[DraftPick]:
        state = self._require_state()
        return list(state.team_rosters[state.config.user_team])

    def my_needs(self) -> dict[str, int]:
        state = self._require_state()
        config = state.config
        needs: dict[str, int] = {}
        for pos, total in config.roster_slots.items():
            filled = sum(1 for p in state.team_rosters[config.user_team] if p.position == pos)
            remaining = total - filled
            if remaining > 0:
                needs[pos] = remaining
        return needs

    def team_on_clock(self) -> int:
        state = self._require_state()
        if state.config.format != DraftFormat.SNAKE:
            msg = f"team_on_clock is not applicable for {state.config.format.value} drafts"
            raise DraftError(msg)
        return self._snake_team(state.current_pick, state.config.teams)

    @staticmethod
    def _snake_team(pick_number: int, teams: int) -> int:
        """Determine which team picks at a given pick number in a snake draft.

        Pick 1..teams = round 1 (ascending), teams+1..2*teams = round 2 (descending), etc.
        """
        zero_based = pick_number - 1
        round_number = zero_based // teams  # 0-indexed round
        position_in_round = zero_based % teams  # 0-indexed position
        if round_number % 2 == 0:
            return position_in_round + 1
        return teams - position_in_round


def build_draft_roster_slots(league: LeagueSettings) -> dict[str, int]:
    slots: dict[str, int] = {}
    for pos, count in league.positions.items():
        if count > 0:
            slots[pos.upper()] = count
    if league.roster_util > 0:
        slots["UTIL"] = league.roster_util
    if league.roster_pitchers > 0:
        slots["P"] = league.roster_pitchers
    return slots
