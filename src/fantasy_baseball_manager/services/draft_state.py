from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import DraftTrade, PlayerIdentity, PlayerType

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow, LeagueSettings


class DraftFormat(StrEnum):
    SNAKE = "snake"
    AUCTION = "auction"
    LIVE = "live"


class DraftError(Exception): ...


# Pool key: (player_id, player_type) — uniquely identifies a two-way player's side
type PoolKey = PlayerIdentity


@dataclass(frozen=True)
class DraftConfig:
    teams: int
    roster_slots: dict[str, int]
    format: DraftFormat
    user_team: int
    season: int
    budget: int = 0
    draft_order: list[int] | None = None


@dataclass(frozen=True)
class DraftPick:
    pick_number: int
    team: int
    player_id: int
    player_name: str
    position: str
    player_type: PlayerType | None = None
    price: int | None = None


@dataclass
class DraftState:
    config: DraftConfig
    picks: list[DraftPick]
    available_pool: dict[PoolKey, DraftBoardRow]
    team_rosters: dict[int, list[DraftPick]]
    team_budgets: dict[int, int]
    current_pick: int = 1
    pick_overrides: dict[int, int] = field(default_factory=dict)


class DraftEngine:
    _state: DraftState | None = None
    _removed_rows: dict[PoolKey, DraftBoardRow]

    def __init__(self) -> None:
        self._removed_rows = {}
        self._trades: list[DraftTrade] = []

    @property
    def trades(self) -> list[DraftTrade]:
        return list(self._trades)

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
        self._trades = []
        pool = {PlayerIdentity(p.player_id, PlayerType(p.player_type)): p for p in players}
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

    def load_keepers(self, keepers: list[DraftPick]) -> None:
        """Pre-populate the user's roster with keeper entries.

        Keepers are added to team_rosters but NOT to picks or current_pick,
        so they fill roster slots and appear in my_roster()/my_needs() without
        counting as draft picks.  Budget is reduced for auction formats.
        """
        state = self._require_state()
        for keeper in keepers:
            state.team_rosters[keeper.team].append(keeper)
            if state.config.format == DraftFormat.AUCTION and keeper.price is not None:
                state.team_budgets[keeper.team] -= keeper.price

    def pick(
        self,
        player_id: int,
        team: int,
        position: str,
        *,
        player_type: str | None = None,
        price: int | None = None,
    ) -> DraftPick:
        state = self._require_state()
        config = state.config

        # Snake: validate team matches expected
        if config.format == DraftFormat.SNAKE:
            expected = self.team_for_pick(state.current_pick)
            if team != expected:
                msg = f"Wrong team: expected team {expected}, got team {team}"
                raise DraftError(msg)

        # Resolve pool key — if player_type not given, find the player in pool
        pool_key: PoolKey | None = None
        if player_type is not None:
            pool_key = PlayerIdentity(player_id, PlayerType(player_type))
        else:
            for key in state.available_pool:
                if key.player_id == player_id:
                    pool_key = key
                    break
        if pool_key is None or pool_key not in state.available_pool:
            # Include available types for this player to aid debugging
            available_types = [key.player_type for key in state.available_pool if key.player_id == player_id]
            type_hint = f" (requested type={player_type!r}, available types={available_types})" if player_type else ""
            msg = f"Player {player_id} is not in the available pool{type_hint}"
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

        player = state.available_pool[pool_key]
        draft_pick = DraftPick(
            pick_number=state.current_pick,
            team=team,
            player_id=player_id,
            player_name=player.player_name,
            player_type=pool_key.player_type,
            position=position,
            price=price,
        )

        # Mutate state
        self._removed_rows[pool_key] = state.available_pool.pop(pool_key)
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

        # Restore player to pool — match by (player_id, player_type)
        restore_key: PoolKey | None = None
        if last.player_type is not None:
            restore_key = PlayerIdentity(last.player_id, last.player_type)
            if restore_key not in self._removed_rows:
                restore_key = None
        else:
            # Legacy pick without player_type — search by player_id
            for key in self._removed_rows:
                if key.player_id == last.player_id:
                    restore_key = key
                    break
        if restore_key is None:
            msg = f"Cannot restore player {last.player_id} — not found in removed rows"
            raise DraftError(msg)
        row = self._removed_rows.pop(restore_key)
        state.available_pool[restore_key] = row

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
        return self.team_for_pick(state.current_pick)

    def team_for_pick(self, pick_number: int) -> int:
        """Resolve which team owns a given pick, checking overrides first."""
        state = self._require_state()
        if pick_number in state.pick_overrides:
            return state.pick_overrides[pick_number]
        if state.config.draft_order is not None:
            return self._custom_snake_team(pick_number, state.config.draft_order)
        return self._snake_team(pick_number, state.config.teams)

    def trade_picks(
        self,
        gives: list[int],
        receives: list[int],
        partner_team: int,
        *,
        team_a: int | None = None,
    ) -> DraftTrade:
        """Execute a pick trade between two teams.

        Args:
            gives: Pick numbers that team_a gives away.
            receives: Pick numbers that team_a receives from partner_team.
            partner_team: The other team in the trade.
            team_a: The team giving picks. Defaults to user_team if not specified.
        """
        state = self._require_state()
        trading_team = team_a if team_a is not None else state.config.user_team

        if not gives:
            msg = "gives must not be empty"
            raise DraftError(msg)
        if not receives:
            msg = "receives must not be empty"
            raise DraftError(msg)

        # Validate ownership and usage
        for pick_num in gives:
            if pick_num < state.current_pick:
                msg = f"Pick {pick_num} has already been used"
                raise DraftError(msg)
            owner = self.team_for_pick(pick_num)
            if owner != trading_team:
                msg = f"Pick {pick_num} belongs to team {owner}, not team {trading_team}"
                raise DraftError(msg)

        for pick_num in receives:
            if pick_num < state.current_pick:
                msg = f"Pick {pick_num} has already been used"
                raise DraftError(msg)
            owner = self.team_for_pick(pick_num)
            if owner != partner_team:
                msg = f"Pick {pick_num} belongs to team {owner}, not partner team {partner_team}"
                raise DraftError(msg)

        # Apply overrides
        for pick_num in gives:
            state.pick_overrides[pick_num] = partner_team
        for pick_num in receives:
            state.pick_overrides[pick_num] = trading_team

        trade = DraftTrade(
            team_a=trading_team,
            team_b=partner_team,
            team_a_gives=list(gives),
            team_b_gives=list(receives),
        )
        self._trades.append(trade)
        return trade

    def undo_trade(self) -> DraftTrade:
        """Undo the most recent trade by rebuilding overrides from remaining trades."""
        if not self._trades:
            msg = "No trades to undo"
            raise DraftError(msg)

        removed = self._trades.pop()
        state = self._require_state()

        # Rebuild overrides from scratch by replaying remaining trades
        state.pick_overrides.clear()
        for trade in self._trades:
            for pick_num in trade.team_a_gives:
                state.pick_overrides[pick_num] = trade.team_b
            for pick_num in trade.team_b_gives:
                state.pick_overrides[pick_num] = trade.team_a

        return removed

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

    @staticmethod
    def _custom_snake_team(pick_number: int, draft_order: list[int]) -> int:
        """Snake draft using a custom team order instead of sequential 1..N."""
        teams = len(draft_order)
        zero_based = pick_number - 1
        round_number = zero_based // teams
        position_in_round = zero_based % teams
        if round_number % 2 == 0:
            return draft_order[position_in_round]
        return draft_order[teams - 1 - position_in_round]


def build_draft_roster_slots(league: LeagueSettings) -> dict[str, int]:
    slots: dict[str, int] = {}
    for pos, count in league.positions.items():
        if count > 0:
            slots[pos] = count
    if league.roster_util > 0:
        slots["UTIL"] = league.roster_util
    if league.pitcher_positions:
        for pos, count in league.pitcher_positions.items():
            if count > 0:
                slots[pos] = count
    elif league.roster_pitchers > 0:
        slots["P"] = league.roster_pitchers
    if league.roster_bench > 0:
        slots["BN"] = league.roster_bench
    return slots
