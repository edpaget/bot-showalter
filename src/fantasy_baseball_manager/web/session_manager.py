from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.domain import DraftSessionPick, DraftSessionRecord, DraftSessionTrade, PickTrade
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftPick,
    PlayerProfileService,
    analyze_roster,
    build_draft_board,
    build_draft_roster_slots,
    compute_category_balance_scores,
    compute_pick_value_curve,
    evaluate_pick_trade,
    load_draft_from_db,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        DraftBoardRow,
        DraftTrade,
        LeagueSettings,
        PickTradeEvaluation,
        Valuation,
    )
    from fantasy_baseball_manager.repos import (
        ADPRepo,
        DraftSessionRepo,
        KeeperCostRepo,
        LeagueKeeperRepo,
        PlayerRepo,
        ProjectionRepo,
        ValuationRepo,
    )
    from fantasy_baseball_manager.services.draft_recommender import CategoryBalanceFn

# A keeper entry: (player_id, player_type_or_none)
KeeperKey = tuple[int, str | None]


class ValuationAdjuster(Protocol):
    def __call__(
        self, kept_keys: set[tuple[int, str | None]], valuations: list[Valuation], season: int
    ) -> list[Valuation]: ...


class KeeperCostDeriver(Protocol):
    def __call__(self, season: int, league_key: str) -> None: ...


@dataclass(frozen=True)
class DraftSessionSummary:
    record: DraftSessionRecord
    pick_count: int


def _is_kept(player_id: int, player_type: str, keeper_set: set[KeeperKey]) -> bool:
    """Check if a valuation is kept. None player_type in keeper matches all types for that player."""
    return any(player_id == kid and (ktype is None or ktype == player_type) for kid, ktype in keeper_set)


def _keeper_set_to_list(keeper_set: set[KeeperKey]) -> list[list[object]]:
    """Serialize keeper set to sorted list for JSON storage."""
    return sorted(([pid, ptype] for pid, ptype in keeper_set), key=lambda k: (k[0], k[1] or ""))


def _list_to_keeper_set(entries: list[list[object]]) -> set[KeeperKey]:
    """Deserialize keeper list from JSON storage to set."""
    return {(int(str(entry[0])), str(entry[1]) if entry[1] is not None else None) for entry in entries}


def _keeper_player_ids_only(keeper_set: set[KeeperKey]) -> set[int]:
    """Extract just player IDs from keeper set (for adjuster compatibility)."""
    return {pid for pid, _ in keeper_set}


def _keeper_picks_from_snapshot(
    snapshot: list[dict[str, object]],
    user_team: int,
    team_names: dict[int, str] | None,
) -> list[DraftPick]:
    """Convert keeper snapshot dicts to DraftPick entries for engine loading.

    Only includes keepers belonging to the user's team (matched by team_name).
    """
    user_team_name = team_names.get(user_team) if team_names else None
    if user_team_name is None:
        return []
    picks: list[DraftPick] = []
    for k in snapshot:
        # Skip keepers that belong to other teams
        if str(k.get("team_name", "")) != user_team_name:
            continue
        cost = k.get("cost")
        picks.append(
            DraftPick(
                pick_number=0,
                team=user_team,
                player_id=int(str(k["player_id"])),
                player_name=str(k["player_name"]),
                position=str(k["position"]),
                price=int(float(str(cost))) if cost is not None else None,
            )
        )
    return picks


class SessionManager:
    def __init__(
        self,
        *,
        session_repo: DraftSessionRepo,
        valuation_repo: ValuationRepo,
        player_repo: PlayerRepo,
        adp_repo: ADPRepo,
        player_profile_service: PlayerProfileService,
        league: LeagueSettings,
        adp_provider: str,
        valuation_adjuster: ValuationAdjuster | None = None,
        league_keeper_repo: LeagueKeeperRepo | None = None,
        projection_repo: ProjectionRepo | None = None,
        keeper_cost_deriver: KeeperCostDeriver | None = None,
        keeper_cost_repo: KeeperCostRepo | None = None,
        projection_system: str | None = None,
        projection_version: str | None = None,
    ) -> None:
        self._repo = session_repo
        self._valuation_repo = valuation_repo
        self._player_repo = player_repo
        self._adp_repo = adp_repo
        self._player_profile_service = player_profile_service
        self._league = league
        self._adp_provider = adp_provider
        self._valuation_adjuster = valuation_adjuster
        self._league_keeper_repo = league_keeper_repo
        self._projection_repo = projection_repo
        self._keeper_cost_deriver = keeper_cost_deriver
        self._keeper_cost_repo = keeper_cost_repo
        self._projection_system = projection_system
        self._projection_version = projection_version
        self._engines: dict[int, DraftEngine] = {}

    def start_session(
        self,
        season: int,
        *,
        system: str = "zar",
        version: str = "1.0",
        teams: int | None = None,
        user_team: int = 1,
        fmt: str = "snake",
        budget: int | None = None,
        keeper_player_ids: set[KeeperKey] | None = None,
        league_key: str | None = None,
        team_names: dict[int, str] | None = None,
        draft_order: list[int] | None = None,
    ) -> tuple[int, DraftEngine]:
        teams = teams or self._league.teams
        budget = budget if budget is not None else self._league.budget

        # Only use explicitly provided keeper IDs — no auto-loading or auto-derivation

        # Build keeper snapshot before filtering the pool
        keeper_snapshot = (
            self._build_keeper_snapshot(season, keeper_player_ids, system, version) if keeper_player_ids else None
        )

        players = self._build_player_pool(season, system, version, keeper_player_ids=keeper_player_ids)
        roster_slots = build_draft_roster_slots(self._league)
        config = DraftConfig(
            teams=teams,
            roster_slots=roster_slots,
            format=DraftFormat(fmt),
            user_team=user_team,
            season=season,
            budget=budget,
            draft_order=draft_order,
        )

        engine = DraftEngine()
        engine.start(players, config)

        # Pre-populate user's roster with keepers so my_roster()/my_needs() include them
        if keeper_snapshot:
            engine.load_keepers(_keeper_picks_from_snapshot(keeper_snapshot, user_team, team_names))

        now = datetime.now(tz=UTC).isoformat()
        record = DraftSessionRecord(
            league=self._league.name,
            season=season,
            teams=teams,
            format=fmt,
            user_team=user_team,
            roster_slots=dict(roster_slots),
            budget=budget,
            status="in_progress",
            created_at=now,
            updated_at=now,
            system=system,
            version=version,
            keeper_player_ids=_keeper_set_to_list(keeper_player_ids) if keeper_player_ids is not None else None,
            keeper_snapshot=keeper_snapshot,
            team_names=team_names,
            draft_order=draft_order,
        )
        session_id = self._repo.create_session(record)
        self._engines[session_id] = engine
        return session_id, engine

    def get_engine(self, session_id: int) -> DraftEngine:
        if session_id in self._engines:
            return self._engines[session_id]

        record = self._repo.load_session(session_id)
        if record is None:
            msg = f"Draft session {session_id} not found"
            raise ValueError(msg)

        keeper_ids = _list_to_keeper_set(record.keeper_player_ids) if record.keeper_player_ids else None
        players = self._build_player_pool(record.season, record.system, record.version, keeper_player_ids=keeper_ids)
        engine = load_draft_from_db(session_id, players, self._repo)

        # Re-load keepers into the engine so my_roster()/my_needs() include them
        if record.keeper_snapshot:
            engine.load_keepers(
                _keeper_picks_from_snapshot(record.keeper_snapshot, record.user_team, record.team_names)
            )

        self._engines[session_id] = engine
        return engine

    def pick(
        self,
        session_id: int,
        player_id: int,
        team: int,
        position: str,
        *,
        player_type: str | None = None,
        price: int | None = None,
    ) -> DraftPick:
        engine = self.get_engine(session_id)
        draft_pick = engine.pick(player_id, team, position, player_type=player_type, price=price)

        now = datetime.now(tz=UTC).isoformat()
        db_pick = DraftSessionPick(
            session_id=session_id,
            pick_number=draft_pick.pick_number,
            team=draft_pick.team,
            player_id=draft_pick.player_id,
            player_name=draft_pick.player_name,
            position=draft_pick.position,
            price=draft_pick.price,
        )
        self._repo.save_pick(db_pick)
        self._repo.update_timestamp(session_id, now)
        return draft_pick

    def undo(self, session_id: int) -> DraftPick:
        engine = self.get_engine(session_id)
        undone = engine.undo()

        now = datetime.now(tz=UTC).isoformat()
        self._repo.delete_pick(session_id, undone.pick_number)
        self._repo.update_timestamp(session_id, now)
        return undone

    def trade_picks(
        self,
        session_id: int,
        gives: list[int],
        receives: list[int],
        partner_team: int,
        *,
        team_a: int | None = None,
    ) -> DraftTrade:
        engine = self.get_engine(session_id)
        trade = engine.trade_picks(gives, receives, partner_team, team_a=team_a)
        now = datetime.now(tz=UTC).isoformat()
        db_trade = DraftSessionTrade(
            session_id=session_id,
            trade_number=len(engine.trades),
            team_a=trade.team_a,
            team_b=trade.team_b,
            team_a_gives=trade.team_a_gives,
            team_b_gives=trade.team_b_gives,
        )
        self._repo.save_trade(db_trade)
        self._repo.update_timestamp(session_id, now)
        return trade

    def undo_trade(self, session_id: int) -> DraftTrade:
        engine = self.get_engine(session_id)
        trade_number = len(engine.trades)
        removed = engine.undo_trade()
        now = datetime.now(tz=UTC).isoformat()
        self._repo.delete_trade(session_id, trade_number)
        self._repo.update_timestamp(session_id, now)
        return removed

    def evaluate_trade(self, session_id: int, gives: list[int], receives: list[int]) -> PickTradeEvaluation:
        record = self._repo.load_session(session_id)
        if record is None:
            msg = f"Draft session {session_id} not found"
            raise ValueError(msg)

        adp_list = self._adp_repo.get_by_season(record.season, provider=self._adp_provider)
        valuations = self._valuation_repo.get_by_season(record.season, system=record.system, version=record.version)
        player_ids = [v.player_id for v in valuations]
        players = self._player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        curve = compute_pick_value_curve(adp_list, valuations, self._league, player_names)
        trade = PickTrade(gives=gives, receives=receives)
        return evaluate_pick_trade(trade, curve)

    def persist_external_pick(self, session_id: int, draft_pick: DraftPick) -> None:
        """Persist a pick that was already applied to the engine (e.g., from Yahoo poller)."""
        now = datetime.now(tz=UTC).isoformat()
        db_pick = DraftSessionPick(
            session_id=session_id,
            pick_number=draft_pick.pick_number,
            team=draft_pick.team,
            player_id=draft_pick.player_id,
            player_name=draft_pick.player_name,
            position=draft_pick.position,
            price=draft_pick.price,
        )
        self._repo.save_pick(db_pick)
        self._repo.update_timestamp(session_id, now)

    def end_session(self, session_id: int) -> None:
        self._repo.update_status(session_id, "complete")
        self._engines.pop(session_id, None)

    def list_sessions(
        self,
        *,
        league: str | None = None,
        season: int | None = None,
        status: str | None = None,
    ) -> list[DraftSessionSummary]:
        records = self._repo.list_sessions(league=league, season=season)
        if status is not None:
            records = [r for r in records if r.status == status]
        return [DraftSessionSummary(record=r, pick_count=self._repo.count_picks(r.id or 0)) for r in records]

    def get_team_names(self, session_id: int) -> dict[int, str] | None:
        record = self._repo.load_session(session_id)
        if record is None:
            msg = f"Draft session {session_id} not found"
            raise ValueError(msg)
        return record.team_names

    def get_keepers(self, session_id: int) -> list[dict[str, object]]:
        record = self._repo.load_session(session_id)
        if record is None:
            msg = f"Draft session {session_id} not found"
            raise ValueError(msg)
        return record.keeper_snapshot or []

    def get_category_balance_fn(self, session_id: int) -> CategoryBalanceFn | None:
        """Return a category-balance scoring function for keeper sessions, None otherwise."""
        if self._projection_repo is None:
            return None
        record = self._repo.load_session(session_id)
        if record is None or not record.keeper_player_ids:
            return None
        projections = self._projection_repo.get_by_season(record.season, system=self._projection_system)
        if self._projection_version is not None:
            projections = [p for p in projections if p.version == self._projection_version]
        if not projections:
            return None
        league = self._league

        def _category_balance(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return compute_category_balance_scores(roster_ids, available_ids, projections, league)

        return _category_balance

    def get_weak_categories(self, session_id: int) -> list[str] | None:
        """Return weak category names for keeper sessions, None otherwise."""
        if self._projection_repo is None:
            return None
        record = self._repo.load_session(session_id)
        if record is None or not record.keeper_player_ids:
            return None
        projections = self._projection_repo.get_by_season(record.season, system=self._projection_system)
        if self._projection_version is not None:
            projections = [p for p in projections if p.version == self._projection_version]
        if not projections:
            return None
        keeper_ids = [int(str(entry[0])) for entry in record.keeper_player_ids]
        analysis = analyze_roster(keeper_ids, projections, self._league)
        return analysis.weakest_categories or None

    def _build_keeper_snapshot(
        self,
        season: int,
        keeper_player_ids: set[KeeperKey],
        system: str,
        version: str,
    ) -> list[dict[str, object]]:
        # Resolve player names/positions
        pids = list(_keeper_player_ids_only(keeper_player_ids))
        players = self._player_repo.get_by_ids(pids)
        player_map = {p.id: p for p in players if p.id is not None}

        # Get keeper team/cost/player_type from league keeper repo if available
        keeper_details: dict[KeeperKey, tuple[str, float | None]] = {}
        if self._league_keeper_repo is not None:
            league_keepers = self._league_keeper_repo.find_by_season_league(season, self._league.name)
            for lk in league_keepers:
                key = (lk.player_id, lk.player_type)
                if key in keeper_player_ids:
                    keeper_details[key] = (lk.team_name, lk.cost)
                elif lk.player_type is None:
                    # Untyped league keeper — match any keeper key with same player_id
                    for kk in keeper_player_ids:
                        if kk[0] == lk.player_id and kk not in keeper_details:
                            keeper_details[kk] = (lk.team_name, lk.cost)

        # Get valuations for value info — use the same system/version as the session
        valuations = self._valuation_repo.get_by_season(season, system=system, version=version)
        # Build val_map keyed by (player_id, player_type)
        val_map: dict[KeeperKey, Valuation] = {}
        for v in valuations:
            val_map[(v.player_id, v.player_type)] = v

        snapshot: list[dict[str, object]] = []
        for pid, ptype in sorted(keeper_player_ids, key=lambda k: (k[0], k[1] or "")):
            player = player_map.get(pid)
            val: Valuation | None = val_map.get((pid, ptype))
            # Fall back to any valuation for this player if typed lookup fails
            if val is None:
                val = next((v for v in valuations if v.player_id == pid), None)
            team_name, cost = keeper_details.get((pid, ptype), ("Unknown", None))
            # Fall back to untyped lookup for legacy keepers
            if team_name == "Unknown":
                team_name, cost = keeper_details.get((pid, None), ("Unknown", None))
            entry: dict[str, object] = {
                "player_id": pid,
                "player_name": f"{player.name_first} {player.name_last}" if player else f"Player {pid}",
                "position": val.position if val is not None else "UTIL",
                "player_type": ptype or (val.player_type if val is not None else None),
                "team_name": team_name,
                "cost": cost,
                "value": val.value if val is not None else 0.0,
            }
            snapshot.append(entry)
        return snapshot

    def _build_player_pool(
        self,
        season: int,
        system: str,
        version: str,
        *,
        keeper_player_ids: set[KeeperKey] | None = None,
    ) -> list[DraftBoardRow]:
        valuations = self._valuation_repo.get_by_season(season, system=system, version=version)

        # Apply keeper adjustments: re-value pool then exclude kept players
        if keeper_player_ids and self._valuation_adjuster is not None:
            valuations = self._valuation_adjuster(keeper_player_ids, valuations, season)
            valuations = [v for v in valuations if not _is_kept(v.player_id, v.player_type, keeper_player_ids)]

        player_ids = [v.player_id for v in valuations]
        players = self._player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = self._adp_repo.get_by_season(season, provider=self._adp_provider)
        profiles = self._player_profile_service.enrich_valuations(valuations, season)

        board = build_draft_board(
            valuations,
            self._league,
            player_names,
            adp=adp_list if adp_list else None,
            profiles=profiles,
        )
        return board.rows
