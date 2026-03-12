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
        LeagueKeeperRepo,
        PlayerRepo,
        ProjectionRepo,
        ValuationRepo,
    )
    from fantasy_baseball_manager.services.draft_recommender import CategoryBalanceFn


class ValuationAdjuster(Protocol):
    def __call__(self, kept_ids: set[int], valuations: list[Valuation], season: int) -> list[Valuation]: ...


@dataclass(frozen=True)
class DraftSessionSummary:
    record: DraftSessionRecord
    pick_count: int


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
        keeper_player_ids: set[int] | None = None,
    ) -> tuple[int, DraftEngine]:
        teams = teams or self._league.teams
        budget = budget if budget is not None else self._league.budget

        # Auto-load keepers from DB when not explicitly provided
        if keeper_player_ids is None and self._league_keeper_repo is not None:
            league_keepers = self._league_keeper_repo.find_by_season_league(season, self._league.name)
            if league_keepers:
                keeper_player_ids = {k.player_id for k in league_keepers}

        # Build keeper snapshot before filtering the pool
        keeper_snapshot = self._build_keeper_snapshot(season, keeper_player_ids) if keeper_player_ids else None

        players = self._build_player_pool(season, system, version, keeper_player_ids=keeper_player_ids)
        roster_slots = build_draft_roster_slots(self._league)
        config = DraftConfig(
            teams=teams,
            roster_slots=roster_slots,
            format=DraftFormat(fmt),
            user_team=user_team,
            season=season,
            budget=budget,
        )

        engine = DraftEngine()
        engine.start(players, config)

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
            keeper_player_ids=sorted(keeper_player_ids) if keeper_player_ids is not None else None,
            keeper_snapshot=keeper_snapshot,
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

        keeper_ids = set(record.keeper_player_ids) if record.keeper_player_ids else None
        players = self._build_player_pool(record.season, record.system, record.version, keeper_player_ids=keeper_ids)
        engine = load_draft_from_db(session_id, players, self._repo)
        self._engines[session_id] = engine
        return engine

    def pick(
        self,
        session_id: int,
        player_id: int,
        team: int,
        position: str,
        *,
        price: int | None = None,
    ) -> DraftPick:
        engine = self.get_engine(session_id)
        draft_pick = engine.pick(player_id, team, position, price=price)

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
        projections = self._projection_repo.get_by_season(record.season)
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
        projections = self._projection_repo.get_by_season(record.season)
        if not projections:
            return None
        analysis = analyze_roster(record.keeper_player_ids, projections, self._league)
        return analysis.weakest_categories or None

    def _build_keeper_snapshot(
        self,
        season: int,
        keeper_player_ids: set[int],
    ) -> list[dict[str, object]]:
        # Resolve player names/positions
        players = self._player_repo.get_by_ids(list(keeper_player_ids))
        player_map = {p.id: p for p in players if p.id is not None}

        # Get keeper team/cost from league keeper repo if available
        keeper_details: dict[int, tuple[str, float | None]] = {}
        if self._league_keeper_repo is not None:
            league_keepers = self._league_keeper_repo.find_by_season_league(season, self._league.name)
            for lk in league_keepers:
                if lk.player_id in keeper_player_ids:
                    keeper_details[lk.player_id] = (lk.team_name, lk.cost)

        # Get valuations for value info
        valuations = self._valuation_repo.get_by_season(season, system="zar", version="1.0")
        val_map = {v.player_id: v for v in valuations}

        snapshot: list[dict[str, object]] = []
        for pid in sorted(keeper_player_ids):
            player = player_map.get(pid)
            val = val_map.get(pid)
            team_name, cost = keeper_details.get(pid, ("Unknown", None))
            entry: dict[str, object] = {
                "player_id": pid,
                "player_name": f"{player.name_first} {player.name_last}" if player else f"Player {pid}",
                "position": val.position if val else "UTIL",
                "team_name": team_name,
                "cost": cost,
                "value": val.value if val else 0.0,
            }
            snapshot.append(entry)
        return snapshot

    def _build_player_pool(
        self,
        season: int,
        system: str,
        version: str,
        *,
        keeper_player_ids: set[int] | None = None,
    ) -> list[DraftBoardRow]:
        valuations = self._valuation_repo.get_by_season(season, system=system, version=version)

        # Apply keeper adjustments: re-value pool then exclude kept players
        if keeper_player_ids and self._valuation_adjuster is not None:
            valuations = self._valuation_adjuster(keeper_player_ids, valuations, season)
            valuations = [v for v in valuations if v.player_id not in keeper_player_ids]

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
