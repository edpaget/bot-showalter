from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import DraftSessionPick, DraftSessionRecord
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftPick,
    PlayerProfileService,
    build_draft_board,
    build_draft_roster_slots,
    load_draft_from_db,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow, LeagueSettings
    from fantasy_baseball_manager.repos import (
        ADPRepo,
        DraftSessionRepo,
        PlayerRepo,
        ValuationRepo,
    )


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
    ) -> None:
        self._repo = session_repo
        self._valuation_repo = valuation_repo
        self._player_repo = player_repo
        self._adp_repo = adp_repo
        self._player_profile_service = player_profile_service
        self._league = league
        self._adp_provider = adp_provider
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
    ) -> tuple[int, DraftEngine]:
        teams = teams or self._league.teams
        budget = budget if budget is not None else self._league.budget

        players = self._build_player_pool(season, system, version)
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

        players = self._build_player_pool(record.season, "zar", "1.0")
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

    def _build_player_pool(self, season: int, system: str, version: str) -> list[DraftBoardRow]:
        valuations = self._valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

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
