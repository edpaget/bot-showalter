import queue
from io import StringIO

from rich.console import Console

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_recommendation import Recommendation
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.services.draft_session import DraftSession
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
)


def _make_player(player_id: int, name: str, position: str, value: float = 10.0) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=PlayerType.BATTER,
        position=position,
        value=value,
        category_z_scores={},
    )


def _make_yahoo_pick(**overrides: object) -> YahooDraftPick:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "season": 2026,
        "round": 1,
        "pick": 1,
        "team_key": "449.l.12345.t.1",
        "yahoo_player_key": "449.p.100",
        "player_id": 100,
        "player_name": "Mike Trout",
        "position": "OF",
    }
    defaults.update(overrides)
    return YahooDraftPick(**defaults)  # type: ignore[arg-type]


PLAYERS = [
    _make_player(100, "Mike Trout", "OF", 30.0),
    _make_player(200, "Aaron Judge", "OF", 28.0),
    _make_player(300, "Mookie Betts", "SS", 25.0),
    _make_player(400, "Freddie Freeman", "1B", 20.0),
]

SNAKE_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"OF": 3, "SS": 1, "1B": 1, "UTIL": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)

TEAM_MAP = {
    "449.l.12345.t.1": 1,
    "449.l.12345.t.2": 2,
}


def _fake_recommend(state: object, *, limit: int = 5) -> list[Recommendation]:
    return [
        Recommendation(
            player_id=200,
            player_name="Aaron Judge",
            position="OF",
            value=28.0,
            score=0.95,
            reason="best value",
        )
    ]


def _make_yahoo_session(
    commands: list[str],
    yahoo_queue: queue.Queue[YahooDraftPick] | None = None,
) -> tuple[StringIO, DraftSession]:
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, width=120, highlight=False)
    engine = DraftEngine()
    engine.start(PLAYERS, SNAKE_CONFIG)

    cmd_iter = iter(commands)

    def fake_input(_prompt: str = "") -> str:
        return next(cmd_iter)

    session = DraftSession(
        engine=engine,
        players=PLAYERS,
        console=test_console,
        recommend_fn=_fake_recommend,
        input_fn=fake_input,
        yahoo_pick_queue=yahoo_queue,
        team_map=TEAM_MAP,
    )
    return buf, session


class TestDraftSessionYahoo:
    def test_queue_drain_before_prompt(self) -> None:
        yahoo_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        yahoo_queue.put(_make_yahoo_pick(team_key="449.l.12345.t.1"))

        buf, session = _make_yahoo_session(["quit"], yahoo_queue)
        session.run()
        output = buf.getvalue()

        assert "Yahoo pick: Mike Trout" in output
        assert "team 1" in output

    def test_recommendations_shown_after_yahoo_picks(self) -> None:
        yahoo_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        yahoo_queue.put(_make_yahoo_pick(team_key="449.l.12345.t.1"))

        buf, session = _make_yahoo_session(["quit"], yahoo_queue)
        session.run()
        output = buf.getvalue()

        assert "Recommendations" in output
        assert "Aaron Judge" in output

    def test_undo_still_works(self) -> None:
        yahoo_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        yahoo_queue.put(_make_yahoo_pick(team_key="449.l.12345.t.1"))

        buf, session = _make_yahoo_session(["undo", "quit"], yahoo_queue)
        session.run()
        output = buf.getvalue()

        assert "Yahoo pick: Mike Trout" in output
        assert "Undid pick #1" in output

    def test_empty_queue_normal_behavior(self) -> None:
        yahoo_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        buf, session = _make_yahoo_session(["quit"], yahoo_queue)
        session.run()
        output = buf.getvalue()

        # Should not show any Yahoo pick output
        assert "Yahoo pick" not in output
        assert "Goodbye" in output

    def test_none_queue_is_noop(self) -> None:
        buf, session = _make_yahoo_session(["quit"], None)
        session.run()
        output = buf.getvalue()
        assert "Yahoo pick" not in output
        assert "Goodbye" in output

    def test_unmapped_pick_is_skipped(self) -> None:
        yahoo_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        yahoo_queue.put(_make_yahoo_pick(player_id=None))

        buf, session = _make_yahoo_session(["quit"], yahoo_queue)
        session.run()
        output = buf.getvalue()

        assert "skipped" in output.lower()
