from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.domain.yahoo_league import YahooTeam
from fantasy_baseball_manager.services.draft_state import DraftConfig, DraftEngine, DraftFormat
from fantasy_baseball_manager.services.draft_translation import build_team_map, ingest_yahoo_pick


def _make_yahoo_pick(**overrides: object) -> YahooDraftPick:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "season": 2026,
        "round": 1,
        "pick": 1,
        "team_key": "449.l.12345.t.1",
        "yahoo_player_key": "449.p.1234",
        "player_id": 100,
        "player_name": "Mike Trout",
        "position": "OF",
    }
    defaults.update(overrides)
    return YahooDraftPick(**defaults)  # type: ignore[arg-type]


def _make_player(player_id: int, name: str, position: str, value: float = 10.0) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=1,
        player_type="batter",
        position=position,
        value=value,
        category_z_scores={},
    )


_SNAKE_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"OF": 3, "SP": 2, "C": 1, "UTIL": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)

_AUCTION_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"OF": 3, "SP": 2, "C": 1, "UTIL": 1},
    format=DraftFormat.AUCTION,
    user_team=1,
    season=2026,
    budget=260,
)

_TEAM_MAP = {
    "449.l.12345.t.1": 1,
    "449.l.12345.t.2": 2,
}


class TestBuildTeamMap:
    def test_maps_team_keys_to_ids(self) -> None:
        teams = [
            YahooTeam(
                team_key="449.l.12345.t.1",
                league_key="449.l.12345",
                team_id=1,
                name="Team A",
                manager_name="Alice",
                is_owned_by_user=True,
            ),
            YahooTeam(
                team_key="449.l.12345.t.2",
                league_key="449.l.12345",
                team_id=2,
                name="Team B",
                manager_name="Bob",
                is_owned_by_user=False,
            ),
        ]
        result = build_team_map(teams)
        assert result == {"449.l.12345.t.1": 1, "449.l.12345.t.2": 2}


class TestIngestYahooPick:
    def test_snake_pick_translation(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1")
        result = ingest_yahoo_pick(engine, yahoo_pick, _TEAM_MAP)

        assert result is not None
        assert result.player_id == 100
        assert result.team == 1
        assert result.position == "OF"

    def test_auction_pick_with_cost(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _AUCTION_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1", cost=55)
        result = ingest_yahoo_pick(engine, yahoo_pick, _TEAM_MAP)

        assert result is not None
        assert result.price == 55
        assert result.player_id == 100

    def test_unmapped_player_skipped(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(player_id=None)
        result = ingest_yahoo_pick(engine, yahoo_pick, _TEAM_MAP)

        assert result is None

    def test_player_not_in_pool_skipped(self) -> None:
        players = [_make_player(200, "Aaron Judge", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        # player_id 100 is not in the pool
        yahoo_pick = _make_yahoo_pick(player_id=100)
        result = ingest_yahoo_pick(engine, yahoo_pick, _TEAM_MAP)

        assert result is None

    def test_unknown_team_key_skipped(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.99999.t.99")
        result = ingest_yahoo_pick(engine, yahoo_pick, _TEAM_MAP)

        assert result is None
