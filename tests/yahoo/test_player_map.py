import sqlite3

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.yahoo_player import YahooPlayerMap
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper


class TestExactLookup:
    def test_returns_existing_mapping(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))

        map_repo.upsert(
            YahooPlayerMap(
                yahoo_player_key="449.p.12345",
                player_id=player_id,
                player_type="batter",
                yahoo_name="Mike Trout",
                yahoo_team="LAA",
                yahoo_positions="CF,LF",
            )
        )
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        result = mapper.resolve(
            {
                "player_key": "449.p.12345",
                "name": "Mike Trout",
                "editorial_team_abbr": "LAA",
                "eligible_positions": ["CF", "LF"],
            }
        )
        assert result is not None
        assert result.player_id == player_id
        assert result.yahoo_player_key == "449.p.12345"


class TestMLBAMFallback:
    def test_resolves_via_mlbam_id(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        result = mapper.resolve(
            {
                "player_key": "449.p.12345",
                "name": "Mike Trout",
                "editorial_team_abbr": "LAA",
                "eligible_positions": ["CF", "LF"],
                "player_id": 545361,
            }
        )
        assert result is not None
        assert result.player_id == player_id
        assert result.player_type == "batter"

        # Verify mapping was persisted
        persisted = map_repo.get_by_yahoo_key("449.p.12345")
        assert persisted is not None
        assert persisted.player_id == player_id


class TestNameFallback:
    def test_resolves_via_name_search(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        result = mapper.resolve(
            {
                "player_key": "449.p.12345",
                "name": "Mike Trout",
                "editorial_team_abbr": "LAA",
                "eligible_positions": ["CF", "LF"],
            }
        )
        assert result is not None
        assert result.player_id == player_id

        # Verify mapping was persisted
        persisted = map_repo.get_by_yahoo_key("449.p.12345")
        assert persisted is not None

    def test_resolves_last_name_only(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        result = mapper.resolve(
            {
                "player_key": "449.p.12345",
                "name": "Mike Trout",
                "editorial_team_abbr": "LAA",
                "eligible_positions": ["CF"],
            }
        )
        assert result is not None
        assert result.player_id == player_id


class TestTwoWayPlayer:
    def test_two_yahoo_keys_map_to_same_player(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_id = player_repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)

        batter_result = mapper.resolve(
            {
                "player_key": "449.p.11111",
                "name": "Shohei Ohtani",
                "editorial_team_abbr": "LAD",
                "eligible_positions": ["DH", "Util"],
                "player_id": 660271,
            }
        )
        pitcher_result = mapper.resolve(
            {
                "player_key": "449.p.22222",
                "name": "Shohei Ohtani",
                "editorial_team_abbr": "LAD",
                "eligible_positions": ["SP"],
                "player_id": 660271,
            }
        )

        assert batter_result is not None
        assert pitcher_result is not None
        assert batter_result.player_id == pitcher_result.player_id == player_id
        assert batter_result.player_type == "batter"
        assert pitcher_result.player_type == "pitcher"
        assert batter_result.yahoo_player_key != pitcher_result.yahoo_player_key


class TestUnresolved:
    def test_returns_none_and_logs_warning(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        mapper = YahooPlayerMapper(map_repo, player_repo)

        result = mapper.resolve(
            {
                "player_key": "449.p.99999",
                "name": "Unknown Player",
                "editorial_team_abbr": "???",
                "eligible_positions": ["DH"],
            }
        )
        assert result is None

        # Verify no mapping was persisted
        assert map_repo.get_by_yahoo_key("449.p.99999") is None


class TestInferPlayerType:
    def test_pitcher_positions(self) -> None:
        assert YahooPlayerMapper.infer_player_type(["SP"]) == "pitcher"
        assert YahooPlayerMapper.infer_player_type(["RP"]) == "pitcher"
        assert YahooPlayerMapper.infer_player_type(["P"]) == "pitcher"
        assert YahooPlayerMapper.infer_player_type(["SP", "RP"]) == "pitcher"

    def test_batter_positions(self) -> None:
        assert YahooPlayerMapper.infer_player_type(["CF", "LF"]) == "batter"
        assert YahooPlayerMapper.infer_player_type(["1B"]) == "batter"
        assert YahooPlayerMapper.infer_player_type(["DH"]) == "batter"
        assert YahooPlayerMapper.infer_player_type(["C", "1B"]) == "batter"

    def test_utility_only_is_batter(self) -> None:
        assert YahooPlayerMapper.infer_player_type(["Util"]) == "batter"
        assert YahooPlayerMapper.infer_player_type(["BN"]) == "batter"

    def test_empty_positions_is_batter(self) -> None:
        assert YahooPlayerMapper.infer_player_type([]) == "batter"
