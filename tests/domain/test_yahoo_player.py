import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.yahoo_player import YahooPlayerMap


class TestYahooPlayerMap:
    def test_construct_with_required_fields(self) -> None:
        mapping = YahooPlayerMap(
            yahoo_player_key="449.p.12345",
            player_id=42,
            player_type=PlayerType.BATTER,
            yahoo_name="Mike Trout",
            yahoo_team="LAA",
            yahoo_positions="CF,LF",
        )
        assert mapping.yahoo_player_key == "449.p.12345"
        assert mapping.player_id == 42
        assert mapping.player_type == "batter"
        assert mapping.yahoo_name == "Mike Trout"
        assert mapping.yahoo_team == "LAA"
        assert mapping.yahoo_positions == "CF,LF"

    def test_optional_id_defaults_to_none(self) -> None:
        mapping = YahooPlayerMap(
            yahoo_player_key="449.p.12345",
            player_id=42,
            player_type=PlayerType.BATTER,
            yahoo_name="Mike Trout",
            yahoo_team="LAA",
            yahoo_positions="CF,LF",
        )
        assert mapping.id is None

    def test_construct_with_id(self) -> None:
        mapping = YahooPlayerMap(
            yahoo_player_key="449.p.12345",
            player_id=42,
            player_type=PlayerType.BATTER,
            yahoo_name="Mike Trout",
            yahoo_team="LAA",
            yahoo_positions="CF,LF",
            id=1,
        )
        assert mapping.id == 1

    def test_frozen(self) -> None:
        mapping = YahooPlayerMap(
            yahoo_player_key="449.p.12345",
            player_id=42,
            player_type=PlayerType.BATTER,
            yahoo_name="Mike Trout",
            yahoo_team="LAA",
            yahoo_positions="CF,LF",
        )
        with pytest.raises(AttributeError):
            mapping.yahoo_player_key = "other"  # type: ignore[misc]

    def test_two_way_player_same_player_id(self) -> None:
        batter = YahooPlayerMap(
            yahoo_player_key="449.p.11111",
            player_id=100,
            player_type=PlayerType.BATTER,
            yahoo_name="Shohei Ohtani",
            yahoo_team="LAD",
            yahoo_positions="DH",
        )
        pitcher = YahooPlayerMap(
            yahoo_player_key="449.p.22222",
            player_id=100,
            player_type=PlayerType.PITCHER,
            yahoo_name="Shohei Ohtani",
            yahoo_team="LAD",
            yahoo_positions="SP",
        )
        assert batter.player_id == pitcher.player_id
        assert batter.yahoo_player_key != pitcher.yahoo_player_key
        assert batter.player_type == "batter"
        assert pitcher.player_type == "pitcher"
