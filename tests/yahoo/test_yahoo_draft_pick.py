from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick


class TestYahooDraftPick:
    def test_construction(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=3,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=545361,
            player_name="Mike Trout",
            position="OF",
        )
        assert pick.league_key == "449.l.12345"
        assert pick.season == 2026
        assert pick.round == 1
        assert pick.pick == 3
        assert pick.team_key == "449.l.12345.t.1"
        assert pick.yahoo_player_key == "449.p.1234"
        assert pick.player_id == 545361
        assert pick.player_name == "Mike Trout"
        assert pick.position == "OF"

    def test_cost_defaults_to_none(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=545361,
            player_name="Mike Trout",
            position="OF",
        )
        assert pick.cost is None

    def test_id_defaults_to_none(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=545361,
            player_name="Mike Trout",
            position="OF",
        )
        assert pick.id is None

    def test_auction_pick_with_cost(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=545361,
            player_name="Mike Trout",
            position="OF",
            cost=55,
        )
        assert pick.cost == 55

    def test_unmapped_player_id_none(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.9999",
            player_id=None,
            player_name="Unknown Player",
            position="OF",
        )
        assert pick.player_id is None

    def test_frozen(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=545361,
            player_name="Mike Trout",
            position="OF",
        )
        with pytest.raises(FrozenInstanceError):
            pick.player_name = "Shohei Ohtani"  # type: ignore[misc]
