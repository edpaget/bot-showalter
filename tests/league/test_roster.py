from unittest.mock import MagicMock

from fantasy_baseball_manager.league.roster import YahooRosterSource


def _make_league(teams_data: dict[str, dict[str, str]], rosters: dict[str, list[dict[str, object]]]) -> MagicMock:
    league = MagicMock()
    league.teams.return_value = teams_data
    league.league_id = "mlb.l.12345"

    def to_team(team_key: str) -> MagicMock:
        team = MagicMock()
        team.roster.return_value = rosters.get(team_key, [])
        return team

    league.to_team.side_effect = to_team
    return league


class TestYahooRosterSource:
    def test_fetches_rosters_with_players(self) -> None:
        teams = {
            "mlb.l.12345.t.1": {"name": "Team Alpha"},
        }
        rosters = {
            "mlb.l.12345.t.1": [
                {
                    "player_id": 9876,
                    "name": "Mike Trout",
                    "position_type": "B",
                    "eligible_positions": ["CF", "OF", "Util"],
                },
                {
                    "player_id": 5432,
                    "name": "Shohei Ohtani",
                    "position_type": "B",
                    "eligible_positions": ["Util", "DH"],
                },
            ],
        }
        league = _make_league(teams, rosters)
        source = YahooRosterSource(league)
        result = source.fetch_rosters()

        assert result.league_key == "mlb.l.12345"
        assert len(result.teams) == 1
        team = result.teams[0]
        assert team.team_key == "mlb.l.12345.t.1"
        assert team.team_name == "Team Alpha"
        assert len(team.players) == 2

        trout = team.players[0]
        assert trout.yahoo_id == "9876"
        assert trout.name == "Mike Trout"
        assert trout.position_type == "B"
        assert trout.eligible_positions == ("CF", "OF", "Util")

    def test_multiple_teams(self) -> None:
        teams = {
            "mlb.l.12345.t.1": {"name": "Team Alpha"},
            "mlb.l.12345.t.2": {"name": "Team Beta"},
        }
        rosters = {
            "mlb.l.12345.t.1": [
                {"player_id": 100, "name": "Player A", "position_type": "B", "eligible_positions": ["1B"]},
            ],
            "mlb.l.12345.t.2": [
                {"player_id": 200, "name": "Player B", "position_type": "P", "eligible_positions": ["SP"]},
            ],
        }
        league = _make_league(teams, rosters)
        source = YahooRosterSource(league)
        result = source.fetch_rosters()

        assert len(result.teams) == 2
        names = {t.team_name for t in result.teams}
        assert names == {"Team Alpha", "Team Beta"}

    def test_empty_league(self) -> None:
        league = _make_league({}, {})
        source = YahooRosterSource(league)
        result = source.fetch_rosters()

        assert result.league_key == "mlb.l.12345"
        assert len(result.teams) == 0

    def test_team_with_no_players(self) -> None:
        teams = {"mlb.l.12345.t.1": {"name": "Empty Team"}}
        rosters = {"mlb.l.12345.t.1": []}
        league = _make_league(teams, rosters)
        source = YahooRosterSource(league)
        result = source.fetch_rosters()

        assert len(result.teams) == 1
        assert len(result.teams[0].players) == 0

    def test_player_id_stored_as_string(self) -> None:
        teams = {"mlb.l.12345.t.1": {"name": "Team"}}
        rosters = {
            "mlb.l.12345.t.1": [
                {"player_id": 42, "name": "Player", "position_type": "B", "eligible_positions": []},
            ],
        }
        league = _make_league(teams, rosters)
        source = YahooRosterSource(league)
        result = source.fetch_rosters()

        assert result.teams[0].players[0].yahoo_id == "42"
        assert isinstance(result.teams[0].players[0].yahoo_id, str)
