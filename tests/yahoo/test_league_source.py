from typing import Any

from fantasy_baseball_manager.yahoo.league_source import YahooLeagueSource


class FakeYahooFantasyClient:
    """Fake client that returns canned responses."""

    def __init__(
        self,
        league_settings: dict[str, Any] | None = None,
        teams: dict[str, Any] | None = None,
    ) -> None:
        self._league_settings = league_settings or _DEFAULT_LEAGUE_SETTINGS
        self._teams = teams or _DEFAULT_TEAMS

    def get_league_settings(self, league_key: str) -> dict[str, Any]:
        return self._league_settings

    def get_teams(self, league_key: str) -> dict[str, Any]:
        return self._teams


_DEFAULT_LEAGUE_SETTINGS = {
    "fantasy_content": {
        "league": [
            {
                "league_key": "449.l.12345",
                "league_id": "12345",
                "name": "Test League",
                "season": "2026",
                "num_teams": 12,
                "game_code": "mlb",
                "renew": "422.l.54321",
            },
            {
                "settings": [
                    {
                        "draft_type": "live_standard_draft",
                        "uses_keeper": "1",
                    }
                ]
            },
        ]
    }
}

_DEFAULT_TEAMS = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
                "teams": {
                    "0": {
                        "team": [
                            [
                                {"team_key": "449.l.12345.t.1"},
                                {"team_id": "1"},
                                {"name": "Team One"},
                                [],
                                [],
                                [],
                                {"managers": [{"manager": {"nickname": "Alice", "is_current_login": "1"}}]},
                            ]
                        ]
                    },
                    "1": {
                        "team": [
                            [
                                {"team_key": "449.l.12345.t.2"},
                                {"team_id": "2"},
                                {"name": "Team Two"},
                                [],
                                [],
                                [],
                                {"managers": [{"manager": {"nickname": "Bob"}}]},
                            ]
                        ]
                    },
                    "count": 2,
                }
            },
        ]
    }
}


class TestYahooLeagueSource:
    def test_source_type(self) -> None:
        source = YahooLeagueSource(FakeYahooFantasyClient())  # type: ignore[arg-type]
        assert source.source_type == "yahoo_league"

    def test_fetch_returns_league_and_teams(self) -> None:
        source = YahooLeagueSource(FakeYahooFantasyClient())  # type: ignore[arg-type]
        result = source.fetch(league_key="449.l.12345", game_key="449")

        assert result["league"]["league_key"] == "449.l.12345"
        assert result["league"]["name"] == "Test League"
        assert result["league"]["season"] == 2026
        assert result["league"]["num_teams"] == 12
        assert result["league"]["draft_type"] == "live_standard_draft"
        assert result["league"]["is_keeper"] is True
        assert result["league"]["game_key"] == "449"

    def test_fetch_parses_teams(self) -> None:
        source = YahooLeagueSource(FakeYahooFantasyClient())  # type: ignore[arg-type]
        result = source.fetch(league_key="449.l.12345", game_key="449")

        teams = result["teams"]
        assert len(teams) == 2
        assert teams[0]["team_key"] == "449.l.12345.t.1"
        assert teams[0]["name"] == "Team One"
        assert teams[0]["manager_name"] == "Alice"
        assert teams[0]["is_owned_by_user"] is True
        assert teams[1]["team_key"] == "449.l.12345.t.2"
        assert teams[1]["name"] == "Team Two"
        assert teams[1]["manager_name"] == "Bob"
        assert teams[1]["is_owned_by_user"] is False

    def test_fetch_non_keeper_league(self) -> None:
        settings = {
            "fantasy_content": {
                "league": [
                    {
                        "league_key": "449.l.99999",
                        "league_id": "99999",
                        "name": "Redraft",
                        "season": "2026",
                        "num_teams": 10,
                        "game_code": "mlb",
                    },
                    {
                        "settings": [
                            {
                                "draft_type": "live_auction",
                                "uses_keeper": "0",
                            }
                        ]
                    },
                ]
            }
        }
        source = YahooLeagueSource(FakeYahooFantasyClient(league_settings=settings))  # type: ignore[arg-type]
        result = source.fetch(league_key="449.l.99999", game_key="449")
        assert result["league"]["is_keeper"] is False
        assert result["league"]["draft_type"] == "live_auction"
