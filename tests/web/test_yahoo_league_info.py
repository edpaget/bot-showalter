from typing import TYPE_CHECKING

from fantasy_baseball_manager.config_toml import WebConfig
from fantasy_baseball_manager.domain.yahoo_league import YahooLeagueInfo
from fantasy_baseball_manager.web.types import WebConfigType, YahooLeagueInfoType

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


_SAMPLE_INFO = YahooLeagueInfo(
    league_key="449.l.12345",
    league_name="Test Fantasy League",
    season=2026,
    num_teams=12,
    is_keeper=True,
    max_keepers=5,
    user_team_name="My Team",
)


class TestYahooLeagueInfoType:
    def test_from_domain_round_trip(self) -> None:
        result = YahooLeagueInfoType.from_domain(_SAMPLE_INFO)
        assert result.league_key == "449.l.12345"
        assert result.league_name == "Test Fantasy League"
        assert result.season == 2026
        assert result.num_teams == 12
        assert result.is_keeper is True
        assert result.max_keepers == 5
        assert result.user_team_name == "My Team"

    def test_from_domain_none_fields(self) -> None:
        info = YahooLeagueInfo(
            league_key="449.l.99",
            league_name="Redraft League",
            season=2026,
            num_teams=10,
            is_keeper=False,
            max_keepers=None,
            user_team_name=None,
        )
        result = YahooLeagueInfoType.from_domain(info)
        assert result.is_keeper is False
        assert result.max_keepers is None
        assert result.user_team_name is None


class TestWebConfigTypeWithYahooLeague:
    def test_with_yahoo_league_info(self) -> None:
        config = WebConfig()
        result = WebConfigType.from_domain(config, yahoo_league_info=_SAMPLE_INFO)
        assert result.yahoo_league is not None
        assert result.yahoo_league.league_name == "Test Fantasy League"

    def test_without_yahoo_league_info(self) -> None:
        config = WebConfig()
        result = WebConfigType.from_domain(config)
        assert result.yahoo_league is None


class TestYahooTeamsQuery:
    _QUERY = """
        query YahooTeams($leagueKey: String!) {
            yahooTeams(leagueKey: $leagueKey) {
                teamKey
                name
                managerName
                isOwnedByUser
            }
        }
    """

    def test_returns_teams(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooTeams"]
        assert len(data) == 2
        dynasty = next(t for t in data if t["teamKey"] == "449.l.12345.t.1")
        assert dynasty["name"] == "Dynasty Kings"
        assert dynasty["managerName"] == "Alice"
        assert dynasty["isOwnedByUser"] is True
        rival = next(t for t in data if t["teamKey"] == "449.l.12345.t.2")
        assert rival["isOwnedByUser"] is False

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestYahooStandingsQuery:
    _QUERY = """
        query YahooStandings($leagueKey: String!, $season: Int!) {
            yahooStandings(leagueKey: $leagueKey, season: $season) {
                teamKey
                teamName
                finalRank
                statValues
            }
        }
    """

    def test_returns_standings_sorted_by_rank(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooStandings"]
        assert len(data) == 2
        assert data[0]["finalRank"] == 1
        assert data[0]["teamName"] == "Dynasty Kings"
        assert data[0]["statValues"]["HR"] == 250.0
        assert data[1]["finalRank"] == 2
        assert data[1]["teamName"] == "Rival Squad"

    def test_returns_empty_for_unknown_season(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2020},
            },
        )
        assert response.status_code == 200
        assert response.json()["data"]["yahooStandings"] == []

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestWebConfigQueryYahooLeague:
    def test_returns_null_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        webConfig {
                            projections { system version }
                            valuations { system version }
                            yahooLeague {
                                leagueKey
                                leagueName
                                season
                                numTeams
                                isKeeper
                                maxKeepers
                                userTeamName
                            }
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["webConfig"]
        assert data["yahooLeague"] is None
