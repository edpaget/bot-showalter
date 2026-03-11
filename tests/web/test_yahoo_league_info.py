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
